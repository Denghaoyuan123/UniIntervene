"""HIL-VLM dataset: reads buffer_nttg pkl files and produces training samples."""

import glob
import json
import os
import pickle
import re
from typing import Any, Optional

import torch

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from memory.recover_target_mining import (
    RecoverTargetMiningConfig,
    compute_recover_mining_intervention_mask,
    detect_decline_window,
    extract_recovery_segment_actions,
    find_recover_target,
    _episode_values,
)
from episode_split import episode_stratum, stratified_episode_split

N_BINS       = 201          # number of RewardLabel discretization bins
FAIL_VALUE   = -500         # NTTG failure value (same as nttg_label_buffer.py)
SIDE_KEY     = "side_policy_256"
WRIST_KEY    = "wrist_1"
ALT_SIDE_KEY = "external"
ALT_WRIST_KEY = "wrist"
DEFAULT_INSTR = "Pick up the object and place it in the target location."
REQUIRED_VF_DIR_DEFAULT = None
ROUND2_MEMORY_BANK_DEFAULT = ""
DEFAULT_ROUND2_MEMORY_GOAL_AGGREGATE = "top1"
DEFAULT_OFFLINE_INTERVENTION_KEY = "vf_mining_intervention_label"


def _normalize_task_key(text: str) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_task_route_key(text: str) -> str:
    """Normalize task text/name into a canonical routing key, e.g. 'Fold the towel.' -> 'fold_towel'."""
    s = str(text or "").strip().lower()
    s = s.replace("_", " ")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "_")


def _discretize_raw_value(raw_value: float, n_bins: int = N_BINS) -> int:
    """raw_value in [FAIL_VALUE, 0] -> bin in [0, n_bins-1] (documentation/external tools only; the training RewardLabel uses the VF)."""
    v = float(raw_value)
    v_norm = np.clip(v / abs(FAIL_VALUE), -1.0, 0.0)
    return int(round((v_norm + 1.0) * (n_bins - 1)))


def _discretize_vf_pred(
    vf_value: float, n_bins: int, vmin: float, vmax: float
) -> int:
    """Clip a scalar VF value to [vmin, vmax] and map it linearly to [0, n_bins-1]."""
    v = float(vf_value)
    lo, hi = float(vmin), float(vmax)
    v = np.clip(v, lo, hi)
    if hi <= lo:
        t = 0.5
    else:
        t = (v - lo) / (hi - lo)
    return int(round(float(t) * (n_bins - 1)))


def _action_to_text(action: np.ndarray, a_min: np.ndarray, a_max: np.ndarray,
                    n_bins: int = 1000) -> str:
    """7-D action -> space-separated integer string in [0,1000]."""
    a = np.clip(action, a_min, a_max)
    a_norm = (a - a_min) / np.where(a_max - a_min > 1e-8, a_max - a_min, 1.0)
    a_int = np.round(a_norm * n_bins).astype(int).tolist()
    return " ".join(map(str, a_int))


def _state_to_text_from_stats(
    state: np.ndarray,
    s_mean: np.ndarray,
    s_std: np.ndarray,
    n_bins: int = 1000,
    state_clip_k: float = 4.0,
) -> str:
    """D-dim state -> z-score normalized and quantized to [0,n_bins] text."""
    s = np.asarray(state, dtype=np.float64).reshape(-1)
    mu = np.asarray(s_mean, dtype=np.float64).reshape(-1)
    sd = np.asarray(s_std, dtype=np.float64).reshape(-1)
    d = min(int(s.shape[0]), int(mu.shape[0]), int(sd.shape[0]))
    if d <= 0:
        return ""
    s = s[:d]
    mu = mu[:d]
    sd = np.maximum(sd[:d], 1e-8)
    z = (s - mu) / sd
    k = max(1e-6, float(state_clip_k))
    z = np.clip(z, -k, k)
    t = (z + k) / (2.0 * k)
    b = np.round(t * float(n_bins)).astype(int).tolist()
    return " ".join(map(str, b))


def _state_seq_to_text_from_stats(
    states: np.ndarray,
    s_mean: np.ndarray,
    s_std: np.ndarray,
    n_bins: int = 1000,
    state_clip_k: float = 4.0,
) -> str:
    lines = ["Trajectory:"]
    arr = np.asarray(states, dtype=np.float64)
    for k in range(int(arr.shape[0])):
        body = _state_to_text_from_stats(arr[k], s_mean, s_std, n_bins=n_bins, state_clip_k=state_clip_k)
        lines.append(f"step {k}: {body}")
    return "\n".join(lines)


def _transition_state_vector(tr: dict[str, Any]) -> Optional[np.ndarray]:
    """Get the state vector of one transition (prefers the last frame of observations.state)."""
    if not isinstance(tr, dict):
        return None
    obs = tr.get("observations")
    if not isinstance(obs, dict) or ("state" not in obs):
        return None
    try:
        st = np.asarray(obs["state"], dtype=np.float64)
    except Exception:
        return None
    if st.ndim == 2 and st.shape[0] >= 1:
        return np.asarray(st[-1], dtype=np.float64).reshape(-1)
    if st.ndim == 1:
        return np.asarray(st, dtype=np.float64).reshape(-1)
    return None


def _states_along_recovery_flats(
    ds: "HILVLMDataset",
    flats: list[int],
    state_dim: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    d = max(1, int(state_dim))
    for i in flats:
        tr = ds._get_transition_by_flat(int(i))
        if tr is None:
            continue
        sv = _transition_state_vector(tr)
        if sv is None or sv.shape[0] < d:
            continue
        rows.append(np.asarray(sv[:d], dtype=np.float64))
    if not rows:
        return np.zeros((0, d), dtype=np.float64)
    return np.stack(rows, axis=0).astype(np.float64)


def _poses_along_recovery_flats(
    ds: "HILVLMDataset",
    flats: list[int],
) -> np.ndarray:
    """Collect the per-frame tcp_pose+gripper (7-D) along the recovery trajectory."""
    rows: list[np.ndarray] = []
    for i in flats:
        tr = ds._get_transition_by_flat(int(i))
        if tr is None:
            continue
        sv = _transition_state_vector(tr)
        if sv is None:
            continue
        arr = np.asarray(sv, dtype=np.float32).reshape(-1)
        if arr.size == 9:
            s11 = np.concatenate([np.zeros(2, dtype=np.float32), arr])
        elif arr.size == 11:
            s11 = arr
        else:
            continue
        pose7 = np.concatenate([s11[[0]], s11[5:11]])  # gripper + tcp_pose
        rows.append(pose7)
    if not rows:
        return np.zeros((0, 7), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def _detect_plateau_window(
    values: np.ndarray,
    t: int,
    *,
    window: int = 8,
    flat_ratio: float = 0.60,
    delta: float = 0.03,
    amp_mult: float = 4.0,
) -> bool:
    """Plateau trigger: most steps in the recent window barely change and the overall amplitude is bounded."""
    start = max(0, int(t) - int(window) + 1)
    seg = np.asarray(values[start : int(t) + 1], dtype=np.float64)
    if seg.shape[0] < 2:
        return False
    diffs = np.abs(np.diff(seg))
    fr = float((diffs <= float(delta)).mean()) if diffs.size > 0 else 0.0
    amp = float(seg.max() - seg.min())
    return bool(fr >= float(flat_ratio) and amp <= float(delta) * float(amp_mult))


def format_trajectory_target_text(
    traj_rows: np.ndarray,
    a_min: np.ndarray,
    a_max: np.ndarray,
    n_bins: int = 1000,
) -> str:
    """Multi-step trajectory -> parseable Trajectory:\\nstep k: ... text."""
    lines = ["Trajectory:"]
    for k in range(int(traj_rows.shape[0])):
        lines.append(f"step {k}: {_action_to_text(traj_rows[k], a_min, a_max, n_bins)}")
    return "\n".join(lines)


def build_stitched_recovery_trajectory(
    episode: list,
    t_loc: int,
    target_index: int,
    action_key: str,
) -> tuple[Optional[torch.Tensor], dict[str, Any]]:
    """Within one episode, split [t_loc, B] at its midpoint into two segments and concatenate their action rows in order."""
    meta: dict[str, Any] = {
        "round2_stitch_blue_len": None,
        "round2_stitch_single_segment": True,
    }
    d = int(target_index - t_loc)
    if d <= 0:
        return None, meta
    split_local = t_loc + max(1, d // 2)
    if split_local >= target_index:
        traj = extract_recovery_segment_actions(
            episode, t_loc, target_index, action_key
        )
        return traj, meta
    a = extract_recovery_segment_actions(
        episode, t_loc, split_local, action_key
    )
    b = extract_recovery_segment_actions(
        episode, split_local + 1, target_index, action_key
    )
    if a is None or b is None:
        return None, meta
    traj = torch.cat([a, b], dim=0)
    meta["round2_stitch_blue_len"] = int(a.shape[0])
    meta["round2_stitch_single_segment"] = False
    return traj, meta


def build_retrieval_stitched_trajectory(
    current_action: np.ndarray,
    green_traj_rows: np.ndarray,
    bridge_steps: int = 2,
) -> tuple[Optional[np.ndarray], int]:
    """Retrieval-driven stitching: bridge segment (interpolation from the current action to the start of the retrieved segment) + retrieved recovery trajectory."""
    if green_traj_rows is None or int(green_traj_rows.shape[0]) <= 0:
        return None, 0
    green = np.asarray(green_traj_rows, dtype=np.float32)
    cur = np.asarray(current_action, dtype=np.float32).reshape(-1)
    g0 = np.asarray(green[0], dtype=np.float32).reshape(-1)
    n_bridge = max(0, int(bridge_steps))
    if n_bridge <= 0:
        return green.astype(np.float32), 0
    bridge_rows = [cur.astype(np.float32)]
    if n_bridge > 1:
        for i in range(1, n_bridge):
            alpha = float(i) / float(n_bridge)
            bridge_rows.append(((1.0 - alpha) * cur + alpha * g0).astype(np.float32))
    bridge = np.stack(bridge_rows, axis=0).astype(np.float32)
    stitched = np.concatenate([bridge, green], axis=0).astype(np.float32)
    return stitched, int(bridge.shape[0])


def _read_manifest_vf_dir(buffer_dir: str) -> tuple[Optional[str], Optional[str]]:
    man_path = os.path.join(buffer_dir, "vf_value_pred_manifest.json")
    if not os.path.isfile(man_path):
        return None, None
    try:
        with open(man_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None, man_path
    vf_dir = payload.get("vf_dir")
    if vf_dir is None:
        return None, man_path
    return str(vf_dir), man_path


def _load_round2_memory_bank_router(router_json_path: str) -> dict[str, str]:
    path = str(router_json_path or "").strip()
    if not path:
        return {}
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise ValueError(f"round2 memory bank router not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"round2 memory bank router must be a JSON object: {path}")
    out: dict[str, str] = {}
    for raw_key, raw_value in payload.items():
        if isinstance(raw_value, dict):
            bank_path = (
                raw_value.get("path")
                or raw_value.get("bank_path")
                or raw_value.get("memory_bank_path")
            )
        else:
            bank_path = raw_value
        key = _normalize_task_route_key(str(raw_key))
        bank_path = os.path.expanduser(str(bank_path or "").strip())
        if key and bank_path:
            out[key] = bank_path
    return out


def format_memory_goal_state_text(
    traj_rows: np.ndarray,
    a_min: np.ndarray,
    a_max: np.ndarray,
    n_bins: int = 1000,
) -> str:
    """Round2 stage one: last-frame action of the recovery segment -> a single line of bins (no multi-step Trajectory: header, to avoid confusion with the model output format)."""
    if traj_rows is None or int(traj_rows.shape[0]) == 0:
        return ""
    last = traj_rows[-1]
    body = _action_to_text(last, a_min, a_max, n_bins)
    return (
        "Target recovery goal state (7 action bins):\n"
        + body
    )


def format_memory_goal_state_text_from_state(
    traj_rows_state: np.ndarray,
    s_mean: np.ndarray,
    s_std: np.ndarray,
    n_bins: int = 1000,
    state_clip_k: float = 4.0,
) -> str:
    """Round2 stage one (state mode): last-frame state of the recovery segment -> a single line of bins."""
    if traj_rows_state is None or int(traj_rows_state.shape[0]) == 0:
        return ""
    last = np.asarray(traj_rows_state[-1], dtype=np.float64).reshape(-1)
    body = _state_to_text_from_stats(
        last, s_mean, s_std, n_bins=n_bins, state_clip_k=state_clip_k
    )
    dim = int(last.shape[0])
    return (
        f"Target recovery goal state ({dim} state bins, normalized):\n"
        + body
    )


def _obs_to_pil(obs: dict, side_key: str = SIDE_KEY, wrist_key: str = WRIST_KEY) -> list:
    """Get the two camera images from an observation dict, preferring the primary keys and falling back to external/wrist."""
    if not isinstance(obs, dict):
        return []
    best: list = []
    for pair in [
        (str(side_key), str(wrist_key)),
        (str(ALT_SIDE_KEY), str(ALT_WRIST_KEY)),
    ]:
        cur = []
        for key in pair:
            if key in obs:
                arr = np.array(obs[key])  # (T_stack, H, W, 3) or (H, W, 3)
                if arr.ndim == 4:
                    arr = arr[0]  # take frame 0
                cur.append(Image.fromarray(arr.astype(np.uint8)))
        if len(cur) > len(best):
            best = cur
        if len(best) == 2:
            break
    return best


class HILVLMDataset(Dataset):
    """Parameters"""

    def __init__(
        self,
        buffer_dir: str,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        action_min: np.ndarray = None,
        action_max: np.ndarray = None,
        instr: str = DEFAULT_INSTR,
        n_bins_rew: int = N_BINS,
        n_bins_act: int = 1000,
        intervene_only_action: bool = True,
        recover_mining_config: Optional[RecoverTargetMiningConfig] = None,
        vf_reward_clip_min: float = -1.5,
        vf_reward_clip_max: float = 1.5,
        required_vf_dir: Optional[str] = REQUIRED_VF_DIR_DEFAULT,
        round2_bridge_steps: int = 2,
        round2_retrieval_source: str = "memory_bank",
        round2_memory_bank_path: str = ROUND2_MEMORY_BANK_DEFAULT,
        round2_memory_bank_router_path: str = "",
        round2_task_route_mode: str = "strict",
        round2_memory_top_k: int = 5,
        round2_memory_goal_aggregate: str = DEFAULT_ROUND2_MEMORY_GOAL_AGGREGATE,
        round2_target_space: str = "state",
        state_dim: int = 3,
        state_clip_k: float = 4.0,
        state_mean: np.ndarray = None,
        state_std: np.ndarray = None,
        stage1_include_reward_label: bool = False,
        intervention_label_mode: str = "recover_mask",
        intervention_label_source: str = "auto",
        intervention_label_key: str = DEFAULT_OFFLINE_INTERVENTION_KEY,
        intervention_expand_steps: int = 0,
        task_aware_instr: bool = True,
        round2_task_aware_filter: bool = True,
        enable_round2: bool = True,
        use_external_wrist_keys: bool = False,
    ):
        self.instr = instr
        self.n_bins_rew = n_bins_rew
        self.n_bins_act = n_bins_act
        self.intervene_only_action = intervene_only_action
        self._recover_mining_config = recover_mining_config or RecoverTargetMiningConfig(
            value_key="vf_value_pred",
            value_normalization="minmax_episode",
            require_intervention_signal=False,
        )
        self._recover_mining_mask: list[bool] = []
        self._vf_reward_vmin = float(vf_reward_clip_min)
        self._vf_reward_vmax = float(vf_reward_clip_max)
        self._required_vf_dir = required_vf_dir
        self._round2_bridge_steps = max(0, int(round2_bridge_steps))
        self._round2_retrieval_source = str(round2_retrieval_source).strip().lower()
        if self._round2_retrieval_source != "memory_bank":
            raise ValueError("round2_retrieval_source only supports memory_bank (strict mode)")
        self._round2_memory_bank_path = str(round2_memory_bank_path)
        self._round2_memory_bank_default_path = str(round2_memory_bank_path)
        self._round2_memory_bank_router = _load_round2_memory_bank_router(round2_memory_bank_router_path)
        self._round2_memory_bank_router_path = str(round2_memory_bank_router_path)
        self._round2_task_route_mode = str(round2_task_route_mode).strip().lower() or "strict"
        if self._round2_task_route_mode not in ("strict", "fallback"):
            raise ValueError(f"unsupported round2_task_route_mode: {round2_task_route_mode}")
        self._round2_memory_bank_by_path: dict[str, Any] = {}
        self._round2_memory_top_k = max(1, int(round2_memory_top_k))
        self._round2_memory_goal_aggregate = str(round2_memory_goal_aggregate).strip().lower()
        if self._round2_memory_goal_aggregate not in ("top1", "topk_mean"):
            raise ValueError(f"unsupported round2_memory_goal_aggregate: {round2_memory_goal_aggregate}")
        self._round2_target_space = str(round2_target_space).strip().lower()
        if self._round2_target_space not in ("state", "action", "state11_pose"):
            raise ValueError(f"unsupported round2_target_space: {round2_target_space}")
        self._state_dim = max(1, int(state_dim))
        self._state_clip_k = max(1e-6, float(state_clip_k))
        self.s_mean = None if state_mean is None else np.asarray(state_mean, dtype=np.float64).reshape(-1)
        self.s_std = None if state_std is None else np.asarray(state_std, dtype=np.float64).reshape(-1)
        self._stage1_include_reward_label = bool(stage1_include_reward_label)
        self._intervention_label_mode = str(intervention_label_mode).strip().lower()
        if self._intervention_label_mode not in ("recover_mask", "decline_only"):
            raise ValueError(f"unsupported intervention_label_mode: {intervention_label_mode}")
        self._intervention_label_source = str(intervention_label_source).strip().lower()
        if self._intervention_label_source not in ("auto", "offline", "recover_mask"):
            raise ValueError(
                f"unsupported intervention_label_source: {intervention_label_source} "
                "(choices: auto/offline/recover_mask)"
            )
        self._intervention_label_key = str(intervention_label_key).strip() or DEFAULT_OFFLINE_INTERVENTION_KEY
        self._intervention_expand_steps = max(0, int(intervention_expand_steps))
        self._task_aware_instr = bool(task_aware_instr)
        self._round2_task_aware_filter = bool(round2_task_aware_filter)
        self._enable_round2 = bool(enable_round2)
        self._use_external_wrist_keys = bool(use_external_wrist_keys)
        self._side_key = ALT_SIDE_KEY if self._use_external_wrist_keys else SIDE_KEY
        self._wrist_key = ALT_WRIST_KEY if self._use_external_wrist_keys else WRIST_KEY
        self._round2_green_pool: list[dict[str, Any]] = []
        self._round2_memory_bank = None
        assert split in ("train", "val", "test")
        self._split = str(split)
        self._aux_transitions: list[dict] = []
        self._aux_episode_span_by_stem: dict[str, tuple[int, int]] = {}

        pkls = sorted(glob.glob(os.path.join(buffer_dir, "transitions_*_nttg_vf.pkl")))
        if not pkls:
            pkls = sorted(glob.glob(os.path.join(buffer_dir, "transitions_*_nttg.pkl")))
        if not pkls:
            pkls = sorted(glob.glob(os.path.join(buffer_dir, "*_episode_*.pkl")))
        assert pkls, f"no nttg pkl files found: {buffer_dir}"
        if self._required_vf_dir and any(p.endswith("_nttg_vf.pkl") for p in pkls):
            found_vf_dir, man_path = _read_manifest_vf_dir(buffer_dir)
            if not found_vf_dir:
                raise ValueError(
                    f"buffer={buffer_dir} has no readable vf_value_pred_manifest.json or it has no vf_dir; "
                    f"the required vf_dir is {self._required_vf_dir}"
                )
            got = os.path.realpath(found_vf_dir)
            req = os.path.realpath(self._required_vf_dir)
            if got != req:
                where = man_path or os.path.join(buffer_dir, "vf_value_pred_manifest.json")
                raise ValueError(
                    f"VF source mismatch: {where} records vf_dir={found_vf_dir}\n"
                    f"required vf_dir={self._required_vf_dir}"
                )

        def _task_from_pkl_path(fp: str) -> tuple[str, bool]:
            """Return (task_name, has_mining) for a pkl file."""
            task = "unknown"
            has_mining = False
            try:
                with open(fp, "rb") as f:
                    ep = pickle.load(f)
                if isinstance(ep, list) and ep:
                    task, has_mining = episode_stratum(ep, self._intervention_label_key)
            except Exception:
                pass
            return task, has_mining

        by_task_mining: dict[tuple[str, bool], list[str]] = {}
        for fp in pkls:
            tname, has_mining = _task_from_pkl_path(fp)
            by_task_mining.setdefault((tname, has_mining), []).append(fp)

        pkls = stratified_episode_split(
            by_task_mining,
            split=split,
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio),
            seed=int(seed),
        )

        self.transitions = []
        episode_spans: list[tuple[int, int]] = []
        episode_stems: list[str] = []
        for p in pkls:
            with open(p, "rb") as f:
                ep = pickle.load(f)
            if not isinstance(ep, list) or not ep:
                continue
            stem = os.path.splitext(os.path.basename(p))[0]
            start = len(self.transitions)
            self.transitions.extend(ep)
            end = len(self.transitions)
            episode_spans.append((start, end))
            episode_stems.append(stem)

        self.episode_spans: list[tuple[int, int]] = episode_spans
        self._episode_stems: list[str] = episode_stems
        self._episode_span_by_stem: dict[str, tuple[int, int]] = {
            str(st): (int(s0), int(e0)) for (s0, e0), st in zip(self.episode_spans, self._episode_stems)
        }
        self._task_name_flat: list[str] = [self._task_from_transition(tr) for tr in self.transitions]
        self._task_route_key_flat: list[str] = [self._resolve_round2_task_route_key(tr) for tr in self.transitions]

        for ti, tr in enumerate(self.transitions):
            if "vf_value_pred" not in tr:
                raise ValueError(
                    f"transition[{ti}] is missing vf_value_pred; run scripts/annotate_buffer_vf_value_pred.py "
                    f"to produce transitions_*_nttg_vf.pkl, or point buffer_dir at that directory."
                )
            try:
                float(tr["vf_value_pred"])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"transition[{ti}] vf_value_pred is not numeric: {tr.get('vf_value_pred')!r}"
                ) from e

        self._recover_mining_mask = [False] * len(self.transitions)
        offline_mask = [False] * len(self.transitions)
        offline_hits = 0
        for i, tr in enumerate(self.transitions):
            if self._intervention_label_key not in tr:
                continue
            offline_hits += 1
            offline_mask[i] = self._as_bool_label(tr.get(self._intervention_label_key))

        use_offline_mask = False
        if self._intervention_label_source == "offline":
            if offline_hits <= 0:
                raise ValueError(
                    f"intervention_label_source=offline but key={self._intervention_label_key!r} is not in the data"
                )
            use_offline_mask = True
        elif self._intervention_label_source == "auto" and offline_hits > 0:
            use_offline_mask = True

        if use_offline_mask:
            self._recover_mining_mask = list(offline_mask)
            if self._intervention_expand_steps > 0:
                for s, e in episode_spans:
                    m = self._recover_mining_mask[s:e]
                    if not any(bool(x) for x in m):
                        continue
                    expand = int(self._intervention_expand_steps)
                    m_exp = list(m)
                    for j, b in enumerate(m):
                        if not bool(b):
                            continue
                        lo = max(0, int(j) - expand)
                        hi = min(len(m_exp), int(j) + expand + 1)
                        for k in range(lo, hi):
                            m_exp[k] = True
                    self._recover_mining_mask[s:e] = m_exp
            src_desc = f"offline key={self._intervention_label_key} hits={offline_hits}/{len(self.transitions)}"
        else:
            for s, e in episode_spans:
                sub = self.transitions[s:e]
                if self._intervention_label_mode == "decline_only":
                    cfg = self._recover_mining_config
                    vals = _episode_values(sub, cfg)
                    inter_idx = []
                    for j, trj in enumerate(sub):
                        inter = bool(trj.get(cfg.intervention_key, False))
                        if cfg.labels_key in trj and trj.get(cfg.labels_key) == 2:
                            inter = True
                        if inter:
                            inter_idx.append(j)
                    m = [False] * len(sub)
                    for t_loc in range(len(sub)):
                        if cfg.require_intervention_signal:
                            ok = any(abs(int(t_loc) - int(k)) <= int(cfg.intervention_margin) for k in inter_idx)
                            if not ok:
                                continue
                        d = detect_decline_window(
                            values=vals,
                            t=t_loc,
                            window=cfg.decline_window,
                            slope_threshold=cfg.slope_threshold,
                            min_drop=cfg.min_drop,
                            min_down_ratio=cfg.min_down_ratio,
                        )
                        pflag = _detect_plateau_window(
                            vals,
                            t=t_loc,
                            window=8,
                            flat_ratio=0.60,
                            delta=0.03,
                            amp_mult=4.0,
                        )
                        if d is not None or pflag:
                            m[t_loc] = True
                else:
                    m = compute_recover_mining_intervention_mask(sub, self._recover_mining_config)
                if self._intervention_expand_steps > 0 and any(bool(x) for x in m):
                    expand = int(self._intervention_expand_steps)
                    m_exp = list(m)
                    for j, b in enumerate(m):
                        if not bool(b):
                            continue
                        lo = max(0, int(j) - expand)
                        hi = min(len(m_exp), int(j) + expand + 1)
                        for k in range(lo, hi):
                            m_exp[k] = True
                    m = m_exp
                if len(m) != e - s:
                    raise RuntimeError("recover_mining mask length does not match the episode")
                for j, b in enumerate(m):
                    self._recover_mining_mask[s + j] = b
            src_desc = f"online recover_mining mode={self._intervention_label_mode}"
        self._round2_green_pool = self._build_round2_green_pool() if self._enable_round2 else []
        self._round2_memory_bank_paths = self._iter_round2_bank_paths()

        if action_min is not None and action_max is not None:
            self.a_min = np.array(action_min, dtype=np.float64)
            self.a_max = np.array(action_max, dtype=np.float64)
        else:
            all_actions = np.stack([np.array(tr["actions"]) for tr in self.transitions])
            self.a_min = all_actions.min(axis=0)
            self.a_max = all_actions.max(axis=0)

        state_rows: list[np.ndarray] = []
        for tr in self.transitions:
            sv = _transition_state_vector(tr)
            if sv is None:
                continue
            if sv.shape[0] < self._state_dim:
                continue
            state_rows.append(np.asarray(sv[: self._state_dim], dtype=np.float64))
        if (self.s_mean is None or self.s_std is None) and state_rows:
            S = np.stack(state_rows, axis=0).astype(np.float64)
            self.s_mean = np.mean(S, axis=0)
            self.s_std = np.std(S, axis=0)
        if self.s_mean is None or self.s_std is None:
            self.s_mean = np.zeros((self._state_dim,), dtype=np.float64)
            self.s_std = np.ones((self._state_dim,), dtype=np.float64)
        self.s_mean = np.asarray(self.s_mean, dtype=np.float64).reshape(-1)
        self.s_std = np.maximum(np.asarray(self.s_std, dtype=np.float64).reshape(-1), 1e-8)

        n_int = sum(
            1 for i, tr in enumerate(self.transitions) if self._read_intervened(tr, i)
        )
        n_total = len(self.transitions)
        n_pool = len(self._round2_green_pool)
        extra = (
            f"recover_mining(value={self._recover_mining_config.value_key} "
            f"require_human={self._recover_mining_config.require_intervention_signal})  "
            f"RewardLabel=vf_value_pred clip=[{self._vf_reward_vmin},{self._vf_reward_vmax}]"
        )
        print(
            f"[HILVLMDataset] split={split}  pkls={len(pkls)}  transitions={n_total}  "
            f"{extra}  "
            f"intervene_only_action={intervene_only_action}  round2_green_pool={n_pool}  "
            f"round2_source={self._round2_retrieval_source} topk={self._round2_memory_top_k} agg={self._round2_memory_goal_aggregate}  "
            f"stage1_reward={self._stage1_include_reward_label}  "
            f"intervention_src={src_desc}  "
            f"task_aware_instr={self._task_aware_instr}  "
            f"round2_task_filter={self._round2_task_aware_filter}  "
            f"obs_keys=({self._side_key},{self._wrist_key})  "
            f"intervened={n_int} ({100*n_int/max(n_total,1):.1f}%)"
        )
        if split == "train":
            print(f"  action min: {self.a_min.round(4)}")
            print(f"  action max: {self.a_max.round(4)}")
            print(f"  state mean(dim={self._state_dim}): {self.s_mean.round(4)}")
            print(f"  state std (dim={self._state_dim}): {self.s_std.round(4)}")

    def _read_intervened(self, _tr: dict, idx: int) -> bool:
        """recover_mining mask (same failure frames as the recover_target_mining bank build); idx is a flat index."""
        return bool(self._recover_mining_mask[idx])

    @staticmethod
    def _as_bool_label(v: Any) -> bool:
        if v is None:
            return False
        try:
            return bool(float(v) > 0.5)
        except Exception:
            return bool(v)

    @staticmethod
    def _task_from_transition(tr: dict) -> str:
        if not isinstance(tr, dict):
            return ""
        for key in ("task_text", "task_prompt", "task_name", "task", "source_task"):
            v = tr.get(key)
            if v is not None and str(v).strip():
                return str(v).strip()
        return ""

    def _resolve_instr(self, tr: dict) -> str:
        if bool(self._task_aware_instr):
            t = self._task_from_transition(tr)
            if t:
                return t
        return str(self.instr)

    def _resolve_round2_task_route_key(self, tr: dict, fallback_text: str = "") -> str:
        if not isinstance(tr, dict):
            return _normalize_task_route_key(fallback_text)
        candidates = []
        if fallback_text:
            candidates.append(fallback_text)
        for key in ("task_name", "source_dataset", "task_text", "task_prompt", "task", "source_task"):
            v = tr.get(key)
            if v is not None and str(v).strip():
                candidates.append(str(v).strip())
        for cand in candidates:
            route_key = _normalize_task_route_key(cand)
            if route_key:
                return route_key
        return ""

    def _resolve_round2_memory_bank_path(self, tr: dict, anchor_task: str = "") -> tuple[str, str]:
        candidates: list[str] = []
        if anchor_task:
            candidates.append(str(anchor_task))
        base_task = self._task_from_transition(tr)
        if base_task:
            candidates.append(base_task)
        if isinstance(tr, dict):
            for key in ("task_name", "source_dataset", "task_text", "task_prompt", "task", "source_task"):
                v = tr.get(key)
                if v is not None and str(v).strip():
                    candidates.append(str(v).strip())
        for cand in candidates:
            route_key = _normalize_task_route_key(cand)
            if route_key and route_key in self._round2_memory_bank_router:
                return route_key, self._round2_memory_bank_router[route_key]
        fallback = self._round2_memory_bank_default_path or self._round2_memory_bank_path
        if fallback:
            if self._round2_memory_bank_router and self._round2_task_route_mode == "strict":
                known = ", ".join(sorted(self._round2_memory_bank_router.keys()))
                miss = candidates[0] if candidates else ""
                raise ValueError(
                    f"round2 memory bank route miss for task={miss!r}; known={known}"
                )
            return "", fallback
        if self._round2_memory_bank_router:
            known = ", ".join(sorted(self._round2_memory_bank_router.keys()))
            miss = candidates[0] if candidates else ""
            raise ValueError(
                f"round2 memory bank route miss for task={miss!r}; known={known}"
            )
        raise ValueError("round2 memory bank has no valid path configured")

    def _get_round2_memory_bank(self, bank_path: str):
        path = os.path.expanduser(str(bank_path or "").strip())
        if not path:
            return None
        bank = self._round2_memory_bank_by_path.get(path)
        if bank is not None:
            return bank
        if not os.path.isfile(path):
            raise ValueError(f"round2 memory bank not found: {path}")
        from memory.memory_bank import MemoryBank

        bank = MemoryBank.load(path, map_location="cpu")
        self._round2_memory_bank_by_path[path] = bank
        if self._round2_memory_bank is None:
            self._round2_memory_bank = bank
        return bank

    def _iter_round2_bank_paths(self) -> list[str]:
        paths: list[str] = []
        for p in self._round2_memory_bank_router.values():
            p = os.path.expanduser(str(p or "").strip())
            if p and p not in paths:
                paths.append(p)
        fallback = os.path.expanduser(str(self._round2_memory_bank_default_path or self._round2_memory_bank_path or "").strip())
        if fallback and fallback not in paths:
            paths.append(fallback)
        return paths

    def _flat_index_from_stem_and_local(self, episode_stem: str, local_idx: int) -> Optional[int]:
        key = str(episode_stem)
        span = self._episode_span_by_stem.get(key)
        if span is None:
            m = re.match(r"^(transitions_)(\\d+)(_nttg_vf)$", key)
            if m:
                pref, num_s, suff = m.groups()
                try:
                    num_i = int(num_s)
                    cand = f"{pref}{num_i:06d}{suff}"
                    span = self._episode_span_by_stem.get(cand)
                except Exception:
                    span = None
        if span is not None:
            s0, e0 = span
            li = int(local_idx)
            if li < 0 or (s0 + li) >= e0:
                return None
            return int(s0 + li)

        aux_span = self._aux_episode_span_by_stem.get(key)
        if aux_span is None:
            m = re.match(r"^(transitions_)(\\d+)(_nttg_vf)$", key)
            if m:
                pref, num_s, suff = m.groups()
                try:
                    num_i = int(num_s)
                    cand = f"{pref}{num_i:06d}{suff}"
                    aux_span = self._aux_episode_span_by_stem.get(cand)
                except Exception:
                    aux_span = None
        if aux_span is None:
            return None
        s0, e0 = aux_span
        li = int(local_idx)
        if li < 0 or (s0 + li) >= e0:
            return None
        return -(int(s0 + li) + 1)

    def register_aux_transitions(
        self,
        transitions: list,
        episode_spans: list,
        episode_stems: list,
    ) -> None:
        """Register auxiliary transitions from another split for cross-split bank lookups."""
        self._aux_transitions = list(transitions)
        self._aux_episode_span_by_stem = {
            str(st): (int(s0), int(e0))
            for (s0, e0), st in zip(episode_spans, episode_stems)
        }

    def rebuild_round2_green_pool(self) -> None:
        """Rebuild memory-bank green pool after aux transitions are registered."""
        if not bool(self._enable_round2):
            self._round2_green_pool = []
            return
        self._round2_green_pool = self._build_round2_green_pool()

    def _get_transition_by_flat(self, flat: int) -> Optional[dict]:
        """Resolve a flat index to a transition dict."""
        if flat >= 0:
            if flat < len(self.transitions):
                return self.transitions[flat]
            return None
        aux_flat = -(flat + 1)
        if aux_flat < len(self._aux_transitions):
            return self._aux_transitions[aux_flat]
        return None

    def _memory_item_recovery_flats(
        self,
        ep_stem: str,
        failure_index: int,
        target_index: int,
        traj_len: int,
        meta: Any,
    ) -> list[int]:
        """Build global flat indices for state-space round2 targets from memory item metadata."""
        out: list[int] = []
        md = meta if isinstance(meta, dict) else {}

        loc = md.get("recovery_local_indices")
        if isinstance(loc, (list, tuple)) and len(loc) > 0:
            for x in loc:
                try:
                    fi = self._flat_index_from_stem_and_local(ep_stem, int(x))
                except Exception:
                    fi = None
                if fi is not None:
                    out.append(int(fi))

        if not out:
            rf = md.get("recovery_flats")
            if isinstance(rf, (list, tuple)) and len(rf) > 0:
                for x in rf:
                    try:
                        xi = int(x)
                    except Exception:
                        continue
                    if 0 <= xi < len(self.transitions):
                        out.append(xi)

        if not out:
            s0 = int(failure_index)
            t0 = int(target_index)
            loc_idx: list[int] = []
            if s0 >= 0 and t0 >= s0:
                loc_idx = list(range(s0, t0 + 1))
            elif s0 >= 0 and int(traj_len) > 0:
                loc_idx = list(range(s0, s0 + int(traj_len)))
            for li in loc_idx:
                fi = self._flat_index_from_stem_and_local(ep_stem, int(li))
                if fi is not None:
                    out.append(int(fi))

        if not out:
            tf = self._flat_index_from_stem_and_local(ep_stem, int(target_index))
            if tf is not None:
                out.append(int(tf))

        dedup: list[int] = []
        seen = set()
        for x in out:
            if x in seen:
                continue
            dedup.append(int(x))
            seen.add(int(x))
        return dedup

    def _warp_traj_endpoints(self, traj_rows: np.ndarray, start_override: Optional[np.ndarray], goal_override: Optional[np.ndarray]) -> np.ndarray:
        tr = np.asarray(traj_rows, dtype=np.float32).copy()
        if tr.ndim != 2 or tr.shape[0] <= 0:
            return tr
        if start_override is None and goal_override is None:
            return tr
        tlen = int(tr.shape[0])
        s_ref = tr[0].copy()
        g_ref = tr[-1].copy()
        ds = np.zeros_like(s_ref) if start_override is None else (np.asarray(start_override, dtype=np.float32).reshape(-1) - s_ref)
        dg = np.zeros_like(g_ref) if goal_override is None else (np.asarray(goal_override, dtype=np.float32).reshape(-1) - g_ref)
        if tlen == 1:
            tr[0] = tr[0] + ds + dg
            return tr
        for i in range(tlen):
            a = float(i) / float(tlen - 1)
            tr[i] = tr[i] + (1.0 - a) * ds + a * dg
        return tr

    def _build_round2_green_pool_from_memory_bank(self) -> list[dict[str, Any]]:
        pool: list[dict[str, Any]] = []
        bank_paths = self._iter_round2_bank_paths()
        n_all = 0
        n_kept = 0
        for bank_path in bank_paths:
            bank = self._get_round2_memory_bank(bank_path)
            if bank is None:
                continue
            for item_idx, it in enumerate(bank.items):
                n_all += 1
                tr = getattr(it, "recovery_action_trajectory", None)
                if tr is None:
                    continue
                tr_np = np.asarray(tr.detach().cpu().numpy() if hasattr(tr, "detach") else tr, dtype=np.float32)
                if tr_np.ndim != 2 or tr_np.shape[0] <= 0:
                    continue
                ep_stem = str(getattr(it, "episode_id", ""))
                t_flat = self._flat_index_from_stem_and_local(ep_stem, int(getattr(it, "target_index", -1)))
                if t_flat is None:
                    continue
                a_flat = self._flat_index_from_stem_and_local(ep_stem, int(getattr(it, "failure_index", -1)))
                meta = getattr(it, "metadata", None) or {}
                imp = 0.0
                q_score = 1.0
                if isinstance(meta, dict):
                    try:
                        if meta.get("vf_delta") is not None:
                            imp = float(meta.get("vf_delta"))
                        elif meta.get("vf_recover_value") is not None and meta.get("vf_progress_value") is not None:
                            imp = float(meta.get("vf_recover_value")) - float(meta.get("vf_progress_value"))
                        elif meta.get("recover_value") is not None and meta.get("vf_ref_value") is not None:
                            imp = float(meta.get("recover_value")) - float(meta.get("vf_ref_value"))
                    except Exception:
                        imp = 0.0
                    try:
                        q_score = float(meta.get("quality_score", 1.0))
                    except Exception:
                        q_score = 1.0
                steps = max(1, int(tr_np.shape[0]))
                slope = float(imp / float(steps))
                pool.append(
                    {
                        "episode_stem": ep_stem,
                        "anchor_flat_idx": int(a_flat) if a_flat is not None else -1,
                        "target_flat_idx": int(t_flat),
                        "start_action": np.asarray(tr_np[0], dtype=np.float32),
                        "goal_action": np.asarray(tr_np[-1], dtype=np.float32),
                        "traj_rows": tr_np,
                        "recovery_flats": self._memory_item_recovery_flats(
                            ep_stem=ep_stem,
                            failure_index=int(getattr(it, "failure_index", -1)),
                            target_index=int(getattr(it, "target_index", -1)),
                            traj_len=int(tr_np.shape[0]),
                            meta=meta,
                        ),
                        "improvement": float(imp),
                        "slope_up": float(slope),
                        "quality_score": float(q_score),
                        "recover_steps": int(steps),
                        "task_tag": str(
                            (meta.get("task_text") if isinstance(meta, dict) else None)
                            or (meta.get("task_name") if isinstance(meta, dict) else None)
                            or (meta.get("task") if isinstance(meta, dict) else None)
                            or getattr(it, "task_text", None)
                            or (self._task_name_flat[t_flat] if (0 <= t_flat < len(self._task_name_flat)) else "")
                            or ""
                        ).strip(),
                        "memory_bank_path": bank_path,
                        "memory_bank_item_index": int(item_idx),
                    }
                )
                n_kept += 1
        print(f"[HILVLMDataset] round2 memory_bank={','.join(bank_paths) if bank_paths else self._round2_memory_bank_path} items={n_kept}/{n_all} (in-split available)")
        return pool

    @staticmethod
    def _to_tensor_1d(x: Any) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().float().reshape(-1)
        else:
            t = torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1)
        if int(t.numel()) <= 0:
            return None
        return t

    def _memory_item_to_green_dict(self, it: Any) -> dict[str, Any]:
        tr = getattr(it, "recovery_action_trajectory", None)
        if tr is None:
            raise ValueError("memory item is missing recovery_action_trajectory")
        tr_np = np.asarray(tr.detach().cpu().numpy() if hasattr(tr, "detach") else tr, dtype=np.float32)
        if tr_np.ndim != 2 or tr_np.shape[0] <= 0:
            raise ValueError("memory item recovery_action_trajectory has an invalid shape")
        ep_stem = str(getattr(it, "episode_id", ""))
        t_flat = self._flat_index_from_stem_and_local(ep_stem, int(getattr(it, "target_index", -1)))
        if t_flat is None:
            return None
        a_flat = self._flat_index_from_stem_and_local(ep_stem, int(getattr(it, "failure_index", -1)))
        meta = getattr(it, "metadata", None) or {}
        imp = 0.0
        q_score = 1.0
        if isinstance(meta, dict):
            if meta.get("vf_delta") is not None:
                imp = float(meta.get("vf_delta"))
            elif meta.get("vf_recover_value") is not None and meta.get("vf_progress_value") is not None:
                imp = float(meta.get("vf_recover_value")) - float(meta.get("vf_progress_value"))
            elif meta.get("recover_value") is not None and meta.get("vf_ref_value") is not None:
                imp = float(meta.get("recover_value")) - float(meta.get("vf_ref_value"))
            q_score = float(meta.get("quality_score", 1.0))
        steps = max(1, int(tr_np.shape[0]))
        slope = float(imp / float(steps))
        return {
            "episode_stem": ep_stem,
            "anchor_flat_idx": int(a_flat) if a_flat is not None else -1,
            "target_flat_idx": int(t_flat),
            "start_action": np.asarray(tr_np[0], dtype=np.float32),
            "goal_action": np.asarray(tr_np[-1], dtype=np.float32),
            "traj_rows": tr_np,
            "recovery_flats": self._memory_item_recovery_flats(
                ep_stem=ep_stem,
                failure_index=int(getattr(it, "failure_index", -1)),
                target_index=int(getattr(it, "target_index", -1)),
                traj_len=int(tr_np.shape[0]),
                meta=meta,
            ),
            "improvement": float(imp),
            "slope_up": float(slope),
            "quality_score": float(q_score),
            "recover_steps": int(steps),
            "task_tag": str(
                (meta.get("task_text") if isinstance(meta, dict) else None)
                or (meta.get("task_name") if isinstance(meta, dict) else None)
                or (meta.get("task") if isinstance(meta, dict) else None)
                or getattr(it, "task_text", None)
                or (self._task_name_flat[t_flat] if (0 <= t_flat < len(self._task_name_flat)) else "")
                or ""
            ).strip(),
        }

    def _retrieve_round2_green_from_memory_bank_strict(
        self,
        tr: dict[str, Any],
        anchor_stem: str,
        anchor_local_idx: int,
        anchor_task: str,
        current_vf: Optional[float],
        anchor_route_key: str = "",
    ) -> list[tuple[float, dict[str, Any]]]:
        route_key, bank_path = self._resolve_round2_memory_bank_path(tr, anchor_task=anchor_task or anchor_route_key)
        bank = self._get_round2_memory_bank(bank_path)
        if bank is None:
            raise RuntimeError("round2 memory bank is not loaded yet")

        fe = None
        fe_key = None
        failure_keys = (
            "failure_emb_1d",
            "failure_embedding",
            "failure_emb",
            "failure_feature",
            "failure_latent",
            "jepa_failure_embedding",
            "retrieval_failure_emb_1d",
        )
        for k in failure_keys:
            if k in tr:
                fe = self._to_tensor_1d(tr.get(k))
                if fe is not None:
                    fe_key = k
                    break
        if fe is None and isinstance(tr.get("metadata"), dict):
            meta = tr.get("metadata")
            for k in failure_keys:
                if k in meta:
                    fe = self._to_tensor_1d(meta.get(k))
                    if fe is not None:
                        fe_key = f"metadata.{k}"
                        break
        if fe is None:
            raise ValueError("current sample is missing failure_emb_1d (or an equivalent key)")

        if bank.key_tensor is not None:
            key_dim = int(bank.key_tensor.shape[1])
            fe_dim = int(fe.numel())
            if fe_dim != key_dim:
                raise ValueError(
                    f"failure feature dim {fe_dim} != memory key dim {key_dim}; "
                    f"source={fe_key}; check where the failure field comes from"
                )

        q = bank.query_builder.build_query(fe)
        if bank.key_tensor is not None and int(q.numel()) != int(bank.key_tensor.shape[1]):
            raise RuntimeError(f"memory query dim {int(q.numel())} != key dim {int(bank.key_tensor.shape[1])}")

        task_filter = anchor_task if (bool(self._round2_task_aware_filter) and anchor_task) else None
        rr = bank.retrieve(
            query=q,
            topk=max(1, int(self._round2_memory_top_k)),
            task_filter=task_filter,
            exclude_episode_id=anchor_stem,
            exclude_failure_index=int(anchor_local_idx),
        )

        out: list[tuple[float, dict[str, Any]]] = []
        idx_list = rr.indices[0].detach().cpu().tolist() if rr.indices is not None else []
        for rank, (item_idx, score, mem_it) in enumerate(zip(idx_list, rr.scores[0].detach().cpu().tolist(), rr.items[0])):
            g = self._memory_item_to_green_dict(mem_it)
            if g is None:
                continue
            if current_vf is not None:
                goal_flat = int(g.get("target_flat_idx", 0))
                goal_tr = self._get_transition_by_flat(goal_flat)
                if goal_tr is None:
                    continue
                goal_val = goal_tr.get("vf_value_pred") if isinstance(goal_tr, dict) else None
                goal_vf = float(goal_val)
                if (not np.isfinite(goal_vf)) or (float(goal_vf) <= float(current_vf)):
                    continue
            g["memory_bank_path"] = bank_path
            g["memory_bank_route_key"] = route_key
            g["memory_bank_item_index"] = int(item_idx)
            g["memory_bank_rank"] = int(rank)
            g["memory_bank_score"] = float(score)
            g["memory_bank_task_filter"] = str(task_filter or "")
            g["memory_bank_item_task_text"] = str(getattr(mem_it, "task_text", "") or "")
            g["memory_bank_item_task_id"] = str(getattr(mem_it, "task_id", "") or "")
            g["memory_bank_item_episode_id"] = str(getattr(mem_it, "episode_id", "") or "")
            g["memory_bank_item_failure_index"] = int(getattr(mem_it, "failure_index", -1))
            g["memory_bank_item_target_index"] = int(getattr(mem_it, "target_index", -1))
            out.append((-float(score), g))
        return out

    def _build_round2_green_pool(self) -> list[dict[str, Any]]:
        return self._build_round2_green_pool_from_memory_bank()

    def _build_round2_green_pool_from_dataset(self) -> list[dict[str, Any]]:
        """Pre-build the pool of retrievable recovery segments from recover-mining anchors in this dataset."""
        pool: list[dict[str, Any]] = []
        cfg = self._recover_mining_config
        for (s, e), stem in zip(self.episode_spans, self._episode_stems):
            ep = self.transitions[s:e]
            values = _episode_values(ep, cfg)
            for t_loc in range(len(ep)):
                flat_idx = s + t_loc
                if not self._recover_mining_mask[flat_idx]:
                    continue
                decline = detect_decline_window(
                    values=values,
                    t=t_loc,
                    window=cfg.decline_window,
                    slope_threshold=cfg.slope_threshold,
                    min_drop=cfg.min_drop,
                    min_down_ratio=cfg.min_down_ratio,
                )
                if decline is None:
                    continue
                recover = find_recover_target(
                    values=values,
                    anchor_index=t_loc,
                    decline_start=decline.start_index,
                    recover_tolerance=cfg.recover_tolerance,
                    pre_drop_lookback=cfg.pre_drop_lookback,
                    fallback_to_best_future=cfg.fallback_to_best_future,
                )
                if recover is None or recover.target_index <= t_loc:
                    continue
                traj = extract_recovery_segment_actions(ep, t_loc, recover.target_index, cfg.action_key)
                if traj is None or int(traj.shape[0]) <= 0:
                    continue
                traj_np = traj.detach().cpu().numpy().astype(np.float32)
                improvement = float(recover.target_value - values[t_loc])
                steps = max(1, int(recover.target_index - t_loc))
                slope_up = float(improvement / float(steps))
                pool.append(
                    {
                        "episode_stem": stem,
                        "anchor_flat_idx": int(flat_idx),
                        "target_flat_idx": int(s + recover.target_index),
                        "start_action": np.asarray(traj_np[0], dtype=np.float32),
                        "traj_rows": traj_np,
                        "improvement": improvement,
                        "slope_up": slope_up,
                        "quality_score": float(1.0 + max(0.0, improvement) + 2.0 * max(0.0, slope_up)),
                        "recover_steps": int(steps),
                        "task_tag": str(self._task_name_flat[flat_idx]).strip() if 0 <= flat_idx < len(self._task_name_flat) else "",
                    }
                )
        return pool

    def _retrieve_round2_green(
        self,
        current_action: np.ndarray,
        anchor_flat_idx: int,
        tr: Optional[dict[str, Any]] = None,
        current_vf: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Retrieve a recovery target using the encoded observation-instruction context."""
        if self._round2_retrieval_source != "memory_bank":
            raise ValueError("round2_retrieval_source must be memory_bank")
        if tr is None:
            raise ValueError("_retrieve_round2_green needs the current transition to build the query")
        if self._round2_memory_bank is None:
            raise RuntimeError("round2 memory bank is not loaded")

        anchor_stem, anchor_local_idx = self.episode_stem_and_local_index(anchor_flat_idx)
        anchor_task = ""
        if 0 <= int(anchor_flat_idx) < len(self._task_name_flat):
            anchor_task = str(self._task_name_flat[int(anchor_flat_idx)]).strip()

        picked = self._retrieve_round2_green_from_memory_bank_strict(
            tr=tr,
            anchor_stem=anchor_stem,
            anchor_local_idx=anchor_local_idx,
            anchor_task=anchor_task,
            current_vf=current_vf,
        )
        if not picked:
            return None  # all candidates were cross-split or filtered; caller treats as no green
        if self._round2_memory_goal_aggregate == "top1":
            picked = picked[:1]

        top1 = dict(picked[0][1])
        if self._round2_target_space == "state11_pose":
            rf = [int(x) for x in (top1.get("recovery_flats") or [])]
            traj = _poses_along_recovery_flats(self, rf).astype(np.float32, copy=False)
        else:
            traj = np.asarray(top1.get("traj_rows"), dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] <= 0:
            raise RuntimeError("retrieved traj_rows is invalid")

        start_override = None
        goal_override = None
        if self._round2_memory_goal_aggregate == "topk_mean" and len(picked) > 1:
            if self._round2_target_space == "state11_pose":
                starts = []
                goals = []
                for _, it in picked:
                    rf_it = [int(x) for x in (it.get("recovery_flats") or [])]
                    pose_it = _poses_along_recovery_flats(self, rf_it).astype(np.float32, copy=False)
                    if pose_it.ndim != 2 or pose_it.shape[0] <= 0:
                        continue
                    starts.append(np.asarray(pose_it[0], dtype=np.float32))
                    goals.append(np.asarray(pose_it[-1], dtype=np.float32))
            else:
                starts = [np.asarray(it["start_action"], dtype=np.float32) for _, it in picked]
                goals = [np.asarray(it.get("goal_action", np.asarray(it["traj_rows"], dtype=np.float32)[-1]), dtype=np.float32) for _, it in picked]
            if starts and goals:
                start_override = np.mean(np.stack(starts, axis=0), axis=0).astype(np.float32)
                goal_override = np.mean(np.stack(goals, axis=0), axis=0).astype(np.float32)
        warped = self._warp_traj_endpoints(traj, start_override=start_override, goal_override=goal_override)
        top1["traj_rows"] = warped
        top1["start_action"] = np.asarray(warped[0], dtype=np.float32)
        top1["goal_action"] = np.asarray(warped[-1], dtype=np.float32)
        top1["memory_topk_used"] = int(len(picked))
        top1["memory_goal_aggregate"] = str(self._round2_memory_goal_aggregate)
        top1["memory_goal_synthetic"] = bool(self._round2_memory_goal_aggregate == "topk_mean" and len(picked) > 1)
        return top1

    def __len__(self):
        return len(self.transitions)

    def intervention_label_at(self, idx: int) -> bool:
        """Same intervened flag as __getitem__, so eval statistics do not need to rebuild the target."""
        return self._read_intervened(self.transitions[idx], idx)

    def _episode_bounds_for_flat_idx(self, idx: int) -> tuple[int, int]:
        for s, e in self.episode_spans:
            if s <= idx < e:
                return s, e
        raise RuntimeError(f"flat idx {idx} is not inside any episode")

    def episode_stem_and_local_index(self, flat_idx: int) -> tuple[str, int]:
        """Aligned with the memory bank episode_id (path.stem at build time) and failure_index=t."""
        for (s, e), stem in zip(self.episode_spans, self._episode_stems):
            if s <= flat_idx < e:
                return stem, int(flat_idx - s)
        raise RuntimeError(f"flat idx {flat_idx} is not inside any episode")

    def __getitem__(self, idx):
        tr = self.transitions[idx]

        imgs = _obs_to_pil(tr["observations"], self._side_key, self._wrist_key)

        intervened = self._read_intervened(tr, idx)
        intervention_txt = "yes" if intervened else "no"

        action = np.array(tr["actions"], dtype=np.float32)
        current_action_text = (
            f"Current proposed action (7 bins in [0,{self.n_bins_act}]): "
            + _action_to_text(action, self.a_min, self.a_max, self.n_bins_act)
        )
        state_v = _transition_state_vector(tr)
        if state_v is None or state_v.shape[0] < self._state_dim:
            state_v = np.zeros((self._state_dim,), dtype=np.float64)
        else:
            state_v = np.asarray(state_v[: self._state_dim], dtype=np.float64)
        current_state_text = (
            f"Current normalized state ({self._state_dim} bins in [0,{self.n_bins_act}]): "
            + _state_to_text_from_stats(
                state_v,
                self.s_mean,
                self.s_std,
                n_bins=self.n_bins_act,
                state_clip_k=self._state_clip_k,
            )
        )

        if "vf_value_pred" not in tr:
            raise ValueError(
                f"sample idx={idx} is missing vf_value_pred (use the annotate_buffer_vf_value_pred output)"
            )
        reward_bin = _discretize_vf_pred(
            tr["vf_value_pred"],
            self.n_bins_rew,
            self._vf_reward_vmin,
            self._vf_reward_vmax,
        )

        if self._stage1_include_reward_label:
            target = f"Intervention: {intervention_txt}\nRewardLabel: {reward_bin}"
        else:
            target = f"Intervention: {intervention_txt}"

        has_round2 = False
        round2_target: Optional[str] = None
        round2_action_traj: Optional[np.ndarray] = None
        round2_images: Optional[list] = None
        round2_goal_user_block = ""
        stitch_blue_len: Optional[int] = None
        stitch_single = True
        round2_improvement = 0.0
        round2_slope_up = 0.0
        round2_goal_vf_delta = 0.0
        round2_quality_score = 1.0
        poses_np: Optional[np.ndarray] = None
        round2_memory_bank_path = ""
        round2_memory_route_key = ""
        round2_memory_item_index = -1
        round2_memory_rank = -1
        round2_memory_score = float("nan")
        round2_memory_task_filter = ""
        round2_memory_item_task_text = ""
        round2_memory_item_task_id = ""
        round2_memory_episode_id = ""
        round2_memory_failure_index = -1
        round2_memory_target_index = -1
        round2_memory_query_task = str(self._task_name_flat[idx]).strip() if 0 <= idx < len(self._task_name_flat) else ""
        round2_memory_query_route_key = str(self._task_route_key_flat[idx]).strip() if 0 <= idx < len(self._task_route_key_flat) else ""
        if intervened and self._enable_round2:
            current_vf = float(tr.get("vf_value_pred"))
            green = self._retrieve_round2_green(action, idx, tr=tr, current_vf=current_vf)
            if green is not None:
                if self._round2_target_space == "state11_pose":
                    rf = green.get("recovery_flats") or []
                    poses_np = _poses_along_recovery_flats(self, list(rf))
                    stitched_np = poses_np
                    blue_len = 0
                else:
                    stitched_np, blue_len = build_retrieval_stitched_trajectory(
                        current_action=action,
                        green_traj_rows=green["traj_rows"],
                        bridge_steps=self._round2_bridge_steps,
                    )
                if stitched_np is not None and int(stitched_np.shape[0]) > 0:
                    if self._round2_target_space == "state11_pose":
                        if int(poses_np.shape[0]) > 0:
                            round2_target = "__pose7__"  # sentinel; the actual data is in round2_pose_traj
                            round2_goal_user_block = ""
                        else:
                            poses_np = None
                            round2_target = None
                    elif self._round2_target_space == "state":
                        poses_np = None
                        rf = green.get("recovery_flats") or []
                        states_np = _states_along_recovery_flats(self, list(rf), self._state_dim)
                        if int(states_np.shape[0]) > 0:
                            round2_target = _state_seq_to_text_from_stats(
                                states_np,
                                self.s_mean,
                                self.s_std,
                                n_bins=self.n_bins_act,
                                state_clip_k=self._state_clip_k,
                            )
                            round2_goal_user_block = format_memory_goal_state_text_from_state(
                                states_np,
                                self.s_mean,
                                self.s_std,
                                n_bins=self.n_bins_act,
                                state_clip_k=self._state_clip_k,
                            )
                        else:
                            round2_target = None
                            round2_goal_user_block = ""
                    else:
                        poses_np = None
                        round2_action_traj = np.asarray(stitched_np, dtype=np.float32).copy()
                        round2_target = format_trajectory_target_text(
                            stitched_np, self.a_min, self.a_max, self.n_bins_act
                        )
                        round2_goal_user_block = format_memory_goal_state_text(
                            stitched_np, self.a_min, self.a_max, self.n_bins_act
                        )
                    target_flat = int(green["target_flat_idx"])
                    tr_b = self._get_transition_by_flat(target_flat)
                    if round2_target and tr_b is not None:
                        try:
                            goal_vf = float(tr_b.get("vf_value_pred"))
                            if np.isfinite(goal_vf) and np.isfinite(current_vf):
                                round2_goal_vf_delta = float(goal_vf - current_vf)
                        except Exception:
                            round2_goal_vf_delta = 0.0
                        round2_images = imgs + _obs_to_pil(tr_b["observations"], self._side_key, self._wrist_key)
                        has_round2 = True
                        stitch_blue_len = int(blue_len)
                        stitch_single = False
                        round2_improvement = float(green.get("improvement", 0.0))
                        round2_slope_up = float(green.get("slope_up", 0.0))
                        q = float(
                            green.get(
                                "quality_score",
                                1.0 + max(0.0, round2_improvement) + 2.0 * max(0.0, round2_slope_up),
                            )
                        )
                        round2_quality_score = float(min(q, 3.0))
                        round2_memory_bank_path = str(green.get("memory_bank_path", "") or "")
                        round2_memory_route_key = str(green.get("memory_bank_route_key", "") or "")
                        round2_memory_item_index = int(green.get("memory_bank_item_index", -1))
                        round2_memory_rank = int(green.get("memory_bank_rank", -1))
                        round2_memory_score = float(green.get("memory_bank_score", float("nan")))
                        round2_memory_task_filter = str(green.get("memory_bank_task_filter", "") or "")
                        round2_memory_item_task_text = str(green.get("memory_bank_item_task_text", "") or "")
                        round2_memory_item_task_id = str(green.get("memory_bank_item_task_id", "") or "")
                        round2_memory_episode_id = str(green.get("memory_bank_item_episode_id", "") or "")
                        round2_memory_failure_index = int(green.get("memory_bank_item_failure_index", -1))
                        round2_memory_target_index = int(green.get("memory_bank_item_target_index", -1))

        return {
            "flat_idx": int(idx),
            "images": imgs,
            "instr": self._resolve_instr(tr),
            "current_action_text": current_action_text,
            "current_state_text": current_state_text,
            "target": target,
            "intervened": intervened,
            "actions": action,
            "state": state_v.astype(np.float32),
            "vf_value_pred": float(tr.get("vf_value_pred", 0.0)),
            "task_name": str(self._task_name_flat[idx]).strip() if 0 <= idx < len(self._task_name_flat) else "",
            "task_route_key": str(self._task_route_key_flat[idx]).strip() if 0 <= idx < len(self._task_route_key_flat) else "",
            "has_round2": has_round2,
            "round2_memory_bank_path": round2_memory_bank_path,
            "round2_memory_route_key": round2_memory_route_key,
            "round2_memory_item_index": int(round2_memory_item_index),
            "round2_memory_rank": int(round2_memory_rank),
            "round2_memory_score": float(round2_memory_score),
            "round2_memory_task_filter": round2_memory_task_filter,
            "round2_memory_item_task_text": round2_memory_item_task_text,
            "round2_memory_item_task_id": round2_memory_item_task_id,
            "round2_memory_episode_id": round2_memory_episode_id,
            "round2_memory_failure_index": int(round2_memory_failure_index),
            "round2_memory_target_index": int(round2_memory_target_index),
            "round2_memory_query_task": round2_memory_query_task,
            "round2_memory_query_route_key": round2_memory_query_route_key,
            "round2_target": round2_target,
            "round2_target_space": self._round2_target_space,
            "round2_action_traj": round2_action_traj,
            "round2_pose_traj": poses_np,
            "round2_images": round2_images,
            "round2_goal_user_block": round2_goal_user_block,
            "round2_stitch_blue_len": stitch_blue_len,
            "round2_stitch_single_segment": stitch_single,
            "round2_improvement": float(round2_improvement),
            "round2_slope_up": float(round2_slope_up),
            "round2_goal_vf_delta": float(round2_goal_vf_delta),
            "round2_quality_score": float(round2_quality_score),
            "stage_weight": float(tr.get("stage_weight", 1.0)),
            "indicator": int(tr.get("indicator", -1)),
        }

    def get_action_stats(self) -> dict:
        """Return min/max, which can be stored in dataset_stats for QwenActor.set_dataset_stats."""
        return {"min": self.a_min.tolist(), "max": self.a_max.tolist()}

    def get_state_norm_stats(self) -> dict:
        return {
            "mean": np.asarray(self.s_mean, dtype=np.float64).tolist(),
            "std": np.asarray(self.s_std, dtype=np.float64).tolist(),
            "dim": int(self._state_dim),
            "clip_k": float(self._state_clip_k),
        }


if __name__ == "__main__":
    import sys
    buf = sys.argv[1] if len(sys.argv) > 1 else "buffer_data/buffer_nttg"
    for spl in ("train", "val", "test"):
        ds = HILVLMDataset(buf, split=spl)
        n_int = sum(ds.intervention_label_at(i) for i in range(len(ds)))
        print(f"  {spl}: {len(ds)} samples, intervened {n_int} ({100*n_int/max(len(ds),1):.1f}%)")
    ds = HILVLMDataset(buf, split="train")
    val_ds = HILVLMDataset(
        buf,
        split="val",
        action_min=ds.a_min,
        action_max=ds.a_max,
        state_mean=ds.s_mean,
        state_std=ds.s_std,
        state_dim=ds._state_dim,
        state_clip_k=ds._state_clip_k,
    )
    sample = ds[0]
    print("\n=== train sample 0 ===")
    print(f"  target:\n{sample['target']}")
    print("dataset_stats:", ds.get_action_stats())
