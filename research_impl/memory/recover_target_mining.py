from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch


ArrayLike = Sequence[float] | np.ndarray | torch.Tensor


def _as_numpy(values: ArrayLike) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float32)
    return np.asarray(values, dtype=np.float32)


def _as_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    return torch.as_tensor(x, dtype=torch.float32)


@dataclass
class DeclineWindow:
    start_index: int
    end_index: int
    slope: float
    drop: float
    down_ratio: float
    values: np.ndarray


@dataclass
class RecoverResult:
    reference_index: int
    reference_value: float
    target_index: int
    target_value: float
    used_fallback: bool
    vf_stable_start: Optional[int] = None
    vf_ref_value: Optional[float] = None
    vf_target_value: Optional[float] = None


@dataclass
class RecoverTargetMiningConfig:
    decline_window: int = 8
    min_down_ratio: float = 0.7
    slope_threshold: float = -0.01
    min_drop: float = 0.05
    pre_drop_lookback: int = 8
    recover_tolerance: float = 0.02
    require_intervention_signal: bool = True
    intervention_margin: int = 2
    intervention_key: str = "intervened"
    labels_key: str = "labels"
    task_key: str = "task"
    value_key: str = "raw_value"
    value_normalization: str = "none"
    action_key: str = "actions"
    fallback_to_best_future: bool = True
    skip_if_no_recover: bool = False
    action_chunk_length: int = 8
    use_vf_stable_recover: bool = False
    vf_recover_tolerance: float = 0.02
    vf_stable_steps: int = 1
    min_steps_after_target: int = 0
    skip_if_recovery_not_all_teleop: bool = False


def _transition_human_teleop(tr: dict, *, intervention_key: str, labels_key: str) -> bool:
    """Return whether a transition contains a human intervention."""
    if bool(tr.get(intervention_key, False)):
        return True
    if labels_key in tr and tr.get(labels_key) == 2:
        return True
    return False


def recovery_segment_teleop_stats(
    episode: Sequence[dict],
    start_index: int,
    end_index: int,
    *,
    intervention_key: str = "intervened",
    labels_key: str = "labels",
) -> tuple[int, int, float, bool]:
    """Summarize human intervention coverage in an inclusive recovery segment."""
    if end_index < start_index or start_index < 0 or end_index >= len(episode):
        return (0, 0, 0.0, False)
    total = 0
    human = 0
    for idx in range(start_index, end_index + 1):
        total += 1
        if _transition_human_teleop(episode[idx], intervention_key=intervention_key, labels_key=labels_key):
            human += 1
    frac = float(human) / float(total) if total > 0 else 0.0
    all_h = human == total and total > 0
    return (human, total, frac, all_h)


def detect_decline_window(
    values: ArrayLike,
    t: int,
    window: int = 8,
    slope_threshold: float = -0.01,
    min_drop: float = 0.05,
    min_down_ratio: float = 0.7,
) -> Optional[DeclineWindow]:
    """Detect whether time t lies in a sustained progress/value decline window."""
    values = _as_numpy(values)
    start = max(0, t - window + 1)
    segment = values[start : t + 1]
    if segment.shape[0] < 2:
        return None

    x = np.arange(segment.shape[0], dtype=np.float32)
    slope = float(np.polyfit(x, segment, deg=1)[0])
    diffs = np.diff(segment)
    down_ratio = float((diffs < 0).mean()) if diffs.size > 0 else 0.0
    drop = float(segment[0] - segment[-1])

    if slope > slope_threshold:
        return None
    if drop < min_drop:
        return None
    if down_ratio < min_down_ratio:
        return None

    return DeclineWindow(
        start_index=start,
        end_index=t,
        slope=slope,
        drop=drop,
        down_ratio=down_ratio,
        values=segment.copy(),
    )


def _nearest_intervention_index(
    episode: Sequence[dict],
    intervention_key: str,
    labels_key: str,
) -> list[int]:
    indices = []
    for idx, transition in enumerate(episode):
        intervened = bool(transition.get(intervention_key, False))
        if labels_key in transition and transition.get(labels_key) == 2:
            intervened = True
        if intervened:
            indices.append(idx)
    return indices


def _is_intervention_candidate(
    t: int,
    intervention_indices: Sequence[int],
    margin: int,
    require_intervention_signal: bool,
) -> bool:
    if not require_intervention_signal:
        return True
    return any(abs(t - idx) <= margin for idx in intervention_indices)


def find_pre_drop_reference(
    values: ArrayLike,
    decline_start: int,
    lookback: int = 8,
) -> tuple[int, float]:
    """Find the progress level before the drop began."""
    values = _as_numpy(values)
    left = max(0, decline_start - lookback)
    reference_slice = values[left : decline_start + 1]
    local_idx = int(np.argmax(reference_slice))
    reference_index = left + local_idx
    return reference_index, float(values[reference_index])


def _episode_values(
    episode: Sequence[dict],
    config: RecoverTargetMiningConfig,
) -> np.ndarray:
    """Read the value sequence and apply the configured episode normalization."""
    raw = np.asarray(
        [float(step.get(config.value_key, 0.0)) for step in episode],
        dtype=np.float32,
    )
    norm = getattr(config, "value_normalization", "none") or "none"
    if norm == "minmax_episode":
        vmin, vmax = float(raw.min()), float(raw.max())
        span = vmax - vmin
        if span < 1e-8:
            return np.zeros_like(raw, dtype=np.float32)
        return ((raw - vmin) / span).astype(np.float32)
    return raw


def find_recover_target_vf_stable(
    progress_track: ArrayLike,
    values_raw_for_target: ArrayLike,
    anchor_index: int,
    decline_start: int,
    pre_drop_lookback: int,
    eps: float,
    stable_steps: int,
    min_steps_after_target: int,
) -> Optional[RecoverResult]:
    """Find the first stable future state that recovers to the pre-drop value."""
    prog = _as_numpy(progress_track)
    raw_t = _as_numpy(values_raw_for_target)
    if prog.shape[0] != raw_t.shape[0]:
        return None
    n = int(prog.shape[0])
    ref_idx, ref_prog = find_pre_drop_reference(
        values=prog,
        decline_start=decline_start,
        lookback=pre_drop_lookback,
    )
    v_ref = float(prog[ref_idx])
    if not np.isfinite(v_ref):
        return None
    k = max(1, int(stable_steps))
    thr = v_ref - float(eps)
    last_start = n - k
    if last_start < anchor_index + 1:
        return None
    for t0 in range(anchor_index + 1, last_start + 1):
        seg = prog[t0 : t0 + k]
        if not np.all(np.isfinite(seg)):
            continue
        if not bool(np.all(seg >= thr)):
            continue
        target_index = t0 + k - 1
        rem = (n - 1) - target_index
        if rem < int(min_steps_after_target):
            continue
        return RecoverResult(
            reference_index=ref_idx,
            reference_value=float(ref_prog),
            target_index=target_index,
            target_value=float(prog[target_index]),
            used_fallback=False,
            vf_stable_start=int(t0),
            vf_ref_value=float(v_ref),
            vf_target_value=float(prog[target_index]),
        )
    return None


def _safe_raw_value(raw_row: np.ndarray, idx: int) -> Optional[float]:
    if idx < 0 or idx >= int(raw_row.shape[0]):
        return None
    v = float(raw_row[idx])
    return v if np.isfinite(v) else None


def _quality_score_from_vf_delta(delta: float, steps: int) -> float:
    steps = max(1, int(steps))
    slope = float(delta) / float(steps)
    return float(1.0 + max(0.0, float(delta)) + 2.0 * max(0.0, slope))


def compute_recover_mining_intervention_mask(
    episode: Sequence[dict],
    config: RecoverTargetMiningConfig,
) -> list[bool]:
    """Return the intervention mask induced by decline and recovery mining."""
    values = _episode_values(episode, config)
    raw_row = np.asarray(
        [float(step.get("raw_value", float("nan"))) for step in episode],
        dtype=np.float32,
    )
    intervention_indices = _nearest_intervention_index(
        episode,
        intervention_key=config.intervention_key,
        labels_key=config.labels_key,
    )
    out = [False] * len(episode)
    for t in range(len(episode)):
        if not _is_intervention_candidate(
            t=t,
            intervention_indices=intervention_indices,
            margin=config.intervention_margin,
            require_intervention_signal=config.require_intervention_signal,
        ):
            continue
        decline = detect_decline_window(
            values=values,
            t=t,
            window=config.decline_window,
            slope_threshold=config.slope_threshold,
            min_drop=config.min_drop,
            min_down_ratio=config.min_down_ratio,
        )
        if decline is None:
            continue
        if config.use_vf_stable_recover:
            recover = find_recover_target_vf_stable(
                progress_track=values,
                values_raw_for_target=raw_row,
                anchor_index=t,
                decline_start=decline.start_index,
                pre_drop_lookback=config.pre_drop_lookback,
                eps=config.vf_recover_tolerance,
                stable_steps=config.vf_stable_steps,
                min_steps_after_target=config.min_steps_after_target,
            )
        else:
            recover = find_recover_target(
                values=values,
                anchor_index=t,
                decline_start=decline.start_index,
                recover_tolerance=config.recover_tolerance,
                pre_drop_lookback=config.pre_drop_lookback,
                fallback_to_best_future=config.fallback_to_best_future,
            )
        if recover is None:
            if config.skip_if_no_recover:
                continue
            continue
        out[t] = True
    return out


def extract_recovery_segment_actions(
    episode: Sequence[dict],
    start_index: int,
    end_index: int,
    action_key: str = "actions",
) -> Optional[torch.Tensor]:
    """Stack the inclusive action segment from ``start_index`` to ``end_index``."""
    if end_index < start_index or start_index < 0 or end_index >= len(episode):
        return None
    rows: list[torch.Tensor] = []
    for idx in range(start_index, end_index + 1):
        if action_key not in episode[idx]:
            return None
        rows.append(torch.as_tensor(episode[idx][action_key], dtype=torch.float32))
    if not rows:
        return None
    return torch.stack(rows, dim=0)


def find_recover_target(
    values: ArrayLike,
    anchor_index: int,
    decline_start: int,
    recover_tolerance: float = 0.02,
    pre_drop_lookback: int = 8,
    fallback_to_best_future: bool = True,
) -> Optional[RecoverResult]:
    """Search forward for the earliest future state that recovers to pre-drop level."""
    values = _as_numpy(values)
    ref_idx, ref_value = find_pre_drop_reference(
        values=values,
        decline_start=decline_start,
        lookback=pre_drop_lookback,
    )

    for future_idx in range(anchor_index + 1, len(values)):
        if values[future_idx] >= ref_value - recover_tolerance:
            return RecoverResult(
                reference_index=ref_idx,
                reference_value=ref_value,
                target_index=future_idx,
                target_value=float(values[future_idx]),
                used_fallback=False,
            )

    if not fallback_to_best_future or anchor_index + 1 >= len(values):
        return None

    future_slice = values[anchor_index + 1 :]
    best_local_idx = int(np.argmax(future_slice))
    target_index = anchor_index + 1 + best_local_idx
    return RecoverResult(
        reference_index=ref_idx,
        reference_value=ref_value,
        target_index=target_index,
        target_value=float(values[target_index]),
        used_fallback=True,
    )


def default_action_chunk_extractor(
    episode: Sequence[dict],
    start_index: int,
    chunk_length: int,
    action_key: str = "actions",
) -> Optional[torch.Tensor]:
    chunk = []
    for idx in range(start_index, min(start_index + chunk_length, len(episode))):
        if action_key not in episode[idx]:
            return None
        chunk.append(torch.as_tensor(episode[idx][action_key], dtype=torch.float32))
    if not chunk:
        return None
    return torch.stack(chunk, dim=0)

