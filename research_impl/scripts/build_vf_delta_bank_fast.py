#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hil_vlm_eval_vf import build_retrieval_inputs, compute_failure_embedding, load_model  # noqa: E402
from hil_vlm_dataset import _obs_to_pil  # noqa: E402
from episode_split import episode_stratum, stratified_episode_split  # noqa: E402
from memory_constraints import recovery_chunk_indices, segment_is_eligible  # noqa: E402
from memory.memory_bank import MemoryBank, MemoryItem  # noqa: E402
from memory.query_builder import MemoryQueryBuilder, QueryBuilderConfig  # noqa: E402


@dataclass
class Cand:
    episode_id: str
    episode_idx: int
    task_name: str
    source_file: str
    start: int
    end: int
    start_bin: int
    delta: float
    span: int
    slope: float
    start_vf: float
    end_vf: float
    task_text: str
    kf_tier: int = 0
    kf_dist: float = float('inf')
    kf_nearest: int = -1
    kf_vf: float = float('nan')


def _log(msg: str) -> None:
    import time
    print(f"[vf_delta_bank_fast] {time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}", flush=True)


def _manifest_sha256(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: os.path.basename(value)):
        digest.update(os.path.basename(path).encode('utf-8'))
        with open(path, 'rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def _load_episode(path: str) -> list[dict[str, Any]]:
    with open(path, 'rb') as f:
        ep = pickle.load(f)
    if not isinstance(ep, list):
        raise TypeError(path)
    return ep


def _as_state11(x: Any) -> list[float] | None:
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 11:
        y = arr
    elif arr.size == 9:
        y = np.concatenate([np.zeros(2, dtype=np.float32), arr], axis=0)
    else:
        return None
    return y.astype(np.float32).tolist()


def _get_obs_state(tr: dict, key: str) -> list[float] | None:
    obs = tr.get(key, None)
    if not isinstance(obs, dict):
        return None
    return _as_state11(obs.get('state', None))


def _episode_task_name(ep: list[dict[str, Any]]) -> str:
    for tr in ep:
        if not isinstance(tr, dict):
            continue
        for k in ('task_name', 'task', 'task_id', 'source_task'):
            v = tr.get(k)
            if v:
                return str(v)
    return 'unknown'


def _episode_task_text(ep: list[dict[str, Any]], fallback: str) -> str:
    for tr in ep:
        if not isinstance(tr, dict):
            continue
        for k in ('task_prompt', 'task_text'):
            v = tr.get(k)
            if v:
                return str(v)
    return fallback


def _normalize_task_filters(task_filter: str) -> set[str]:
    vals = {x.strip().lower() for x in str(task_filter or '').split(',') if x.strip()}
    return vals


def _task_allowed(task_name: str, task_filters: set[str]) -> bool:
    if not task_filters:
        return True
    t = str(task_name).strip().lower()
    return t in task_filters


def _normalize_values(vals: np.ndarray) -> np.ndarray:
    if vals.size == 0:
        return vals.astype(np.float32)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    span = vmax - vmin
    if span < 1e-8:
        return np.zeros_like(vals, dtype=np.float32)
    return ((vals - vmin) / span).astype(np.float32)


def _episode_success_label(ep: list[dict[str, Any]], source_file: str = '') -> bool | None:
    """Return episode-level success flag."""
    vals: list[bool] = []
    for tr in ep:
        if not isinstance(tr, dict):
            continue
        if 'success' in tr:
            vals.append(bool(tr.get('success')))
            continue
        infos = tr.get('infos')
        if isinstance(infos, dict) and ('success' in infos):
            vals.append(bool(infos.get('success')))
    if not vals:
        b = os.path.basename(str(source_file)).lower()
        if 'success' in b:
            return True
        if ('failure' in b) or ('fail' in b):
            return False
        return None
    if all(vals):
        return True
    if not any(vals):
        return False
    return None


def _find_gripper_keyframes(ep: list[dict[str, Any]]) -> list[int]:
    """Detect gripper state change indices from observations.state."""
    vals: list[float | None] = []
    for tr in ep:
        if not isinstance(tr, dict):
            vals.append(None)
            continue
        s = _get_obs_state(tr, 'observations')
        if s is None or len(s) < 2:
            vals.append(None)
        else:
            vals.append(float(s[1]))
    kfs: list[int] = []
    prev: float | None = None
    for i, v in enumerate(vals):
        if v is None:
            continue
        if prev is None:
            prev = v
            continue
        if abs(v - prev) > 0.5:
            kfs.append(i)
        prev = v
    if kfs:
        return kfs

    acts: list[float] = []
    for tr in ep:
        if not isinstance(tr, dict):
            acts.append(float('nan'))
            continue
        a = tr.get('actions', None)
        try:
            arr = np.asarray(a, dtype=float).ravel()
            acts.append(float(arr[-1]) if arr.size >= 7 else float('nan'))
        except Exception:
            acts.append(float('nan'))
    finite = [x for x in acts if x == x]
    _cmd_like = False
    if len(finite) >= 2:
        _bounded = all(-1.05 <= x <= 1.05 for x in finite)
        _at_ext = sum(1 for x in finite if min(abs(x - c) for c in (-1.0, 0.0, 1.0)) <= 0.05)
        _cmd_like = _bounded and (_at_ext / len(finite)) >= 0.50
    if _cmd_like:
        prev_a: float | None = None
        for i, v in enumerate(acts):
            if v != v:
                continue
            if prev_a is None:
                prev_a = v
                continue
            if abs(v - prev_a) > 0.5:
                kfs.append(i)
            prev_a = v
    return kfs


def _annotate_kf_tier(c: Cand, kfs: list[int]) -> None:
    """Assign keyframe tier based on distance from segment midpoint to nearest keyframe."""
    if not kfs:
        c.kf_tier = 0
        c.kf_dist = float('inf')
        c.kf_nearest = -1
        return
    mid = (c.start + c.end) / 2.0
    half = max(1.0, c.span / 2.0)
    best_k = min(kfs, key=lambda k: abs(mid - k))
    dist = abs(mid - best_k)
    c.kf_nearest = int(best_k)
    c.kf_dist = float(dist)
    if dist <= half * 0.5:
        c.kf_tier = 2
    elif dist <= half:
        c.kf_tier = 1
    else:
        c.kf_tier = 0


def _find_candidates(
    ep: list[dict[str, Any]],
    episode_id: str,
    episode_idx: int,
    source_file: str,
    delta_thr: float,
    target_vf_min: float,
    bins: int,
    task_text: str,
    fixed_span: int = 0,
    keyframes: list[int] | None = None,
) -> list[Cand]:
    vals = np.asarray([float(tr.get('vf_value_pred', 0.0)) for tr in ep], dtype=np.float32)
    n = len(vals)
    out: list[Cand] = []
    if n < 2:
        return out
    kfs = list(keyframes) if keyframes else []
    if fixed_span > 0:
        for s in range(0, n - fixed_span):
            e = s + fixed_span
            delta = float(vals[e] - vals[s])
            span = int(e - s)
            slope = float(delta / max(1, span))
            b = min(bins - 1, int((float(s) / max(1, (n - fixed_span - 1))) * bins))
            if not segment_is_eligible(vals[s], vals[e], delta_thr, target_vf_min):
                continue
            cand = Cand(
                episode_id=episode_id,
                episode_idx=episode_idx,
                task_name=_episode_task_name(ep),
                source_file=os.path.basename(source_file),
                start=s,
                end=e,
                start_bin=b,
                delta=delta,
                span=span,
                slope=slope,
                start_vf=float(vals[s]),
                end_vf=float(vals[e]),
                task_text=task_text,
            )
            if kfs:
                _annotate_kf_tier(cand, kfs)
                if cand.kf_nearest >= 0 and 0 <= cand.kf_nearest < n:
                    cand.kf_vf = float(vals[cand.kf_nearest])
            out.append(cand)
        return out
    for s in range(n - 1):
        base = float(vals[s])
        target = base + float(delta_thr)
        e = None
        for j in range(s + 1, n):
            if float(vals[j]) >= target:
                e = j
                break
        if e is None:
            continue
        if float(vals[e]) < float(target_vf_min):
            continue
        delta = float(vals[e] - vals[s])
        span = int(e - s)
        slope = float(delta / max(1, span))
        b = min(bins - 1, int((float(s) / max(1, n - 1)) * bins))
        cand = Cand(
            episode_id=episode_id,
            episode_idx=episode_idx,
            task_name=_episode_task_name(ep),
            source_file=os.path.basename(source_file),
            start=s,
            end=e,
            start_bin=b,
            delta=delta,
            span=span,
            slope=slope,
            start_vf=float(vals[s]),
            end_vf=float(vals[e]),
            task_text=task_text,
        )
        if kfs:
            _annotate_kf_tier(cand, kfs)
        out.append(cand)
    return out


def _score(c: Cand) -> tuple:
    return (int(c.kf_tier), float(c.delta), float(c.slope), -float(c.start), -int(c.span))


def _apply_fixed_span(cands: list[Cand], fixed_span: int) -> list[Cand]:
    if fixed_span <= 0:
        return cands
    return [c for c in cands if int(c.span) == int(fixed_span)]


def _compute_quotas(bins: int, total: int) -> list[int]:
    base = total // bins
    rem = total % bins
    q = [base] * bins
    for i in range(rem):
        q[i] += 1
    return q


def _transition_to_sample(tr: dict, instr: str) -> dict[str, Any]:
    imgs = _obs_to_pil(tr['observations'])
    return {'images': imgs, 'instr': instr}




def _is_fold_episode(ep: list[dict[str, Any]], source_file: str, buffer_dir: str) -> bool:
    t = _episode_task_name(ep).lower()
    src = os.path.basename(str(source_file)).lower()
    bdir = str(buffer_dir).lower()
    return ('fold' in t) or ('towel' in t) or ('fold' in src) or ('towel' in src) or ('fold' in bdir) or ('towel' in bdir)


def _effective_span_for_episode(ep: list[dict[str, Any]], source_file: str, buffer_dir: str, default_span: int, auto_fold_span16: bool, fold_span: int) -> int:
    if not auto_fold_span16:
        return int(default_span)
    if _is_fold_episode(ep, source_file, buffer_dir):
        return int(fold_span)
    return int(default_span)





def _task_partition_key(task_name: str) -> str:
    t = str(task_name).lower()
    if 'ram' in t:
        return 'ram'
    if 'tube' in t:
        return 'tube'
    return 'other'


def _is_insert_dataset(buffer_dir: str) -> bool:
    b = str(buffer_dir).lower()
    return ('insert' in b)


def _resolve_effective_profile(buffer_dir: str, args) -> tuple[int, int, int, int]:
    """Return (effective_progress_bins, effective_target_items, effective_fixed_span, effective_max_per_episode)."""
    eff_bins = int(args.progress_bins)
    eff_target = int(args.target_items)
    eff_span = int(args.fixed_span)
    eff_max_per_episode = int(getattr(args, 'max_per_episode', 2))
    bdir = str(buffer_dir).lower()
    is_fold_ds = ('fold' in bdir) or ('towel' in bdir)
    if bool(getattr(args, 'auto_fold_profile_20bin12x16', False)) and is_fold_ds:
        eff_bins = int(getattr(args, 'fold_progress_bins', 20))
        per_bin = int(getattr(args, 'fold_items_per_bin', 12))
        eff_target = int(eff_bins * per_bin)
        eff_span = int(getattr(args, 'fold_fixed_span', 16))
        eff_max_per_episode = int(getattr(args, 'fold_max_per_episode', 8))
    return eff_bins, eff_target, eff_span, eff_max_per_episode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--buffer_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--report_json', default='')
    ap.add_argument('--episode_glob', default='transitions_*_nttg_vf.pkl')
    ap.add_argument('--split', choices=('train', 'val', 'test', 'all'), default='train')
    ap.add_argument('--train_ratio', type=float, default=0.8)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--test_ratio', type=float, default=0.1)
    ap.add_argument('--split_seed', type=int, default=42)
    ap.add_argument('--task_name_filter', default='',
                    help='Comma-separated task_name filter.')
    ap.add_argument('--instr', default='Recover the object with a stronger improvement segment.')
    ap.add_argument('--target_items', type=int, default=120)
    ap.add_argument('--progress_bins', type=int, default=10)
    ap.add_argument('--delta_thr', type=float, default=0.5,
                    help='Raw vf_value_pred improvement threshold (no per-episode normalization).')
    ap.add_argument('--fixed_span', type=int, default=8,
                    help='Use only candidates with this exact span (0 disables fixed-span filter).')
    ap.add_argument('--auto_fold_span16', action='store_true',
                    help='When enabled, fold/towel tasks automatically use fold_fixed_span (default 16).')
    ap.add_argument('--fold_fixed_span', type=int, default=16,
                    help='Span to use for fold/towel tasks when --auto_fold_span16 is enabled.')
    ap.add_argument('--auto_fold_profile_20bin12x16', action='store_true',
                    help='Fold/towel profile switch: span=16, bins=20, 12 items per bin (total 240).')
    ap.add_argument('--fold_progress_bins', type=int, default=20,
                    help='Fold profile bins (used when --auto_fold_profile_20bin12x16 is enabled).')
    ap.add_argument('--fold_items_per_bin', type=int, default=12,
                    help='Fold profile items per bin (used when --auto_fold_profile_20bin12x16 is enabled).')
    ap.add_argument('--max_per_episode', type=int, default=2,
                    help='Maximum number of selected segments per episode.')
    ap.add_argument('--insert_balance_ram_tube', action='store_true',
                    help='For insert datasets, force 50/50 RAM/TUBE selection.')
    ap.add_argument('--insert_items_per_bin', type=int, default=12,
                    help='Insert profile items per bin when balance mode is enabled.')
    ap.add_argument('--fold_max_per_episode', type=int, default=8,
                    help='Per-episode cap used by fold profile (when enabled).')
    ap.add_argument('--gripper_keyframe_aware', action='store_true',
                    help='When enabled, prefer fixed-span segments whose midpoint is close to a '
                         'gripper state change (detected via state[1] flips). Auto-enabled with '
                         '--auto_fold_profile_20bin12x16.')
    ap.add_argument('--require_gripper_keyframe', action='store_true',
                    help='When enabled, drop candidates whose segment contains no gripper keyframe '
                         '(i.e. only keep kf_tier>=1). Implies --gripper_keyframe_aware.')
    ap.add_argument('--bin_by_keyframe_vf', action='store_true',
                    help='Bin candidates by VF value at the nearest gripper keyframe (clamped to '
                         '[0,1]) instead of trajectory progress position. Ensures coverage across '
                         'the full VF range when keyframes cluster in one progress region. '
                         'Implies --gripper_keyframe_aware.')
    ap.add_argument('--target_vf_min', type=float, default=0.75)
    ap.add_argument('--base_model_path', required=True)
    ap.add_argument('--lora_path', default='')
    ap.add_argument('--action_chunk_length', type=int, default=8)
    ap.add_argument('--query_embedding_output_dir', default='')
    ap.add_argument('--query_label_key', default='vf_mining_intervention_label')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    buffer_dir = args.buffer_dir
    out_path = args.out
    report_path = args.report_json or out_path.replace('.pt', '_report.json')
    task_filters = _normalize_task_filters(args.task_name_filter)

    eff_bins, eff_target_items, eff_fixed_span, eff_max_per_episode = _resolve_effective_profile(buffer_dir, args)
    insert_balance_mode = bool(getattr(args, 'insert_balance_ram_tube', False)) and _is_insert_dataset(buffer_dir)
    if insert_balance_mode:
        eff_bins = int(args.progress_bins)
        per_bin = int(getattr(args, 'insert_items_per_bin', 12))
        eff_target_items = int(eff_bins * per_bin)

    bdir_lower = str(buffer_dir).lower()
    is_fold_ds = ('fold' in bdir_lower) or ('towel' in bdir_lower)
    require_kf = bool(getattr(args, 'require_gripper_keyframe', False))
    bin_by_kf_vf = bool(getattr(args, 'bin_by_keyframe_vf', False))
    kf_aware = bool(getattr(args, 'gripper_keyframe_aware', False)) or require_kf or bin_by_kf_vf or (
        bool(getattr(args, 'auto_fold_profile_20bin12x16', False)) and is_fold_ds
    )

    _log(f'buffer_dir={buffer_dir}')
    _log(f'out_path={out_path}')
    _log(f'task_name_filter={sorted(task_filters) if task_filters else []}')
    _log(f'effective_profile bins={eff_bins} target_items={eff_target_items} fixed_span={eff_fixed_span} max_per_episode={eff_max_per_episode} insert_balance_mode={insert_balance_mode} gripper_keyframe_aware={kf_aware} require_gripper_keyframe={require_kf} bin_by_keyframe_vf={bin_by_kf_vf}')

    all_files = sorted(glob.glob(os.path.join(buffer_dir, args.episode_glob)))
    if not all_files:
        raise SystemExit('no episodes found')

    if args.split == 'all':
        files = all_files
    else:
        strata: dict[tuple[str, bool], list[str]] = {}
        for path in all_files:
            episode = _load_episode(path)
            task, has_query = episode_stratum(episode, args.query_label_key)
            strata.setdefault((task, has_query), []).append(path)
        files = stratified_episode_split(
            strata,
            split=args.split,
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            seed=int(args.split_seed),
        )
    if not files:
        raise SystemExit(f'no episodes in split={args.split}')

    action_rows = []
    for path in files:
        for transition in _load_episode(path):
            action = np.asarray(transition.get('actions'), dtype=np.float32).reshape(-1)
            if action.size != 7 or not np.isfinite(action).all():
                raise ValueError(f'{path}: expected finite 7D actions')
            action_rows.append(action)
    if not action_rows:
        raise RuntimeError(f'no actions in split={args.split}')
    _log(f'action_schema_ready dims={action_rows[0].shape[0]}')

    _log('loading model')
    model, processor, tensor_device = load_model(
        base_model_path=args.base_model_path,
        lora_path=args.lora_path.strip() or None,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        merge_lora=False,
        load_in_4bit=False,
        load_in_8bit=False,
        device_map_auto=False,
        lora_fp32=False,
    )
    _log(f'model_loaded tensor_device={tensor_device}')

    cands: list[Cand] = []
    n_ep_total = 0
    n_ep_success = 0
    n_ep_failure = 0
    n_ep_unknown = 0
    n_ep_with_kf = 0
    n_kf_total = 0
    for epi, fp in enumerate(files):
        ep = _load_episode(fp)
        n_ep_total += 1
        succ = _episode_success_label(ep, fp)
        if succ is True:
            n_ep_success += 1
        elif succ is False:
            n_ep_failure += 1
            continue
        else:
            n_ep_unknown += 1
            continue
        task_name = _episode_task_name(ep)
        if not _task_allowed(task_name, task_filters):
            continue
        task_text = _episode_task_text(ep, args.instr)
        eff_span = _effective_span_for_episode(
            ep=ep,
            source_file=fp,
            buffer_dir=buffer_dir,
            default_span=int(eff_fixed_span),
            auto_fold_span16=bool(getattr(args, 'auto_fold_span16', False)),
            fold_span=int(getattr(args, 'fold_fixed_span', 16)),
        )
        ep_kfs = _find_gripper_keyframes(ep) if kf_aware else []
        if ep_kfs:
            n_ep_with_kf += 1
            n_kf_total += len(ep_kfs)
        cands.extend(_find_candidates(
            ep, str(epi), epi, fp, args.delta_thr, args.target_vf_min,
            eff_bins, task_text, eff_span, ep_kfs,
        ))
        if (epi + 1) % 50 == 0:
            _log(f'scanned={epi+1}/{len(files)} cands={len(cands)}')
    if kf_aware:
        _log(f'gripper_keyframes episodes_with_kf={n_ep_with_kf} total_kf={n_kf_total}')
    _log(f'episode_filter total={n_ep_total} success={n_ep_success} failure_skipped={n_ep_failure} unknown_skipped={n_ep_unknown}')

    if not cands:
        raise SystemExit('no candidates')
    _log(f'cands_after_fixed_span={len(cands)} fixed_span={int(eff_fixed_span)} auto_fold_span16={bool(getattr(args, "auto_fold_span16", False))} fold_fixed_span={int(getattr(args, "fold_fixed_span", 16))} auto_fold_profile_20bin12x16={bool(getattr(args, "auto_fold_profile_20bin12x16", False))}')
    if require_kf:
        before = len(cands)
        cands = [c for c in cands if c.kf_tier >= 1]
        _log(f'cands_after_require_gripper_keyframe={len(cands)} (filtered_out={before - len(cands)})')
    if not cands:
        raise SystemExit('no candidates after fixed-span/keyframe filter')

    if bin_by_kf_vf:
        rebinned = 0
        skipped = 0
        for c in cands:
            v = float(c.kf_vf) if not (c.kf_vf != c.kf_vf) else None  # NaN check
            if v is None:
                skipped += 1
                continue
            v = max(0.0, min(0.999999, v))
            c.start_bin = min(eff_bins - 1, int(v * eff_bins))
            rebinned += 1
        kfvf_hist = [sum(1 for c in cands if c.start_bin == b) for b in range(eff_bins)]
        _log(f'rebinned_by_keyframe_vf={rebinned} skipped(no_kf_vf)={skipped} bin_hist={kfvf_hist}')

    quotas = _compute_quotas(eff_bins, eff_target_items)
    picked: list[Cand] = []
    ep_used: dict[str, int] = {}

    if insert_balance_mode:
        ram_cands: dict[int, list[Cand]] = {i: [] for i in range(eff_bins)}
        tube_cands: dict[int, list[Cand]] = {i: [] for i in range(eff_bins)}
        for c in cands:
            pk = _task_partition_key(c.task_name)
            if pk == 'ram':
                ram_cands[c.start_bin].append(c)
            elif pk == 'tube':
                tube_cands[c.start_bin].append(c)
        for b in range(eff_bins):
            ram_cands[b].sort(key=_score, reverse=True)
            tube_cands[b].sort(key=_score, reverse=True)
        half = eff_target_items // 2
        q_half = _compute_quotas(eff_bins, half)
        for part in (ram_cands, tube_cands):
            for b in range(eff_bins):
                need = q_half[b]
                lst = part[b]
                while need > 0 and lst:
                    c = lst.pop(0)
                    if ep_used.get(c.episode_id, 0) >= eff_max_per_episode:
                        continue
                    picked.append(c)
                    ep_used[c.episode_id] = ep_used.get(c.episode_id, 0) + 1
                    need -= 1
        if len(picked) < eff_target_items:
            rem: list[Cand] = []
            for part in (ram_cands, tube_cands):
                for b in range(eff_bins):
                    rem.extend(part[b])
            rem.sort(key=_score, reverse=True)
            for c in rem:
                if len(picked) >= eff_target_items:
                    break
                if ep_used.get(c.episode_id, 0) >= eff_max_per_episode:
                    continue
                picked.append(c)
                ep_used[c.episode_id] = ep_used.get(c.episode_id, 0) + 1
    else:
        by_bin: dict[int, list[Cand]] = {i: [] for i in range(eff_bins)}
        for c in cands:
            by_bin[c.start_bin].append(c)
        for b in by_bin:
            by_bin[b].sort(key=_score, reverse=True)
        for b in range(eff_bins):
            need = quotas[b]
            lst = by_bin[b]
            while need > 0 and lst:
                c = lst.pop(0)
                if ep_used.get(c.episode_id, 0) >= eff_max_per_episode:
                    continue
                picked.append(c)
                ep_used[c.episode_id] = ep_used.get(c.episode_id, 0) + 1
                need -= 1
        if len(picked) < eff_target_items:
            rem: list[Cand] = []
            for b in range(eff_bins):
                rem.extend(by_bin[b])
            rem.sort(key=_score, reverse=True)
            for c in rem:
                if len(picked) >= eff_target_items:
                    break
                if ep_used.get(c.episode_id, 0) >= eff_max_per_episode:
                    continue
                picked.append(c)
                ep_used[c.episode_id] = ep_used.get(c.episode_id, 0) + 1

    picked = picked[:eff_target_items]
    _log(f'picked={len(picked)}')
    if len(picked) != int(eff_target_items):
        raise RuntimeError(
            f'bank requires {int(eff_target_items)} items, but only {len(picked)} satisfy the constraints'
        )
    if any(float(c.delta) < float(args.delta_thr) for c in picked):
        raise RuntimeError('selected candidate violates delta threshold')
    if any(float(c.end_vf) < float(args.target_vf_min) for c in picked):
        raise RuntimeError('selected candidate violates target-value threshold')

    items: list[MemoryItem] = []
    by_file = {os.path.basename(fp): fp for fp in files}
    for rank, c in enumerate(picked):
        fp = by_file[c.source_file]
        ep = _load_episode(fp)
        tr_s = ep[c.start]
        tr_e = ep[c.end]
        sample_s = _transition_to_sample(tr_s, c.task_text)
        sample_e = _transition_to_sample(tr_e, c.task_text)
        fe = compute_failure_embedding(model, build_retrieval_inputs(sample_s, processor, tensor_device))[0].detach().float().cpu()
        te = compute_failure_embedding(model, build_retrieval_inputs(sample_e, processor, tensor_device))[0].detach().float().cpu()

        episode_stem = os.path.splitext(c.source_file)[0]
        chunk_indices = recovery_chunk_indices(c.start, c.end, args.action_chunk_length)
        items.append(MemoryItem(
            task_text=c.task_text,
            task_id=c.task_name,
            progress_value=float(c.start_vf),
            failure_embedding=fe,
            target_embedding=te,
            failure_index=int(c.start),
            target_index=int(c.end),
            episode_id=episode_stem,
            success_score=float(c.delta),
            action_chunk=None,
            recovery_action_trajectory=torch.stack(
                [torch.as_tensor(ep[i]['actions'], dtype=torch.float32) for i in chunk_indices], dim=0
            ),
            metadata={
                'episode_idx': int(c.episode_idx),
                'task_name': c.task_name,
                'delta_thr': float(args.delta_thr),
                'start_vf': float(c.start_vf),
                'end_vf': float(c.end_vf),
                'vf_delta': float(c.delta),
                'span': int(c.span),
                'slope': float(c.slope),
                'start_bin': int(c.start_bin),
                'start_state_11d': _get_obs_state(tr_s, 'observations'),
                'end_next_state_11d': _get_obs_state(tr_e, 'next_observations'),
                'recovery_local_indices': chunk_indices,
                'kf_tier': int(c.kf_tier),
                'kf_dist': float(c.kf_dist) if c.kf_dist != float('inf') else None,
                'kf_nearest': int(c.kf_nearest),
                'kf_vf': float(c.kf_vf) if not (c.kf_vf != c.kf_vf) else None,
            },
        ))

    bank = MemoryBank(items=items, query_builder=MemoryQueryBuilder(QueryBuilderConfig()), device='cpu')
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    query_count = 0
    if args.query_embedding_output_dir:
        query_dir = os.path.abspath(args.query_embedding_output_dir)
        os.makedirs(query_dir, exist_ok=True)
        for fp in all_files:
            episode = _load_episode(fp)
            annotated: list[dict[str, Any]] = []
            task_text = _episode_task_text(episode, args.instr)
            for transition in episode:
                row = dict(transition)
                if bool(row.get(args.query_label_key, False)):
                    sample = _transition_to_sample(row, task_text)
                    embedding = compute_failure_embedding(
                        model, build_retrieval_inputs(sample, processor, tensor_device)
                    )[0].detach().float().cpu().numpy()
                    row['failure_emb_1d'] = embedding
                    query_count += 1
                annotated.append(row)
            output_path = os.path.join(query_dir, os.path.basename(fp))
            with open(output_path, 'wb') as handle:
                pickle.dump(annotated, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if query_count == 0:
            raise RuntimeError(f'no positive query labels found for key={args.query_label_key!r}')

    bank.save(out_path)

    report = {
        'split': str(args.split),
        'split_seed': int(args.split_seed),
        'source_episode_count': len(files),
        'source_manifest_sha256': _manifest_sha256(files),
        'query_embedding_count': int(query_count),
        'target_items': int(eff_target_items),
        'picked': len(items),
        'candidate_count': len(cands),
        'quotas': quotas,
        'bin_counts': [sum(1 for c in picked if c.start_bin == b) for b in range(eff_bins)],
        'gripper_keyframe_aware': bool(kf_aware),
        'require_gripper_keyframe': bool(require_kf),
        'bin_by_keyframe_vf': bool(bin_by_kf_vf),
        'kf_tier_counts': {
            'tier2_mid_centered': sum(1 for c in picked if c.kf_tier == 2),
            'tier1_inside_segment': sum(1 for c in picked if c.kf_tier == 1),
            'tier0_no_kf_inside': sum(1 for c in picked if c.kf_tier == 0),
        },
        'episodes_with_keyframes': int(n_ep_with_kf),
        'total_keyframes_detected': int(n_kf_total),
        'params': {
            'delta_thr': float(args.delta_thr),
            'target_vf_min': float(args.target_vf_min),
            'action_chunk_length': int(args.action_chunk_length),
            'task_name_filter': str(args.task_name_filter),
            'progress_bins': int(eff_bins),
            'fixed_span': int(args.fixed_span),
            'auto_fold_span16': bool(getattr(args, 'auto_fold_span16', False)),
            'fold_fixed_span': int(getattr(args, 'fold_fixed_span', 16)),
            'auto_fold_profile_20bin12x16': bool(getattr(args, 'auto_fold_profile_20bin12x16', False)),
            'fold_progress_bins': int(getattr(args, 'fold_progress_bins', 20)),
            'fold_items_per_bin': int(getattr(args, 'fold_items_per_bin', 12)),
            'max_per_episode': int(getattr(args, 'max_per_episode', 2)),
            'fold_max_per_episode': int(getattr(args, 'fold_max_per_episode', 8)),
            'insert_balance_ram_tube': bool(getattr(args, 'insert_balance_ram_tube', False)),
            'insert_items_per_bin': int(getattr(args, 'insert_items_per_bin', 12)),
            'effective_max_per_episode': int(eff_max_per_episode),
            'value_source': 'vf_value_pred',
            'exclude_failure_episodes': True,
        },
        'selected_stats': {
            'min_vf_delta': min(float(c.delta) for c in picked),
            'max_vf_delta': max(float(c.delta) for c in picked),
            'min_start_vf': min(float(c.start_vf) for c in picked),
            'max_start_vf': max(float(c.start_vf) for c in picked),
            'min_target_vf': min(float(c.end_vf) for c in picked),
            'max_target_vf': max(float(c.end_vf) for c in picked),
        },
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    import sys
    main()
