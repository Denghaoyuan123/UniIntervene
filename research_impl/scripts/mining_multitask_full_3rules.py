#!/usr/bin/env python3
"""Mine intervention labels over a buffer with the decline, plateau and abrupt-drop rules."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.recover_target_mining import detect_decline_window  # noqa: E402


def _as_float_array(values: list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _minmax_episode(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    vmin = float(values.min())
    vmax = float(values.max())
    span = vmax - vmin
    if span < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / span).astype(np.float32)


def detect_plateau_window(
    values: np.ndarray,
    t: int,
    *,
    window: int = 8,
    flat_ratio: float = 0.60,
    delta: float = 0.03,
    amp_mult: float = 4.0,
) -> bool:
    start = max(0, int(t) - int(window) + 1)
    seg = np.asarray(values[start : int(t) + 1], dtype=np.float64)
    if seg.shape[0] < 2:
        return False
    diffs = np.abs(np.diff(seg))
    fr = float((diffs <= float(delta)).mean()) if diffs.size > 0 else 0.0
    amp = float(seg.max() - seg.min())
    return bool(fr >= float(flat_ratio) and amp <= float(delta) * float(amp_mult))


def detect_abrupt_drop_window(
    values: np.ndarray,
    t: int,
    *,
    window: int = 8,
    min_drop: float = 0.12,
    recover_horizon: int = 3,
    recover_frac: float = 0.5,
) -> bool:
    """Mark the trough step of a large single-step drop if it does not recover fast."""
    t = int(t)
    start = max(0, t - int(window) + 1)
    seg = np.asarray(values[start : t + 1], dtype=np.float64)
    if seg.shape[0] < 2:
        return False

    diffs = np.diff(seg)
    big_drop_idxs = np.where(diffs <= -float(min_drop))[0]
    if big_drop_idxs.size == 0:
        return False

    drop_rel = int(big_drop_idxs[np.argmin(diffs[big_drop_idxs])])
    trough_abs = start + drop_rel + 1
    if trough_abs != t:
        return False

    drop = float(seg[drop_rel] - seg[drop_rel + 1])
    if drop < float(min_drop):
        return False

    future = np.asarray(values[t + 1 : t + 1 + int(recover_horizon)], dtype=np.float64)
    if future.size == 0:
        return True

    recover_level = float(seg[drop_rel + 1] + float(recover_frac) * drop)
    return not bool(np.any(future >= recover_level))


def _task_name_from_episode(ep: list[dict[str, Any]]) -> str:
    for tr in ep:
        task = tr.get("task_name") or tr.get("task") or tr.get("source_task")
        if task:
            return str(task)
    return "unknown"


@dataclass
class RuleCounts:
    hits: int = 0
    total: int = 0

    @property
    def ratio(self) -> float:
        return float(self.hits) / float(self.total) if self.total > 0 else 0.0


def _load_episode(path: Path) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        ep = pickle.load(f)
    if not isinstance(ep, list):
        raise TypeError(f"episode {path} is not a list")
    return ep


def _save_episode(path: Path, ep: list[dict[str, Any]]) -> None:
    with open(path, "wb") as f:
        pickle.dump(ep, f, protocol=pickle.HIGHEST_PROTOCOL)


def mine_buffer(args: argparse.Namespace) -> dict[str, Any]:
    src = Path(args.buffer_dir).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"buffer_dir not found: {src}")
    if out.exists() and sorted(out.glob(str(args.episode_glob))):
        if not args.overwrite:
            raise FileExistsError(
                f"output_dir already contains mined episodes: {out} (use --overwrite)"
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob(str(args.episode_glob)))
    if not files:
        raise FileNotFoundError(f"no {args.episode_glob} found under {src}")

    per_task: dict[str, dict[str, RuleCounts]] = defaultdict(
        lambda: {k: RuleCounts() for k in ("decline", "plateau", "abrupt", "any")}
    )
    overall = {k: RuleCounts() for k in ("decline", "plateau", "abrupt", "any")}
    episode_rows: list[dict[str, Any]] = []

    for i, fp in enumerate(files, start=1):
        ep = _load_episode(fp)
        if not ep:
            continue
        if any("vf_value_pred" not in tr for tr in ep):
            raise KeyError(f"{fp.name} has no vf_value_pred in every transition")

        raw = _as_float_array([float(tr["vf_value_pred"]) for tr in ep])
        values = _minmax_episode(raw)
        task_name = _task_name_from_episode(ep)

        counts = {k: 0 for k in ("decline", "plateau", "abrupt", "any")}
        for t, tr in enumerate(ep):
            decline = detect_decline_window(
                values,
                t,
                window=int(args.window),
                slope_threshold=float(args.slope_threshold),
                min_drop=float(args.min_drop),
                min_down_ratio=float(args.min_down_ratio),
            ) is not None
            plateau = detect_plateau_window(
                values,
                t,
                window=int(args.window),
                flat_ratio=float(args.flat_ratio),
                delta=float(args.delta),
                amp_mult=float(args.amp_mult),
            )
            abrupt = detect_abrupt_drop_window(
                values,
                t,
                window=int(args.window),
                min_drop=float(args.abrupt_min_drop),
                recover_horizon=int(args.recover_horizon),
                recover_frac=float(args.recover_frac),
            )
            intervene = bool(decline or plateau or abrupt)

            tr["vf_mining_decline"] = bool(decline)
            tr["vf_mining_plateau"] = bool(plateau)
            tr["vf_mining_abrupt_drop"] = bool(abrupt)
            tr["vf_mining_intervention_label"] = bool(intervene)

            counts["decline"] += int(decline)
            counts["plateau"] += int(plateau)
            counts["abrupt"] += int(abrupt)
            counts["any"] += int(intervene)

        out_fp = out / fp.name
        _save_episode(out_fp, ep)

        for key in counts:
            per_task[task_name][key].hits += counts[key]
            per_task[task_name][key].total += len(ep)
            overall[key].hits += counts[key]
            overall[key].total += len(ep)

        episode_rows.append(
            {
                "file": fp.name,
                "task": task_name,
                "length": len(ep),
                "decline_hits": counts["decline"],
                "plateau_hits": counts["plateau"],
                "abrupt_hits": counts["abrupt"],
                "any_hits": counts["any"],
            }
        )

        if i % int(args.print_every) == 0 or i == len(files):
            print(
                f"[mine] {i}/{len(files)} {fp.name} task={task_name} T={len(ep)} "
                f"any={counts['any']}",
                flush=True,
            )

    manifest = {
        "buffer_dir": str(src),
        "output_dir": str(out),
        "label_key": "vf_mining_intervention_label",
        "params": {
            "window": int(args.window),
            "decline": {
                "min_drop": float(args.min_drop),
                "min_down_ratio": float(args.min_down_ratio),
                "slope_threshold": float(args.slope_threshold),
            },
            "plateau": {
                "flat_ratio": float(args.flat_ratio),
                "delta": float(args.delta),
                "amp_mult": float(args.amp_mult),
            },
            "abrupt_drop": {
                "min_drop": float(args.abrupt_min_drop),
                "recover_horizon": int(args.recover_horizon),
                "recover_frac": float(args.recover_frac),
            },
        },
        "tasks": {
            task: {
                key: {
                    "hits": per_task[task][key].hits,
                    "total": per_task[task][key].total,
                    "ratio": per_task[task][key].ratio,
                }
                for key in ("decline", "plateau", "abrupt", "any")
            }
            for task in sorted(per_task)
        },
        "overall": {
            key: {
                "hits": overall[key].hits,
                "total": overall[key].total,
                "ratio": overall[key].ratio,
            }
            for key in ("decline", "plateau", "abrupt", "any")
        },
        "episode_rows": episode_rows,
    }
    (out / "vf_mining_3rules_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] wrote {out / 'vf_mining_3rules_manifest.json'}", flush=True)
    return manifest


def build_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Mine intervention labels over a buffer with the decline, plateau and abrupt-drop rules.")
    ap.add_argument(
        "--buffer_dir",
        required=True,
    )
    ap.add_argument(
        "--output_dir",
        required=True,
    )
    ap.add_argument(
        "--episode_glob",
        default="transitions_*_nttg_vf.pkl",
        help="Episode filename pattern under buffer_dir.",
    )
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--min_drop", type=float, default=0.05)
    ap.add_argument("--min_down_ratio", type=float, default=0.7)
    ap.add_argument("--slope_threshold", type=float, default=-0.01)
    ap.add_argument("--flat_ratio", type=float, default=0.60)
    ap.add_argument("--delta", type=float, default=0.03)
    ap.add_argument("--amp_mult", type=float, default=4.0)
    ap.add_argument("--abrupt_min_drop", type=float, default=0.12)
    ap.add_argument("--recover_horizon", type=int, default=3)
    ap.add_argument("--recover_frac", type=float, default=0.5)
    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = build_args()
    mine_buffer(args)


if __name__ == "__main__":
    main()
