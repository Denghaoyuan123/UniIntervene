#!/usr/bin/env python3
"""Read-only schema check for trusted HIL-SERL episode pickles."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def validate_episode(path: Path) -> tuple[int, bool]:
    with path.open("rb") as f:
        episode = pickle.load(f)  # trusted local buffers only
    if not isinstance(episode, list) or not episode:
        raise ValueError("expected nonempty list[dict]")
    for index, step in enumerate(episode):
        if not isinstance(step, dict):
            raise ValueError(f"step {index}: expected dict")
        for key in ("observations", "next_observations"):
            obs = step.get(key)
            if not isinstance(obs, dict):
                raise ValueError(f"step {index}: missing {key} dict")
            for view in ("external", "wrist"):
                if view not in obs:
                    raise ValueError(f"step {index}: missing {key}.{view}")
                image = np.asarray(obs[view])
                if image.dtype != np.uint8 or image.shape[-1] != 3:
                    raise ValueError(f"step {index}: {key}.{view} must be uint8 RGB")
            if "state" not in obs:
                raise ValueError(f"step {index}: missing {key}.state")
        action = np.asarray(step.get("actions", []), dtype=np.float32).reshape(-1)
        if action.size != 7 or not np.isfinite(action).all():
            raise ValueError(f"step {index}: Fold Towel needs finite 7D action incl. gripper; got {action.size}D")
        if index == len(episode) - 1 and not any(k in step for k in ("rewards", "reward", "success", "failure")):
            raise ValueError("terminal step has no reward/success/failure")
    terminal = episode[-1]
    if "success" in terminal:
        success = bool(np.asarray(terminal["success"]).reshape(-1)[0])
    elif "failure" in terminal:
        success = not bool(np.asarray(terminal["failure"]).reshape(-1)[0])
    else:
        success = float(np.asarray(terminal.get("rewards", terminal.get("reward", 0))).reshape(-1)[0]) > 0
    return len(episode), success


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--max-episodes", type=int, default=20)
    a = p.parse_args()
    files = sorted(a.raw_dir.glob("transitions_*.pkl"))[: a.max_episodes]
    if not files:
        raise SystemExit(f"No transitions_*.pkl under {a.raw_dir}")
    successes = steps = 0
    for path in files:
        try:
            n, success = validate_episode(path)
        except Exception as exc:
            raise SystemExit(f"{path}: {exc}") from exc
        successes += success
        steps += n
    print(f"OK: {len(files)} episodes, {steps} steps, {successes} successes, "
          f"{len(files) - successes} failures; no files changed")


if __name__ == "__main__":
    main()
