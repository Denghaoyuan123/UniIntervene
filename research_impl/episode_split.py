from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def episode_stratum(
    episode: Sequence[object],
    label_key: str = "vf_mining_intervention_label",
) -> tuple[str, bool]:
    task = "unknown"
    has_label = False
    for item in episode:
        if not isinstance(item, dict):
            continue
        if task == "unknown":
            for key in ("task_name", "task", "task_id", "source_task", "task_text"):
                value = item.get(key)
                if value is not None and str(value).strip():
                    task = str(value).strip()
                    break
        if bool(item.get(label_key, False)):
            has_label = True
        if task != "unknown" and has_label:
            break
    return task, has_label


def stratified_episode_split(
    strata: Mapping[tuple[str, bool], Sequence[str]],
    split: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> list[str]:
    """Return a deterministic episode-level split for task/label strata."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unsupported split: {split!r}")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("split ratios must be non-negative")
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("split ratios must sum to 1")

    rng = np.random.default_rng(int(seed))
    selected: list[str] = []
    for key in sorted(strata):
        files = list(strata[key])
        n_total = len(files)
        if n_total == 0:
            continue
        permutation = rng.permutation(n_total)
        files = [files[int(i)] for i in permutation]

        n_train = int(n_total * float(train_ratio))
        n_val = int(n_total * float(val_ratio))
        n_test = max(0, n_total - n_train - n_val)
        if n_total >= 3:
            if n_val == 0:
                n_val = 1
            if n_test == 0:
                n_test = 1
            n_train = max(1, n_total - n_val - n_test)

        val_end = n_train + n_val
        if split == "train":
            selected.extend(files[:n_train])
        elif split == "val":
            selected.extend(files[n_train:val_end])
        else:
            selected.extend(files[val_end:])
    return sorted(selected)
