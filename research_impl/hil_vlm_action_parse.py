"""Parse 7-D actions and action trajectories from model outputs."""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

_DEFAULT_N_BINS = 1000


def _clip_bins(vals: list[int], n_bins: int = _DEFAULT_N_BINS) -> list[int]:
    return [max(0, min(n_bins, int(v))) for v in vals]


def parse_action_bins_strict(text: str) -> Optional[list[int]]:
    """Parse the 7 integers following `Action:`."""
    m = re.search(r"Action:\s*([0-9][0-9\s]*)", text, re.IGNORECASE)
    if not m:
        return None
    parts = m.group(1).strip().split()
    if len(parts) != 7:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None


def _seven_from_int_strings(nums: list[str]) -> Optional[list[int]]:
    if len(nums) < 7:
        return None
    try:
        last7 = [int(nums[-7 + j]) for j in range(7)]
    except ValueError:
        return None
    return _clip_bins(last7)


def parse_action_bins_round2(text: str, n_bins: int = _DEFAULT_N_BINS) -> Optional[list[int]]:
    """Parse the 7 action bins in a Round2 output."""
    strict = parse_action_bins_strict(text)
    if strict is not None:
        return strict

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.match(r"^\s*RewardLabel:\s*\d+", ln, re.IGNORECASE):
            continue
        normalized = re.sub(r",\s*", " ", ln)
        nums = re.findall(r"\b\d+\b", normalized)
        got = _seven_from_int_strings(nums)
        if got is not None:
            return _clip_bins(got, n_bins)

    all_nums = re.findall(r"\b\d+\b", text)
    got = _seven_from_int_strings(all_nums)
    if got is not None:
        return _clip_bins(got, n_bins)
    return None


_STEP_LINE_RE = re.compile(
    r"^\s*step\s+(\d+)\s*:\s*([0-9][0-9\s]*)",
    re.IGNORECASE,
)


def parse_action_trajectory(
    text: str,
    n_bins: int = _DEFAULT_N_BINS,
    *,
    require_trajectory_header: bool = True,
) -> Optional[np.ndarray]:
    """Parse a multi-step trajectory: lines of the form step k: b0 ... b6 (exactly 7 integers per line)."""
    if require_trajectory_header:
        if not re.search(r"Trajectory\s*:", text, re.IGNORECASE):
            return None
    by_step: dict[int, list[int]] = {}
    for ln in text.splitlines():
        m = _STEP_LINE_RE.match(ln.strip())
        if not m:
            continue
        k = int(m.group(1))
        parts = m.group(2).strip().split()
        if len(parts) != 7:
            continue
        try:
            vec = [int(p) for p in parts]
        except ValueError:
            continue
        by_step[k] = _clip_bins(vec, n_bins)
    if not by_step:
        return None
    order = sorted(by_step.keys())
    rows = [by_step[j] for j in order]
    return np.asarray(rows, dtype=np.int64)


def parse_action_trajectory_flexible(
    text: str,
    n_bins: int = _DEFAULT_N_BINS,
    *,
    require_trajectory_header: bool = True,
) -> Optional[np.ndarray]:
    """Parse a multi-step trajectory with variable action dimension: each step k: line has >=3 integers and all lines share the same dimension."""
    if require_trajectory_header and (not re.search(r"Trajectory\s*:", text, re.IGNORECASE)):
        return None

    by_step: dict[int, list[int]] = {}
    dims_seen: set[int] = set()
    for ln in text.splitlines():
        m = _STEP_LINE_RE.match(ln.strip())
        if not m:
            continue
        k = int(m.group(1))
        parts = m.group(2).strip().split()
        if len(parts) < 3:
            continue
        try:
            vec = [int(p) for p in parts]
        except ValueError:
            continue
        dims_seen.add(len(vec))
        by_step[k] = _clip_bins(vec, n_bins)

    if not by_step:
        return None
    if len(dims_seen) != 1:
        return None

    order = sorted(by_step.keys())
    rows = [by_step[j] for j in order]
    return np.asarray(rows, dtype=np.int64)
