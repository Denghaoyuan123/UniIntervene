from __future__ import annotations


def segment_is_eligible(
    start_vf: float,
    end_vf: float,
    delta_threshold: float,
    target_vf_threshold: float,
) -> bool:
    return (
        float(end_vf) - float(start_vf) >= float(delta_threshold)
        and float(end_vf) >= float(target_vf_threshold)
    )


def recovery_chunk_indices(start: int, end: int, horizon: int) -> list[int]:
    if int(start) < 0 or int(end) < int(start):
        raise ValueError("invalid recovery segment bounds")
    if int(horizon) <= 0:
        raise ValueError("recovery horizon must be positive")
    return list(range(int(start), min(int(end) + 1, int(start) + int(horizon))))
