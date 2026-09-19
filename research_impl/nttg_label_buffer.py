"""NTTG labeling script: add raw_value/raw_norm labels to a buffer (transitions_*.pkl)."""

from __future__ import annotations

import pickle
import glob
import os
import argparse
import numpy as np

FAIL_VALUE = -500          # value assigned to every failed episode
BUFFER_DIR = os.path.join(os.path.dirname(__file__), "buffer_data/buffer")


def _to_float(x) -> float:
    if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
        if hasattr(x, "flat"):
            return float(x.flat[0])
        return float(x[0])
    return float(x)


def _to_bool(x) -> bool:
    if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
        if hasattr(x, "flat"):
            return bool(x.flat[0])
        return bool(x[0])
    return bool(x)


def _find_success_from_flags(transitions: list[dict]) -> bool | None:
    for tr in reversed(transitions):
        has_s = "success" in tr
        has_f = "failure" in tr
        if not has_s and not has_f:
            continue
        s = _to_bool(tr.get("success", False)) if has_s else False
        f = _to_bool(tr.get("failure", False)) if has_f else False
        if has_s:
            return bool(s)
        if has_f:
            return not bool(f)
    return None


def is_success(transitions: list, *, success_from_labels: bool = False,
               success_label_value: int | None = None) -> bool:
    """Return whether an episode succeeded."""
    by_flags = _find_success_from_flags(transitions)
    if by_flags is not None:
        return by_flags

    if success_from_labels:
        has_labels = any("labels" in tr for tr in transitions)
        if has_labels and success_label_value is not None:
            return any(tr.get("labels") == success_label_value for tr in transitions)

    last = transitions[-1]
    r = last.get("rewards", last.get("reward", 0))
    r = _to_float(r)
    return r > 0


def label_episode(
    transitions: list,
    fail_value: int = FAIL_VALUE,
    *,
    success_from_labels: bool = False,
    success_label_value: int | None = None,
) -> list:
    """Add raw_value/raw_norm labels to the transitions of one episode."""
    T = len(transitions)
    denom = max(T - 1, 1)
    success = is_success(
        transitions,
        success_from_labels=success_from_labels,
        success_label_value=success_label_value,
    )
    if success:
        for t, tr in enumerate(transitions):
            tr["raw_value"] = t - (T - 1)
            tr["raw_norm"] = 1.0 if T == 1 else (float(t) / float(denom))
    else:
        for tr in transitions:
            tr["raw_value"] = fail_value
            tr["raw_norm"] = 0.0
    return transitions


def process_pkl(
    pkl_path: str,
    output_dir: str,
    fail_value: int,
    *,
    success_from_labels: bool = False,
    success_label_value: int | None = None,
):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        print(f"  [skip] {pkl_path}: top-level object is not a list, skipping")
        return

    labeled = label_episode(
        data,
        fail_value,
        success_from_labels=success_from_labels,
        success_label_value=success_label_value,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_name = os.path.basename(pkl_path).replace(".pkl", "_nttg.pkl")
    out_path = os.path.join(output_dir, out_name)
    with open(out_path, "wb") as f:
        pickle.dump(labeled, f)

    success = is_success(
        data,
        success_from_labels=success_from_labels,
        success_label_value=success_label_value,
    )
    print(f"  {'✓ success' if success else '✗ fail   '} {os.path.basename(pkl_path)} "
          f"({len(data)} steps) -> {out_name}")


def main():
    parser = argparse.ArgumentParser(description="NTTG-label a HIL-SERL buffer")
    parser.add_argument("--buffer_dir", default=BUFFER_DIR, help="Directory containing transitions_*.pkl")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: buffer_nttg next to buffer_dir)")
    parser.add_argument("--fail_value", type=int, default=FAIL_VALUE, help="Value assigned to failed episodes")
    parser.add_argument(
        "--success_from_labels",
        action="store_true",
        help="If a 'labels' field exists, success is labels==success_label_value; otherwise rewards>0.",
    )
    parser.add_argument(
        "--success_label_value",
        type=int,
        default=1,
        help="Used with --success_from_labels: episodes whose labels equal this value count as successful (default 1).",
    )
    args = parser.parse_args()

    buffer_dir = args.buffer_dir
    output_dir = args.output_dir or os.path.join(os.path.dirname(buffer_dir), "buffer_nttg")

    pkls = sorted(glob.glob(os.path.join(buffer_dir, "transitions_*.pkl")))
    if not pkls:
        print(f"No pkl files found: {buffer_dir}")
        return

    print(f"{len(pkls)} pkl files, writing to {output_dir}\n")
    success_cnt = 0
    for p in pkls:
        with open(p, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list) and is_success(
            data,
            success_from_labels=args.success_from_labels,
            success_label_value=args.success_label_value,
        ):
            success_cnt += 1
        process_pkl(
            p,
            output_dir,
            args.fail_value,
            success_from_labels=args.success_from_labels,
            success_label_value=args.success_label_value,
        )

    print(f"\nDone: {len(pkls)} episodes, {success_cnt} succeeded, {len(pkls)-success_cnt} failed")
    print(f"Output directory: {output_dir}")
    print("\nSanity-check example:")
    first_nttg = sorted(glob.glob(os.path.join(output_dir, "transitions_*_nttg.pkl")))
    if first_nttg:
        with open(first_nttg[0], "rb") as f:
            ex = pickle.load(f)
        T = len(ex)
        print(
            f"  {os.path.basename(first_nttg[0])}: T={T}, "
            f"raw_value[0]={ex[0]['raw_value']}, raw_value[-1]={ex[-1]['raw_value']}, "
            f"raw_norm[0]={ex[0].get('raw_norm')}, raw_norm[-1]={ex[-1].get('raw_norm')}"
        )


if __name__ == "__main__":
    main()
