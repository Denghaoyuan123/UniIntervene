#!/usr/bin/env python3
"""Overfit official FAST token CE on a few real 7D trajectory chunks."""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research_impl"))
from fast_decoder_v2 import FASTBPETokenizer, FASTBPEActionDecoder, fast_bpe_ce_loss


def load_chunks(buffer_dir: Path, horizon: int, count: int) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    for name in sorted(glob.glob(str(buffer_dir / "transitions_*.pkl"))):
        with open(name, "rb") as handle:  # trusted local robot buffer only
            episode = pickle.load(handle)
        actions = np.asarray([step["actions"] for step in episode], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            continue
        for start in range(0, max(0, len(actions) - horizon + 1), horizon):
            chunk = actions[start : start + horizon]
            if np.isfinite(chunk).all() and len(chunk) == horizon:
                chunks.append(chunk)
                if len(chunks) >= count:
                    return chunks
    return chunks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--buffer-dir", type=Path, required=True)
    p.add_argument("--fast-tokenizer", type=Path, required=True)
    p.add_argument("--chunks", type=int, default=16)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--report", type=Path)
    args = p.parse_args()

    torch.manual_seed(7)
    np.random.seed(7)
    trajectories = load_chunks(args.buffer_dir, args.horizon, args.chunks)
    if len(trajectories) < 2:
        raise SystemExit("need at least two finite 8x7 trajectory chunks")
    tokenizer = FASTBPETokenizer(str(args.fast_tokenizer), 7, args.horizon)
    encoded = [torch.as_tensor(tokenizer.encode(x), dtype=torch.long) for x in trajectories]
    max_length = max(x.numel() for x in encoded)
    targets = torch.full((len(encoded), max_length), tokenizer.pad_id, dtype=torch.long)
    for index, tokens in enumerate(encoded):
        targets[index, : tokens.numel()] = tokens

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    targets = targets.to(device)
    context_dim = 32
    contexts = torch.randn(len(encoded), context_dim, device=device)
    model = FASTBPEActionDecoder(
        backbone_hidden=context_dim,
        vocab_size=tokenizer.model_vocab_size,
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
        max_tokens=max(128, max_length),
        d_dec=64,
        n_layers=2,
        n_heads=4,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    history = []
    for step in range(args.steps + 1):
        logits = model.decode(model.encode_context(contexts), targets)
        metrics = fast_bpe_ce_loss(logits, targets, tokenizer.pad_id)
        history.append(float(metrics["ce"].detach().cpu()))
        if step in {0, 1, 10, 50, 100, args.steps}:
            print(f"step={step:04d} ce={history[-1]:.6f} acc={float(metrics['acc']):.4f}", flush=True)
        if step == args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        metrics["ce"].backward()
        optimizer.step()

    reconstructed = tokenizer.decode(encoded[0].numpy())
    reconstruction_mae = float(np.abs(reconstructed - trajectories[0]).mean())
    initial, final = history[0], history[-1]
    result = {
        "chunks": len(encoded),
        "horizon": args.horizon,
        "action_dim": 7,
        "min_tokens": min(x.numel() for x in encoded),
        "max_tokens": max_length,
        "initial_ce": initial,
        "final_ce": final,
        "loss_ratio": final / initial,
        "final_token_accuracy": float(metrics["acc"].detach().cpu()),
        "tokenizer_reconstruction_mae": reconstruction_mae,
        "passed": bool(final < 0.2 * initial and float(metrics["acc"]) > 0.9),
    }
    print(json.dumps(result, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit("FAST token CE did not satisfy the overfit convergence criterion")


if __name__ == "__main__":
    main()
