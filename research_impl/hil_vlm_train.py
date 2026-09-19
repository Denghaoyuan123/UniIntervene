"""Train the VQ1 and VQ2 models."""

import argparse
import csv
import json
import os
import pickle
import re
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from transformers import AutoProcessor, BitsAndBytesConfig
import wandb

from hil_vlm_action_parse import parse_action_trajectory, parse_action_trajectory_flexible
from hil_vlm_dataset import HILVLMDataset, N_BINS, RecoverTargetMiningConfig
from fast_decoder import FastDecoder, fast_decoder_losses, fast_decoder_total_loss, build_fast_decoder_gt
from fast_decoder_v2 import (
    FastDecoderV2,
    FastTokenizer,
    build_fast_decoder_v2_gt_tokens,
    fast_decoder_v2_ce_loss,
    FASTBPETokenizer,
    FASTBPEActionDecoder,
    build_fast_bpe_gt_tokens,
    fast_bpe_ce_loss,
)
from vq1_heads import (
    VQ_TOKEN,
    add_vq_query_token,
    extract_h_t,
    FutureHead,
    TwinQHead,
    InterventionHead,
    ValueHistoryEncoder,
    VCurrHead,
    RiskHead,
    FutureTargetEncoder,
    normalized_mse,
    cosine_sim_mean,
    binary_focal_loss_with_logits,
    threshold_sweep_metrics,
    get_variant,
    VARIANT_SPEC,
)




def log(msg: str):
    print(msg, flush=True)


def _register_train_aux_for_validation(train_ds, val_ds) -> None:
    val_ds.register_aux_transitions(
        transitions=train_ds.transitions,
        episode_spans=train_ds.episode_spans,
        episode_stems=train_ds._episode_stems,
    )
    if hasattr(val_ds, "rebuild_round2_green_pool"):
        val_ds.rebuild_round2_green_pool()
    log(f"[data] registered train-only aux transitions for validation lookup: {len(train_ds.transitions)}")


def _collect_round2_retrieval_rows(batch: list[dict], epoch: int, split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample in batch:
        if not bool(sample.get("has_round2", False)):
            continue
        rows.append(
            {
                "epoch": int(epoch),
                "split": str(split),
                "flat_idx": int(sample.get("flat_idx", -1)),
                "task_name": str(sample.get("task_name", "") or ""),
                "task_route_key": str(sample.get("task_route_key", "") or ""),
                "query_task": str(sample.get("round2_memory_query_task", "") or ""),
                "query_route_key": str(sample.get("round2_memory_query_route_key", "") or ""),
                "bank_path": str(sample.get("round2_memory_bank_path", "") or ""),
                "bank_route_key": str(sample.get("round2_memory_route_key", "") or ""),
                "bank_item_index": int(sample.get("round2_memory_item_index", -1) or -1),
                "bank_rank": int(sample.get("round2_memory_rank", -1) or -1),
                "bank_score": float(sample.get("round2_memory_score", float("nan"))),
                "bank_task_filter": str(sample.get("round2_memory_task_filter", "") or ""),
                "bank_item_task_text": str(sample.get("round2_memory_item_task_text", "") or ""),
                "bank_item_task_id": str(sample.get("round2_memory_item_task_id", "") or ""),
                "bank_episode_id": str(sample.get("round2_memory_episode_id", "") or ""),
                "bank_failure_index": int(sample.get("round2_memory_failure_index", -1) or -1),
                "bank_target_index": int(sample.get("round2_memory_target_index", -1) or -1),
                "round2_goal_vf_delta": float(sample.get("round2_goal_vf_delta", 0.0) or 0.0),
                "round2_improvement": float(sample.get("round2_improvement", 0.0) or 0.0),
                "round2_quality_score": float(sample.get("round2_quality_score", 0.0) or 0.0),
            }
        )
    return rows


def _write_round2_retrieval_audit(rows: list[dict[str, object]], out_csv: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_qwen_vl_model_cls(model_path: str | None = None):
    """Pick the Qwen-VL model class for ``model_path`` from its config, falling back to Qwen3-VL then Qwen2.5-VL."""
    arch_hint = None
    model_type_hint = None
    if model_path:
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
            arch_list = list(getattr(cfg, "architectures", []) or [])
            if arch_list:
                arch_hint = str(arch_list[0])
            model_type_hint = str(getattr(cfg, "model_type", "") or "")
        except Exception as exc:
            print(f"[model] AutoConfig probe failed for {model_path}: {exc}")

    def _try(name: str):
        try:
            import importlib
            mod = importlib.import_module("transformers")
            cls = getattr(mod, name, None)
            return cls
        except Exception:
            return None

    if arch_hint:
        cls = _try(arch_hint)
        if cls is not None:
            return cls, arch_hint
    if model_type_hint == "qwen2_5_vl":
        cls = _try("Qwen2_5_VLForConditionalGeneration")
        if cls is not None:
            return cls, "Qwen2_5_VLForConditionalGeneration"
    if model_type_hint == "qwen3_vl":
        cls = _try("Qwen3VLForConditionalGeneration")
        if cls is not None:
            return cls, "Qwen3VLForConditionalGeneration"

    for name in ("Qwen3VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration"):
        cls = _try(name)
        if cls is not None:
            return cls, name
    raise ImportError(
        "Neither Qwen3VLForConditionalGeneration nor Qwen2_5_VLForConditionalGeneration "
        "is available in this transformers installation."
    )


def _cuda_bf16_autocast(device: torch.device, enabled: bool):
    """On some CUDA/PEFT combinations the default fp32 LoRA branch can raise cuBLASLt errors; wrapping the CUDA forward in bf16 autocast avoids it."""
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def make_nttg_weights(transitions):
    """Inverse-frequency weights (NTTG bin balancing)."""
    bins = np.array([t.get("raw_value", -500) for t in transitions])
    b_norm = np.clip(bins / 500.0 + 1.0, 0.0, 1.0) * (N_BINS - 1)
    b_idx = np.round(b_norm).astype(int)
    counts = np.bincount(b_idx, minlength=N_BINS).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    w = 1.0 / counts[b_idx]
    return w / w.mean()


def apply_binary_mask_boost(
    weights: np.ndarray,
    mask: np.ndarray,
    boost: float,
) -> tuple[np.ndarray, dict[str, float]] | tuple[None, None]:
    """Upweight sampling of a binary subset and return statistics of its effective share."""
    b = float(boost)
    if b <= 1.0:
        return None, None
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.size <= 0:
        return None, None
    n_hit = int(m.sum())
    if n_hit <= 0:
        return None, None
    w = np.asarray(weights, dtype=float).reshape(-1)
    w_before = w.copy()
    w = w * (1.0 + (b - 1.0) * m.astype(float))
    w = w / max(w.mean(), 1e-12)
    stats = {
        "n_hit": float(n_hit),
        "n_total": float(m.size),
        "before_min": float(np.min(w_before)),
        "before_max": float(np.max(w_before)),
        "after_min": float(np.min(w)),
        "after_max": float(np.max(w)),
        "effective_share": float(w[m].sum() / max(w.sum(), 1e-12)),
    }
    return w, stats


def _tail_token_ce(logits: torch.Tensor, labels: torch.Tensor, last_k: int) -> torch.Tensor | None:
    """Aux CE on the last supervised tokens (emphasizes the trajectory tail / endpoint)."""
    k = max(0, int(last_k))
    if k <= 0:
        return None
    if logits is None or labels is None or logits.ndim != 3 or labels.ndim != 2:
        return None
    if logits.shape[0] < 1 or labels.shape[0] < 1 or logits.shape[1] < 2 or labels.shape[1] < 2:
        return None
    sl = labels[0, 1:].contiguous()
    sm = sl.ne(-100)
    pos = torch.where(sm)[0]
    if pos.numel() <= 0:
        return None
    pos = pos[-k:]
    lg = logits[0, :-1, :].contiguous()[pos]
    tg = sl[pos]
    if lg.numel() <= 0 or tg.numel() <= 0:
        return None
    return F.cross_entropy(lg, tg, reduction="mean")


_INT_RE = re.compile(r"\b\d+\b")


def _parse_target_trajectory_from_sample(sample: dict) -> np.ndarray | None:
    txt = str(sample.get("round2_target", "") or "")
    return parse_action_trajectory_flexible(txt, n_bins=1000, require_trajectory_header=True)


def _state_stats_from_sample(sample: dict) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    mean = sample.get("state_norm_mean")
    std = sample.get("state_norm_std")
    clip_k = float(sample.get("state_clip_k", 4.0))
    if mean is None or std is None:
        return None, None, clip_k
    m = np.asarray(mean, dtype=np.float64).reshape(-1)
    sd = np.asarray(std, dtype=np.float64).reshape(-1)
    if m.size <= 0 or sd.size <= 0:
        return None, None, clip_k
    return m, np.maximum(sd, 1e-8), clip_k


def _dequantize_bins_state_norm(bins: np.ndarray, clip_k: float, n_bins: int) -> np.ndarray:
    b = np.asarray(bins, dtype=np.float64)
    t = b / float(max(1, int(n_bins)))
    return (2.0 * float(clip_k)) * t - float(clip_k)


def _dequantize_bins_state_norm_torch(bins: torch.Tensor, clip_k: float, n_bins: int) -> torch.Tensor:
    t = bins.to(dtype=torch.float32) / float(max(1, int(n_bins)))
    return (2.0 * float(clip_k)) * t - float(clip_k)


def _parse_goal_bins_from_sample(sample: dict, n_bins: int, action_dim: int = 7) -> np.ndarray | None:
    txt = str(sample.get("round2_goal_user_block", "") or "")
    nums = _INT_RE.findall(txt)
    d = max(1, int(action_dim))
    if len(nums) < d:
        return None
    try:
        arr = np.asarray([int(x) for x in nums[-d:]], dtype=np.int64)
    except Exception:
        return None
    arr = np.clip(arr, 0, int(max(1, n_bins))).astype(np.int64)
    return arr


def _tail_step_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    total_steps: int,
    tail_steps: int,
) -> torch.Tensor | None:
    """Approx step-aware CE: map last-k steps to last-k*tokens_per_step supervised tokens."""
    ts = max(0, int(total_steps))
    tk = max(0, int(tail_steps))
    if ts <= 0 or tk <= 0:
        return None
    if logits is None or labels is None or logits.ndim != 3 or labels.ndim != 2:
        return None
    if logits.shape[0] < 1 or labels.shape[0] < 1 or logits.shape[1] < 2 or labels.shape[1] < 2:
        return None
    sl = labels[0, 1:].contiguous()
    pos = torch.where(sl.ne(-100))[0]
    if pos.numel() <= 0:
        return None
    tokens_per_step = max(1, int(pos.numel() // max(1, ts)))
    k_tokens = min(int(pos.numel()), max(tokens_per_step * tk, 6 * tk))
    take = pos[-k_tokens:]
    lg = logits[0, :-1, :].contiguous()[take]
    tg = sl[take]
    if lg.numel() <= 0 or tg.numel() <= 0:
        return None
    return F.cross_entropy(lg, tg, reduction="mean")


def _round2_tail_weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    total_steps: int,
    tail_steps: int,
    tail_boost: float,
) -> torch.Tensor | None:
    """Main CE with stronger supervision on tail tokens."""
    ts = max(0, int(total_steps))
    tk = max(0, int(tail_steps))
    boost = max(0.0, float(tail_boost))
    if ts <= 0 or tk <= 0 or boost <= 0.0:
        return None
    if logits is None or labels is None or logits.ndim != 3 or labels.ndim != 2:
        return None
    if logits.shape[0] < 1 or labels.shape[0] < 1 or logits.shape[1] < 2 or labels.shape[1] < 2:
        return None

    sl = labels[0, 1:].contiguous()
    pos = torch.where(sl.ne(-100))[0]
    if pos.numel() <= 0:
        return None

    tokens_per_step = max(1, int(pos.numel() // max(1, ts)))
    k_tokens = min(int(pos.numel()), max(tokens_per_step * tk, 6 * tk))
    tail_pos = pos[-k_tokens:]

    lg = logits[0, :-1, :].contiguous()[pos]
    tg = sl[pos]
    if lg.numel() <= 0 or tg.numel() <= 0:
        return None

    ce = F.cross_entropy(lg, tg, reduction="none")
    w = torch.ones_like(ce)
    tail_mask = torch.zeros_like(w, dtype=torch.bool)
    tail_mask[-k_tokens:] = True
    w[tail_mask] = w[tail_mask] * (1.0 + boost)
    den = torch.clamp(w.sum(), min=1e-12)
    return (ce * w).sum() / den


def _round2_smoothness_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    bin_token_ids: torch.Tensor,
    bin_values: torch.Tensor,
    action_dim: int,
) -> torch.Tensor | None:
    """Mean L2 step-difference on expected action bins; smaller is smoother."""
    exp_bins = _expected_bins_from_logits(
        logits,
        labels,
        bin_token_ids=bin_token_ids,
        bin_values=bin_values,
        take_last_bins=512,
    )
    if exp_bins is None or int(exp_bins.numel()) < (2 * int(action_dim)):
        return None
    steps = int(exp_bins.numel() // int(action_dim))
    if steps < 2:
        return None
    traj = exp_bins[: steps * int(action_dim)].reshape(steps, int(action_dim))
    d = traj[1:] - traj[:-1]
    return torch.linalg.norm(d, ord=2, dim=-1).mean()


def _dequantize_bins_action(bins: np.ndarray, a_min: np.ndarray, a_max: np.ndarray, n_bins: int) -> np.ndarray:
    b = np.asarray(bins, dtype=np.float64).reshape(-1)
    lo = np.asarray(a_min, dtype=np.float64).reshape(-1)
    hi = np.asarray(a_max, dtype=np.float64).reshape(-1)
    x = b / float(max(1, int(n_bins)))
    return x * (hi - lo) + lo


def _dequantize_bins_action_torch(
    bins: torch.Tensor,
    a_min: torch.Tensor,
    a_max: torch.Tensor,
    n_bins: int,
) -> torch.Tensor:
    x = bins.to(dtype=torch.float32) / float(max(1, int(n_bins)))
    return x * (a_max - a_min) + a_min


def _goal_alignment_scale(sample: dict, a_min: np.ndarray, a_max: np.ndarray, n_bins: int) -> float:
    """Goal alignment confidence computed in active target space."""
    gt = _parse_target_trajectory_from_sample(sample)
    if gt is None or gt.shape[0] <= 0:
        return 1.0
    d = int(gt.shape[-1])
    gb = _parse_goal_bins_from_sample(sample, n_bins=n_bins, action_dim=d)
    if gb is None or int(gb.shape[0]) != d:
        return 1.0

    if str(sample.get("round2_target_space", "action")).strip().lower() == "state":
        _m, _s, clip_k = _state_stats_from_sample(sample)
        gt_last = _dequantize_bins_state_norm(gt[-1], clip_k, n_bins)
        goal = _dequantize_bins_state_norm(gb, clip_k, n_bins)
    else:
        gt_last = _dequantize_bins_action(gt[-1], a_min[:d], a_max[:d], n_bins)
        goal = _dequantize_bins_action(gb, a_min[:d], a_max[:d], n_bins)

    gap = float(np.mean(np.abs(gt_last - goal)))
    return float(np.clip(1.0 / (1.0 + gap), 0.2, 1.0))


def _build_bin_token_table(tokenizer, n_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids: list[int] = []
    vals: list[float] = []
    max_bin = int(max(1, n_bins))
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_size <= 0:
        try:
            vocab_size = len(tokenizer)
        except Exception:
            vocab_size = 0
    num_re = re.compile(r"^\d{1,4}$")
    for tid in range(max(0, vocab_size)):
        try:
            txt = tokenizer.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception:
            continue
        if not isinstance(txt, str) or not txt:
            continue
        norm = txt.replace("▁", " ").replace("Ġ", " ").strip()
        if not norm:
            continue
        if not num_re.fullmatch(norm):
            continue
        try:
            v = int(norm)
        except Exception:
            continue
        if 0 <= v <= max_bin:
            ids.append(int(tid))
            vals.append(float(v))
    if not ids:
        for i in range(max_bin + 1):
            try:
                toks = tokenizer.encode(str(i), add_special_tokens=False)
            except Exception:
                toks = []
            if isinstance(toks, list) and len(toks) == 1:
                ids.append(int(toks[0]))
                vals.append(float(i))
    if not ids:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.float32)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(vals, dtype=torch.float32)


def _expected_bins_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    bin_token_ids: torch.Tensor,
    bin_values: torch.Tensor,
    take_last_bins: int,
) -> torch.Tensor | None:
    if logits is None or labels is None or logits.ndim != 3 or labels.ndim != 2:
        return None
    if logits.shape[0] < 1 or labels.shape[0] < 1 or int(take_last_bins) <= 0:
        return None
    if bin_token_ids.numel() <= 0 or bin_values.numel() <= 0:
        return None
    sl = labels[0, 1:].contiguous()
    lg = logits[0, :-1, :].contiguous()
    pos = torch.where(sl.ne(-100))[0]
    if pos.numel() <= 0:
        return None
    gt_ids = sl[pos]
    try:
        numeric_mask = torch.isin(gt_ids, bin_token_ids.to(gt_ids.device))
    except Exception:
        keep = set(int(x) for x in bin_token_ids.detach().cpu().tolist())
        numeric_mask = torch.tensor(
            [int(x) in keep for x in gt_ids.detach().cpu().tolist()],
            dtype=torch.bool,
            device=gt_ids.device,
        )
    num_pos = pos[numeric_mask]
    need = int(take_last_bins)
    if num_pos.numel() < need:
        return None
    take = num_pos[-need:]
    bt = bin_token_ids.to(device=lg.device)
    bv = bin_values.to(device=lg.device, dtype=torch.float32)
    logits_bins = lg[take][:, bt]
    probs = torch.softmax(logits_bins.float(), dim=-1)
    exp_bins = torch.matmul(probs, bv)
    return exp_bins


def _round2_geo_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    sample: dict,
    a_min: np.ndarray,
    a_max: np.ndarray,
    n_bins: int,
    bin_token_ids: torch.Tensor,
    bin_values: torch.Tensor,
    tail_k: int,
    margin: float,
    vgain_tau: float,
    gate_mode: str = "goal_vf_delta",
    gate_quantile_tau: float = 0.0,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    gt = _parse_target_trajectory_from_sample(sample)
    if gt is None or gt.shape[0] <= 0:
        return None, None
    action_dim = int(gt.shape[-1])
    mode = str(gate_mode or "goal_vf_delta").strip().lower()
    if mode == "none":
        pass
    elif mode == "round2_improvement":
        imp = float(sample.get("round2_improvement", 0.0))
        if imp <= float(vgain_tau):
            return None, None
    elif mode == "goal_vf_delta":
        dvg = float(sample.get("round2_goal_vf_delta", 0.0))
        if dvg <= float(vgain_tau):
            return None, None
    elif mode == "goal_vf_delta_quantile":
        dvg = float(sample.get("round2_goal_vf_delta", 0.0))
        if dvg <= float(gate_quantile_tau):
            return None, None
    else:
        dvg = float(sample.get("round2_goal_vf_delta", 0.0))
        if dvg <= float(vgain_tau):
            return None, None

    goal_bins_np = _parse_goal_bins_from_sample(sample, n_bins=n_bins, action_dim=action_dim)
    if goal_bins_np is None or int(goal_bins_np.shape[0]) != action_dim:
        return None, None
    exp_last = _expected_bins_from_logits(
        logits,
        labels,
        bin_token_ids=bin_token_ids,
        bin_values=bin_values,
        take_last_bins=action_dim,
    )
    if exp_last is None or int(exp_last.shape[0]) != action_dim:
        return None, None

    dev = exp_last.device
    goal_bins_t = torch.as_tensor(goal_bins_np.astype(np.float32), device=dev)
    if str(sample.get("round2_target_space", "action")).strip().lower() == "state":
        _m, _s, clip_k = _state_stats_from_sample(sample)
        pred_last_c = _dequantize_bins_state_norm_torch(exp_last, clip_k, n_bins)
        goal_c = _dequantize_bins_state_norm_torch(goal_bins_t, clip_k, n_bins)
    else:
        a_min_t = torch.as_tensor(np.asarray(a_min[:action_dim], dtype=np.float32), device=dev)
        a_max_t = torch.as_tensor(np.asarray(a_max[:action_dim], dtype=np.float32), device=dev)
        pred_last_c = _dequantize_bins_action_torch(exp_last, a_min_t, a_max_t, n_bins)
        goal_c = _dequantize_bins_action_torch(goal_bins_t, a_min_t, a_max_t, n_bins)

    l_end = F.smooth_l1_loss(pred_last_c, goal_c, reduction="mean", beta=0.2)

    tk = max(1, int(tail_k))
    exp_tail = _expected_bins_from_logits(
        logits,
        labels,
        bin_token_ids=bin_token_ids,
        bin_values=bin_values,
        take_last_bins=action_dim * tk,
    )
    if exp_tail is None or int(exp_tail.numel()) < (2 * action_dim):
        return l_end, None
    steps = int(exp_tail.numel() // action_dim)
    if steps <= 1:
        return l_end, None

    exp_tail = exp_tail.reshape(steps, action_dim)
    if str(sample.get("round2_target_space", "action")).strip().lower() == "state":
        _m, _s, clip_k = _state_stats_from_sample(sample)
        pred_tail_c = _dequantize_bins_state_norm_torch(exp_tail, clip_k, n_bins)
    else:
        a_min_t = torch.as_tensor(np.asarray(a_min[:action_dim], dtype=np.float32), device=dev)
        a_max_t = torch.as_tensor(np.asarray(a_max[:action_dim], dtype=np.float32), device=dev)
        pred_tail_c = _dequantize_bins_action_torch(exp_tail, a_min_t[None, :], a_max_t[None, :], n_bins)

    goal_tail_c = goal_c[None, :].expand_as(pred_tail_c)
    d = torch.linalg.norm(pred_tail_c - goal_tail_c, ord=2, dim=-1)
    l_tail = torch.relu(d[1:] - d[:-1] + float(margin)).mean()
    return l_end, l_tail


def _geo_weights_for_epoch(args, epoch: int) -> tuple[float, float]:
    w_end = float(getattr(args, "round2_geo_end_weight", 0.0))
    w_tail = float(getattr(args, "round2_geo_tail_weight", 0.0))
    warm = max(0, int(getattr(args, "round2_geo_warmup_epochs", 0)))
    if epoch <= warm:
        return 0.0, 0.0
    if epoch == warm + 1:
        return min(w_end, 0.10), min(w_tail, 0.03)
    return w_end, w_tail


def _eos_early_penalty(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    eos_token_id: int,
    total_steps: int,
    floor_ratio: float,
) -> torch.Tensor | None:
    """Discourage early stop tendency: penalize EOS probability before floor-length token window."""
    if eos_token_id < 0:
        return None
    ts = max(0, int(total_steps))
    if ts <= 0:
        return None
    fr = max(0.0, float(floor_ratio))
    if fr <= 0.0:
        return None
    if logits is None or labels is None or logits.ndim != 3 or labels.ndim != 2:
        return None
    sl = labels[0, 1:].contiguous()
    pos = torch.where(sl.ne(-100))[0]
    if pos.numel() <= 1:
        return None
    tokens_per_step = max(1, int(pos.numel() // max(1, ts)))
    floor_steps = max(1, int(np.ceil(fr * float(ts))))
    floor_tokens = min(int(pos.numel()), max(1, floor_steps * tokens_per_step))
    early = pos[:floor_tokens]
    if early.numel() <= 0:
        return None
    lg = logits[0, :-1, :].contiguous()[early]
    eos_prob = torch.softmax(lg, dim=-1)[:, int(eos_token_id)]
    return torch.mean(eos_prob)


class TwinVFHeads(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.vf_head1 = nn.Linear(hidden_size, 1)
        self.vf_head2 = nn.Linear(hidden_size, 1)
        self.reinit()

    def reinit(self):
        for m in (self.vf_head1, self.vf_head2):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p1 = self.vf_head1(x).squeeze(-1)
        p2 = self.vf_head2(x).squeeze(-1)
        return p1, p2


def _build_episode_bounds_map(episode_spans: list[tuple[int, int]], n_total: int) -> np.ndarray:
    bounds = np.full((n_total, 2), -1, dtype=np.int64)
    for s, e in episode_spans:
        s0, e0 = int(s), int(e)
        if s0 < 0 or e0 <= s0 or e0 > n_total:
            continue
        bounds[s0:e0, 0] = s0
        bounds[s0:e0, 1] = e0
    return bounds


def _twin_targets_from_sample(
    sample: dict,
    transitions: list[dict],
    ep_bounds: np.ndarray,
    device: torch.device,
    gamma: float,
) -> dict | None:
    idx = int(sample.get("flat_idx", -1))
    if idx < 0 or idx >= len(transitions) or idx >= int(ep_bounds.shape[0]):
        return None
    s0, e0 = int(ep_bounds[idx, 0]), int(ep_bounds[idx, 1])
    if s0 < 0 or e0 <= s0:
        return None
    tr = transitions[idx]
    cur_v = float(sample.get("vf_value_pred", tr.get("vf_value_pred", 0.0)))
    return {
        "vf_target": torch.tensor([cur_v], dtype=torch.float32, device=device),
    }


def _pool_hidden_from_outputs(outputs) -> torch.Tensor | None:
    hs = getattr(outputs, "hidden_states", None)
    if hs is None or len(hs) <= 0:
        return None
    last = hs[-1]
    if last is None or last.ndim != 3 or last.shape[0] <= 0:
        return None
    return last[:, -1, :].contiguous()


def _pool_hidden_from_outputs_with_mask(
    outputs,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    hs = getattr(outputs, "hidden_states", None)
    if hs is None or len(hs) <= 0:
        return None
    last = hs[-1]
    if last is None or last.ndim != 3 or last.shape[0] <= 0:
        return None
    if attention_mask is None or attention_mask.ndim != 2:
        return last[:, -1, :].contiguous()
    mask = attention_mask.long()
    seq_lens = mask.size(1) - 1 - torch.flip(mask, dims=[1]).argmax(dim=1)
    b = torch.arange(last.size(0), device=last.device, dtype=torch.long)
    return last[b, seq_lens].contiguous()


def _compute_twin_loss(
    pred1: torch.Tensor,
    pred2: torch.Tensor,
    next_pred_min: torch.Tensor,
    tgt: dict,
    args,
) -> tuple[torch.Tensor, dict]:
    _ = next_pred_min
    vf_target = tgt["vf_target"]
    loss_vf = 0.5 * (F.mse_loss(pred1, vf_target) + F.mse_loss(pred2, vf_target))
    loss_td = loss_vf
    loss_prog = torch.zeros_like(loss_vf)
    loss_mc = torch.zeros_like(loss_vf)
    total = loss_vf
    stats = {
        "loss_td": float(loss_td.detach().item()),
        "loss_prog": float(loss_prog.detach().item()),
        "loss_mc": float(loss_mc.detach().item()),
        "pred1": float(pred1.detach().mean().item()),
        "pred2": float(pred2.detach().mean().item()),
        "pred_min": float(torch.minimum(pred1, pred2).detach().mean().item()),
    }
    return total, stats


def make_indicator_weights(transitions):
    """VF indicator sampling weights: 2.0 for good frames, 0.5 for bad frames, 1.0 when unscored, normalized to mean 1."""
    w = np.array([
        2.0 if t.get("indicator", -1) == 1 else
        0.5 if t.get("indicator", -1) == 0 else
        1.0
        for t in transitions
    ], dtype=float)
    return w / w.mean()


def collate_fn(batch):
    return batch   # fields have different lengths; tokenized per sample


def _indicator_target(sample: dict) -> str:
    """Intervention training target: depends only on the human intervention label intervened."""
    return "yes" if sample.get("intervened", False) else "no"


def build_chat_input(sample, processor, system_msg, device, use_indicator: bool = False):
    """Convert one sample into Qwen3-VL inputs."""
    target = sample["target"]
    user_text = build_stage1_user_text(sample)
    msgs = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in sample["images"]],
                {"type": "text", "text": user_text},
            ],
        },
        {"role": "assistant", "content": target},
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    inputs = processor(
        text=[text],
        images=sample["images"] or None,
        return_tensors="pt",
        padding=True,
    ).to(device)
    return inputs


def build_prompt_only_input(sample, processor, system_msg, device):
    """Build the prompt-only input for the twin heads."""
    user_text = build_stage1_user_text(sample)
    msgs = [
        {"role": "system", "content": system_msg},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in sample["images"]],
                {"type": "text", "text": user_text},
            ],
        },
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return processor(
        text=[text],
        images=sample["images"] or None,
        return_tensors="pt",
        padding=True,
    ).to(device)


def build_round2_prompt_input(sample, processor, device):
    """Goal-conditioned prompt for FAST decoder: A+B frame images + round2 system msg + instruction + suffix."""
    sys_msg, suffix = _round2_msg_suffix(sample)
    imgs = sample.get("round2_images") or sample["images"]
    user_body = f"{sample['instr']}{suffix}"
    goal_block = sample.get("round2_goal_user_block", "")
    if goal_block:
        user_body = f"{user_body}\n\n{goal_block}"
    msgs = [
        {"role": "system", "content": sys_msg},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in imgs],
                {"type": "text", "text": user_body},
            ],
        },
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return processor(
        text=[text],
        images=imgs or None,
        return_tensors="pt",
        padding=True,
    ).to(device)


SYSTEM_MSG = (
    "You are a robot control assistant. "
    "Given robot camera images and a task description, output exactly:\n"
    "Intervention: yes/no"
)

SYSTEM_MSG_ROUND2_ACTION = (
    "You are a robot control assistant. "
    "You will see images before intervention and at the recovery target, "
    "plus a single target goal state as 7 action bins (discretized). "
    "Generate a multi-step trajectory that connects the intervention state to that goal. "
    "The trajectory should not stop early: continue until the end state is close to the target goal state. "
    "Make the final 3 steps converge toward the target goal state, and make the last step as close as possible to the target bins. "
    "Output exactly in this format:\n"
    "Trajectory:\n"
    "step 0: <7 integers in [0,1000] separated by spaces>\n"
    "step 1: ...\n"
)

SYSTEM_MSG_ROUND2_STATE = (
    "You are a robot control assistant. "
    "You will see images before intervention and at the recovery target, "
    "plus a single target goal state as normalized state bins. "
    "Generate a multi-step state trajectory toward that goal. "
    "The trajectory should not stop early and should converge at the tail. "
    "Output exactly in this format:\n"
    "Trajectory:\n"
    "step 0: <D integers in [0,1000] separated by spaces>\n"
    "step 1: ...\n"
)

SYSTEM_MSG_ROUND2 = SYSTEM_MSG_ROUND2_ACTION

ROUND2_USER_TEXT_SUFFIX_ACTION = (
    "\n\nYou will be shown images at intervention and at the recovery target, "
    "and the target recovery goal state (7 bins). "
    "Predict the full recovery action trajectory toward that goal. "
    "Ensure sufficient trajectory length and a clear final convergence to the target goal state."
)

ROUND2_USER_TEXT_SUFFIX_STATE = (
    "\n\nYou will be shown images at intervention and at the recovery target, "
    "and the target recovery normalized state bins. "
    "Predict the full recovery state trajectory toward that goal. "
    "Ensure sufficient trajectory length and a clear final convergence to the target goal state."
)


def _round2_msg_suffix(sample: dict) -> tuple[str, str]:
    if str(sample.get("round2_target_space", "action")).strip().lower() == "state":
        return SYSTEM_MSG_ROUND2_STATE, ROUND2_USER_TEXT_SUFFIX_STATE
    return SYSTEM_MSG_ROUND2_ACTION, ROUND2_USER_TEXT_SUFFIX_ACTION


def build_stage1_user_text(sample: dict) -> str:
    """Round1 user text: in state mode use the current normalized state, otherwise fall back to action text."""
    if str(sample.get("round2_target_space", "action")).strip().lower() == "state":
        txt = str(sample.get("current_state_text", "") or "").strip()
        if txt:
            return f"{sample['instr']}\n\n{txt}"
    return f"{sample['instr']}\n\n{sample['current_action_text']}"


def build_chat_input_round2(sample: dict, processor, device):
    """Round2 SFT: two frame images + user (instruction + suffix + goal block) + assistant trajectory text."""
    if not sample.get("has_round2") or not sample.get("round2_target"):
        raise ValueError("build_chat_input_round2 requires has_round2 and round2_target")
    sys_msg, suffix = _round2_msg_suffix(sample)
    user_body = (
        f"{sample['instr']}{suffix}\n\n"
        f"{sample.get('round2_goal_user_block', '')}"
    )
    target = sample["round2_target"]
    imgs = sample["round2_images"] or []
    msgs = [
        {"role": "system", "content": sys_msg},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in imgs],
                {"type": "text", "text": user_body},
            ],
        },
        {"role": "assistant", "content": target},
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return processor(
        text=[text],
        images=imgs or None,
        return_tensors="pt",
        padding=True,
    ).to(device)


def _stage1_prompt_len(processor, sample) -> int:
    user_text = build_stage1_user_text(sample)
    text_full = processor.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in sample["images"]],
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = processor(text=[text_full], return_tensors="pt").input_ids
    return int(prompt_ids.shape[1])


def _round2_prompt_len(processor, sample) -> int:
    sys_msg, suffix = _round2_msg_suffix(sample)
    user_body = (
        f"{sample['instr']}{suffix}\n\n"
        f"{sample.get('round2_goal_user_block', '')}"
    )
    imgs = sample["round2_images"] or []
    text_full = processor.apply_chat_template(
        [
            {"role": "system", "content": sys_msg},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in imgs],
                    {"type": "text", "text": user_body},
                ],
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = processor(text=[text_full], return_tensors="pt").input_ids
    return int(prompt_ids.shape[1])




def _next_step_indices(ep_bounds: np.ndarray, flat_idx: int) -> tuple[int, bool]:
    """Return (next_flat_idx, is_terminal). At episode end we clamp to self."""
    fi = int(flat_idx)
    if fi < 0 or fi >= int(ep_bounds.shape[0]):
        return fi, True
    s0, e0 = int(ep_bounds[fi, 0]), int(ep_bounds[fi, 1])
    if s0 < 0 or e0 <= s0:
        return fi, True
    nxt = fi + 1
    if nxt >= e0:
        return fi, True
    return nxt, False


def _proxy_v_tp1(
    transitions: list,
    ep_bounds: np.ndarray,
    flat_idx: int,
) -> tuple[float, bool, bool, int]:
    """proxy V(t+1) from dataset vf_value_pred at the next transition."""
    nxt, terminal, fallback = _resolve_next_idx_for_v(transitions, ep_bounds, flat_idx)
    tr = transitions[nxt]
    try:
        v = float(tr.get("vf_value_pred", 0.0))
    except Exception:
        v = 0.0
    return v, bool(terminal), bool(fallback), int(nxt)


def _episode_stem_for_idx(dataset, flat_idx: int) -> str:
    try:
        stem, _ = dataset.episode_stem_and_local_index(int(flat_idx))
        return str(stem)
    except Exception:
        return ""


def _next_obs_uint8_views(
    transitions: list,
    flat_idx: int,
    use_external_wrist_keys: bool = False,
) -> list:
    """Return a list of uint8 ndarray views for o_{t+1}, taken directly from"""
    tr = transitions[int(flat_idx)]
    no = tr.get("next_observations", {}) or {}
    out = []
    keys = ("external", "wrist") if bool(use_external_wrist_keys) else ("side_policy_256", "wrist_1")
    for key in keys:
        if key in no:
            arr = np.asarray(no[key])
            if arr.ndim == 4:                                   # (1, H, W, 3)
                arr = arr[0]
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            out.append(arr)
    return out


def _value_history_window(
    flat_idx: int,
    transitions: list,
    ep_bounds: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect K+1 values and K deltas for the ValueHistoryEncoder."""
    fi = int(flat_idx)
    ep_start = int(ep_bounds[fi, 0]) if (fi >= 0 and fi < ep_bounds.shape[0]) else fi
    values_rev = []  # V_t first, then going backward
    for k in range(K + 1):
        idx = max(ep_start, fi - k)
        try:
            v = float(transitions[idx].get("vf_value_pred", 0.0) or 0.0)
        except Exception:
            v = 0.0
        values_rev.append(v)
    values = list(reversed(values_rev))       # chronological: [V_{t-K}, ..., V_t]
    deltas = [values[j + 1] - values[j] for j in range(K)]
    return np.array(values, dtype=np.float32), np.array(deltas, dtype=np.float32)


def _compute_tvr_target(
    values_hist: np.ndarray,
    gamma: float,
    epsilon: float,
) -> float:
    """TVR_t = (1 - V_t) * Σ_{i=0}^{K-1} γ^i * (ε - δ_i)"""
    K = len(values_hist) - 1
    V_t = float(values_hist[-1])
    total = 0.0
    for i in range(K):
        delta_i = float(values_hist[K - i]) - float(values_hist[K - i - 1])
        total += (gamma ** i) * (epsilon - delta_i)
    return float((1.0 - V_t) * total)


def _resolve_next_idx_for_v(
    transitions: list,
    ep_bounds: np.ndarray,
    flat_idx: int,
) -> tuple[int, bool, bool]:
    """Resolve V(t+1) source index. Returns (next_idx, is_terminal, used_fallback)."""
    fi = int(flat_idx)
    nxt, terminal = _next_step_indices(ep_bounds, fi)
    used_fallback = False
    md_cur = transitions[fi].get("transition_metadata", {}) if isinstance(transitions[fi], dict) else {}
    if isinstance(md_cur, dict) and md_cur.get("episode_id") is not None:
        ep_id_cur = str(md_cur.get("episode_id"))
        meta_term = bool(md_cur.get("episode_terminal") or md_cur.get("terminated"))
        if terminal or meta_term:
            return fi, True, False
        md_nxt = transitions[nxt].get("transition_metadata", {}) if isinstance(transitions[nxt], dict) else {}
        ep_id_nxt = str(md_nxt.get("episode_id", "")) if isinstance(md_nxt, dict) else ""
        if ep_id_nxt and ep_id_nxt != ep_id_cur:
            return fi, True, True
    return nxt, terminal, used_fallback


def _build_vq1_chat_text(sample: dict, processor) -> str:
    """system + user (images + instruction + current_action + <VQ_QUERY>)."""
    user_text = (
        f"{sample['instr']}\n\n"
        f"{sample['current_action_text']}\n"
        f"{VQ_TOKEN}"
    )
    msgs = [
        {"role": "system", "content": SYSTEM_MSG},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in sample["images"]],
                {"type": "text", "text": user_text},
            ],
        },
    ]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def _build_vq1_inputs(sample: dict, processor, device):
    text = _build_vq1_chat_text(sample, processor)
    return processor(
        text=[text],
        images=sample["images"] or None,
        return_tensors="pt",
        padding=True,
    ).to(device)


def _locate_text_decoder_norm(model) -> tuple:
    """Locate the LLM text-decoder's final norm + decoder-block ModuleList."""
    candidates: list[tuple[int, str, nn.ModuleList]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) == 0:
            continue
        first = module[0]
        if not any(hasattr(first, a) for a in ("self_attn", "attn", "attention")):
            continue
        candidates.append((len(module), name, module))
    if not candidates:
        return None, None, None

    def _score(c):
        n, name, _ = c
        lower = name.lower()
        prefer = 1 if ("language" in lower or "text" in lower) else 0
        return (prefer, n)

    candidates.sort(key=_score, reverse=True)
    n, layers_path, layers = candidates[0]

    parent_parts = layers_path.split(".")[:-1]
    parent = model
    for part in parent_parts:
        parent = getattr(parent, part)
    norm = getattr(parent, "norm", None)
    parent_path = ".".join(parent_parts) if parent_parts else "<root>"
    return norm, layers, parent_path


class _LastHiddenCapture:
    """Context manager that hooks a target module to cache its forward output."""

    def __init__(self, target_module, label: str = ""):
        self._target = target_module
        self._handle = None
        self._cache: torch.Tensor | None = None
        self.label = str(label)

    def _hook(self, _module, _inputs, output):
        t = output[0] if isinstance(output, (tuple, list)) else output
        self._cache = t

    def __enter__(self):
        self._cache = None
        self._handle = self._target.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._cache = None
        return False

    @property
    def hidden(self) -> torch.Tensor:
        if self._cache is None:
            raise RuntimeError(
                f"[hook:{self.label}] no hidden captured — was the target module "
                "skipped during forward?"
            )
        return self._cache


def _h_t_from_hidden(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    vq_token_id: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same gather logic as vq1_heads.extract_h_t but operates on a single"""
    B, T, _ = hidden.shape
    matches = input_ids == int(vq_token_id)
    has = matches.any(dim=1)
    if not bool(has.all()):
        missing = torch.where(~has)[0].detach().cpu().tolist()
        raise RuntimeError(f"<VQ_QUERY> is missing from batch rows {missing}")
    pos = (matches.float().cumsum(dim=1) * matches.float()).argmax(dim=1)
    idx = torch.arange(B, device=hidden.device)
    return hidden[idx, pos].contiguous()


def _build_vq1_inputs_batched(samples: list, processor, device):
    """Batched version of _build_vq1_inputs."""
    texts = [_build_vq1_chat_text(s, processor) for s in samples]
    images_per = [list(s["images"] or []) for s in samples]
    use_images = any(len(imgs) > 0 for imgs in images_per)
    return processor(
        text=texts,
        images=images_per if use_images else None,
        return_tensors="pt",
        padding=True,
    ).to(device)


def _gather_future_targets_batched(
    samples: list,
    dataset,
    ep_bounds: np.ndarray,
    future_target,
    args,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return (z_tgt_valid, valid_mask) for the batch."""
    if not future_target.available:
        return None, None
    mode = str(args.future_latent_mode)
    src = str(args.future_target_source)

    if src == "precomputed_jepa" and future_target.precomputed is not None:
        rows: list[torch.Tensor] = []
        valid: list[bool] = []
        for s in samples:
            fi = int(s["flat_idx"])
            stem = _episode_stem_for_idx(dataset, fi)
            md = dataset.transitions[fi].get("transition_metadata", {}) or {}
            step_in_ep = int(md.get(
                "step_in_episode", fi - int(ep_bounds[fi, 0]),
            ))
            z = future_target.precomputed.get_next(
                stem, step_in_ep, mode=mode,
                device=future_target.device, dtype=torch.float32,
            )
            if z is None:
                valid.append(False)
            else:
                valid.append(True)
                rows.append(z.squeeze(0))                # drop leading 1
        if not rows:
            return None, torch.tensor(valid, dtype=torch.bool)
        z_stack = torch.stack(rows, dim=0)
        return z_stack, torch.tensor(valid, dtype=torch.bool)

    if future_target.online is not None:
        views_per = [
            _next_obs_uint8_views(
                dataset.transitions,
                int(s["flat_idx"]),
                use_external_wrist_keys=bool(getattr(args, "use_external_wrist_keys", False)),
            )
            for s in samples
        ]
        valid_mask = torch.tensor(
            [len(v) > 0 for v in views_per], dtype=torch.bool,
        )
        if not bool(valid_mask.any()):
            return None, valid_mask
        z = future_target.online.encode_views(views_per, mode=mode)
        if z is None:
            return None, valid_mask
        z = z.float()
        if z.shape[0] != len(samples):
            return None, valid_mask
        return z[valid_mask], valid_mask
    return None, None


def _align_future_target_dense(
    z_pred: torch.Tensor, z_tgt: torch.Tensor,
) -> torch.Tensor:
    """Match z_tgt's token count to z_pred's along dim=1 (truncate or pad)."""
    n_pred = int(z_pred.shape[1])
    n_tgt = int(z_tgt.shape[1])
    if n_tgt == n_pred:
        return z_tgt
    if n_tgt > n_pred:
        return z_tgt[:, :n_pred]
    pad = torch.zeros(
        z_tgt.size(0), n_pred - n_tgt, z_tgt.size(-1),
        device=z_tgt.device, dtype=z_tgt.dtype,
    )
    return torch.cat([z_tgt, pad], dim=1)


def _gather_value_history_batched(
    samples: list,
    transitions: list,
    ep_bounds: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (values_hist [B, K+1], deltas_hist [B, K]) for the batch."""
    vals_list, delts_list = [], []
    for s in samples:
        v, d = _value_history_window(
            int(s["flat_idx"]), transitions, ep_bounds, int(K),
        )
        vals_list.append(v)
        delts_list.append(d)
    return (
        np.stack(vals_list, axis=0).astype(np.float32, copy=False),
        np.stack(delts_list, axis=0).astype(np.float32, copy=False),
    )


def _gather_proxy_v_tp1_batched(
    samples: list,
    transitions: list,
    ep_bounds: np.ndarray,
    latent_proxy_vf=None,       # Optional[_LatentProxyVF]
    dataset=None,               # HILVLMDataset, needed for episode_stem lookup
) -> tuple[np.ndarray, np.ndarray]:
    """Return (v_tp1 [B], terminal_mask [B]). Terminal rows are excluded from"""
    vs = np.zeros(len(samples), dtype=np.float32)
    terms = np.zeros(len(samples), dtype=bool)

    if latent_proxy_vf is not None and dataset is not None:
        z_list: list[np.ndarray] = []
        task_names: list[str] = []
        pending: list[int] = []   # indices into vs/terms that need latent scoring

        for i, s in enumerate(samples):
            flat_idx = int(s["flat_idx"])
            nxt, terminal, _ = _resolve_next_idx_for_v(transitions, ep_bounds, flat_idx)
            terms[i] = terminal
            if terminal:
                vs[i] = float(transitions[flat_idx].get("vf_value_pred", 0.0) or 0.0)
                continue
            stem, step = dataset.episode_stem_and_local_index(nxt)
            task_name = str(transitions[nxt].get("task_name", "") or "")
            z = latent_proxy_vf._latents.get_next(
                stem, step, mode="pooled",
                device="cpu", dtype=torch.float32,
            )
            if z is None:
                terms[i] = True
                vs[i] = float(transitions[flat_idx].get("vf_value_pred", 0.0) or 0.0)
            else:
                z_list.append(z.reshape(-1).numpy())
                task_names.append(task_name)
                pending.append(i)

        if pending:
            z_batch = torch.from_numpy(np.stack(z_list, axis=0))  # [P, D]
            scores = latent_proxy_vf.score_batch(z_batch, task_names)  # [P]
            for j, idx in enumerate(pending):
                vs[idx] = float(scores[j].item())
    else:
        for i, s in enumerate(samples):
            v, t, _, _ = _proxy_v_tp1(transitions, ep_bounds, int(s["flat_idx"]))
            vs[i] = v
            terms[i] = t

    return vs, terms


def _gather_proxy_v_t_batched(
    samples: list, transitions: list,
) -> np.ndarray:
    """Return V_t [B] from each sample's vf_value_pred."""
    out = []
    for s in samples:
        fi = int(s["flat_idx"])
        try:
            v = float(transitions[fi].get("vf_value_pred", 0.0) or 0.0)
        except Exception:
            v = 0.0
        out.append(v)
    return np.asarray(out, dtype=np.float32)


def _compute_tvr_targets_batched(
    values_hist: np.ndarray, gamma: float, epsilon: float,
) -> np.ndarray:
    """Vectorized TVR target over a batch of history windows."""
    if values_hist.ndim != 2 or values_hist.shape[1] < 2:
        return np.zeros((values_hist.shape[0],), dtype=np.float32)
    K = int(values_hist.shape[1] - 1)
    V_t = values_hist[:, -1]                                       # [B]
    flipped = values_hist[:, ::-1]                                 # V_t, V_{t-1}, ...
    deltas = flipped[:, :-1] - flipped[:, 1:]                      # [B, K]
    gammas = (float(gamma) ** np.arange(K, dtype=np.float32))      # [K]
    inner = (gammas[None, :] * (float(epsilon) - deltas)).sum(axis=1)
    return ((1.0 - V_t) * inner).astype(np.float32, copy=False)


_KNOWN_PROXY_VF_SOURCES: dict[str, tuple[str, bool, str]] = {}


def _resolve_proxy_vf_checkpoint_file(vf_dir: str) -> tuple[str, str]:
    """Return (resolved_checkpoint_path, how_resolved). Mirrors the loader logic in"""
    if not vf_dir:
        return "", "no_vf_dir"
    candidates = [
        ("stitched_value_model_best.pt", "stitched_best"),
        ("latent_gemma_value_model_best.pt", "latent_best"),
        ("value_model_best.pt", "value_best"),
        ("model_best.pt", "model_best"),
    ]
    for fname, tag in candidates:
        cand = os.path.join(vf_dir, fname)
        if os.path.isfile(cand):
            return cand, tag
    proj = os.path.join(vf_dir, "projector_best.pt")
    vh = os.path.join(vf_dir, "value_head_best.pt")
    if os.path.isfile(proj) and os.path.isfile(vh):
        return f"{proj} + {vh}", "two_file_best"
    return "", "no_best_file_found"


class _LatentProxyVF:
    """Online V(t+1) scoring from JEPA latents via RobotLatentValueModel."""

    def __init__(
        self,
        checkpoint_path: str,
        vf_cfg: dict,
        device,
        latent_dir: str,
        latent_manifest: str | None = None,
    ):
        import sys as _sys
        import os as _os

        jepa_dir = _os.path.join(_os.path.dirname(__file__), "JEPA")
        if jepa_dir not in _sys.path:
            _sys.path.insert(0, jepa_dir)
        import train_vf_latent_gemma as _tvl  # type: ignore

        language_model = str(vf_cfg.get("language_model") or "google/gemma-3-270m-it")
        prompt = str(vf_cfg.get("prompt") or "Task: {task_name}")
        latent_dim = int(vf_cfg.get("latent_dim", 1024))
        hidden_dim = int(vf_cfg.get("proj_out", 640))

        self._prompt_template = prompt
        self.device = device

        self._model = _tvl.RobotLatentValueModel(
            language_model,
            prompt,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        ).to(device)

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self._model.load_state_dict(state, strict=False)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        from vq1_heads import _PrecomputedJEPALatent  # type: ignore
        self._latents = _PrecomputedJEPALatent(latent_dir, latent_manifest, mode="pooled")

        log(f"[latent_proxy_vf] loaded  ckpt={checkpoint_path}")
        log(f"[latent_proxy_vf] latent_dir={latent_dir}  manifest_entries={len(self._latents._mapping)}")

    def _make_prompt(self, task_name: str) -> str:
        try:
            return self._prompt_template.format(task_name=task_name)
        except Exception:
            return self._prompt_template

    @torch.no_grad()
    def score_batch(
        self,
        z_batch: torch.Tensor,
        task_names: list[str] | None = None,
    ) -> torch.Tensor:
        """Score a batch of pooled JEPA latents. Returns min(V1,V2) [B] float32."""
        z = z_batch.to(device=self.device, dtype=torch.float32)
        prompts = [self._make_prompt(n) for n in (task_names or [""] * z.shape[0])]
        v = self._model.predict_value_from_latent_feats(z, prompt_texts=prompts)
        return v.float().cpu()

    def v_tp1(
        self,
        episode_stem: str,
        step_in_ep: int,
        task_name: str = "",
    ) -> tuple[float, bool]:
        """Return (V(t+1), is_terminal). Terminal = latent not found or last step."""
        z = self._latents.get_next(episode_stem, step_in_ep, mode="pooled",
                                   device=self.device, dtype=torch.float32)
        if z is None:
            return 0.0, True
        z = z.reshape(1, -1)
        v = self.score_batch(z, task_names=[task_name])
        return float(v[0].item()), False


def _read_proxy_vf_provenance(buffer_dir: str, args=None) -> dict:
    """Inspect the buffer's vf_value_pred_manifest.json + the VF run's config.json"""
    import json as _json

    prov: dict = {
        "proxy_vf_source_name": "unknown",
        "proxy_vf_checkpoint_path": "unknown",
        "proxy_vf_training_script": "unknown",
        "proxy_vf_score_buffer_dir": "unknown",
        "proxy_vf_score_manifest_path": None,
        "proxy_vf_is_latent_based": None,
        "proxy_vf_training_mode": None,
        "proxy_vf_run_dir": None,
        "proxy_vf_checkpoint_resolution": "unknown",
        "proxy_vf_best_val_loss": None,
        "proxy_vf_best_val_epoch": None,
        "proxy_vf_notes": [],
    }
    if args is not None:
        for k in (
            "proxy_vf_source_name", "proxy_vf_checkpoint_path",
            "proxy_vf_training_script", "proxy_vf_score_buffer_dir",
            "proxy_vf_score_manifest_path",
        ):
            v = getattr(args, k, None)
            if v:
                prov[k] = str(v)
        if getattr(args, "proxy_vf_is_latent_based", None) is not None:
            prov["proxy_vf_is_latent_based"] = bool(args.proxy_vf_is_latent_based)
    manifest_path = os.path.join(buffer_dir, "vf_value_pred_manifest.json")
    if os.path.isfile(manifest_path):
        prov["proxy_vf_score_manifest_path"] = manifest_path
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                man = _json.load(f)
        except Exception as exc:
            prov["proxy_vf_notes"].append(f"manifest parse error: {exc}")
            man = {}
        vf_dir = str(man.get("vf_dir", "") or "")
        if vf_dir and prov["proxy_vf_run_dir"] in (None, "unknown"):
            prov["proxy_vf_run_dir"] = vf_dir
        if man.get("buffer_dir") and prov["proxy_vf_score_buffer_dir"] == "unknown":
            prov["proxy_vf_score_buffer_dir"] = str(man["buffer_dir"])
        if vf_dir in _KNOWN_PROXY_VF_SOURCES:
            name, is_latent, script = _KNOWN_PROXY_VF_SOURCES[vf_dir]
            if prov["proxy_vf_source_name"] == "unknown":
                prov["proxy_vf_source_name"] = name
            if prov["proxy_vf_is_latent_based"] is None:
                prov["proxy_vf_is_latent_based"] = bool(is_latent)
            if prov["proxy_vf_training_script"] == "unknown":
                prov["proxy_vf_training_script"] = script
        ckpt_path, how = _resolve_proxy_vf_checkpoint_file(vf_dir)
        if ckpt_path and prov["proxy_vf_checkpoint_path"] == "unknown":
            prov["proxy_vf_checkpoint_path"] = ckpt_path
            prov["proxy_vf_checkpoint_resolution"] = how
        vf_cfg_path = os.path.join(vf_dir, "config.json") if vf_dir else ""
        if vf_cfg_path and os.path.isfile(vf_cfg_path):
            try:
                with open(vf_cfg_path, "r", encoding="utf-8") as f:
                    vf_cfg = _json.load(f)
                if prov["proxy_vf_training_mode"] is None:
                    prov["proxy_vf_training_mode"] = vf_cfg.get("training_mode")
                tm = str(vf_cfg.get("training_mode", "")).lower()
                if prov["proxy_vf_is_latent_based"] is None:
                    prov["proxy_vf_is_latent_based"] = ("latent" in tm)
                if prov["proxy_vf_best_val_loss"] is None:
                    prov["proxy_vf_best_val_loss"] = vf_cfg.get("best_val_loss")
                if prov["proxy_vf_best_val_epoch"] is None:
                    prov["proxy_vf_best_val_epoch"] = vf_cfg.get("best_epoch")
                if prov["proxy_vf_source_name"] == "unknown":
                    prov["proxy_vf_source_name"] = vf_cfg.get("training_mode", "unknown")
            except Exception as exc:
                prov["proxy_vf_notes"].append(f"vf_cfg parse error: {exc}")
        else:
            prov["proxy_vf_notes"].append(
                f"no config.json under VF run dir: {vf_dir!r}"
            )
    else:
        prov["proxy_vf_notes"].append(
            f"no vf_value_pred_manifest.json under {buffer_dir!r}; provenance unverified"
        )

    if prov["proxy_vf_is_latent_based"] is None:
        prov["proxy_vf_is_latent_based"] = "unknown"
    return prov


def _log_proxy_vf_provenance(prov: dict) -> None:
    log("[proxy_vf] ── proxy V(t+1) provenance ──")
    log(f"[proxy_vf] source_name          = {prov.get('proxy_vf_source_name')}")
    log(f"[proxy_vf] training_mode        = {prov.get('proxy_vf_training_mode')}")
    log(f"[proxy_vf] is_latent_based      = {prov.get('proxy_vf_is_latent_based')}")
    log(f"[proxy_vf] training_script      = {prov.get('proxy_vf_training_script')}")
    log(f"[proxy_vf] run_dir              = {prov.get('proxy_vf_run_dir')}")
    log(f"[proxy_vf] checkpoint_path      = {prov.get('proxy_vf_checkpoint_path')}")
    log(f"[proxy_vf] checkpoint_resolved  = {prov.get('proxy_vf_checkpoint_resolution')}")
    log(f"[proxy_vf] score_buffer_dir     = {prov.get('proxy_vf_score_buffer_dir')}")
    log(f"[proxy_vf] manifest_path        = {prov.get('proxy_vf_score_manifest_path')}")
    log(f"[proxy_vf] best_val_loss        = {prov.get('proxy_vf_best_val_loss')}")
    log(f"[proxy_vf] best_val_epoch       = {prov.get('proxy_vf_best_val_epoch')}")
    notes = list(prov.get("proxy_vf_notes") or [])
    if str(prov.get("proxy_vf_source_name")) == "unknown" or \
       str(prov.get("proxy_vf_checkpoint_path")) == "unknown":
        notes.append("WARNING: proxy VF provenance is partially unverified — see fields above")
    for n in notes:
        log(f"[proxy_vf] note: {n}")


def _intervention_pos_weight(transitions: list, mask_fn) -> float:
    pos = sum(1 for i in range(len(transitions)) if mask_fn(i))
    neg = max(1, len(transitions) - pos)
    if pos <= 0:
        return 1.0
    return float(neg) / float(pos)


def _split_param_groups(model, head_modules, lr_backbone: float, lr_heads: float):
    backbone_params = [p for p in model.parameters() if p.requires_grad]
    head_params: list = []
    for m in head_modules:
        if m is None:
            continue
        head_params.extend(p for p in m.parameters() if p.requires_grad)
    return [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": 0.01},
        {"params": head_params,     "lr": lr_heads,    "weight_decay": 0.0 },
    ]


def _stage_lambdas(epoch: int, args) -> tuple[float, float, float]:
    """Optional 3-stage schedule for (lambda_future, lambda_q, lambda_intervention)."""
    if not bool(getattr(args, "stage_schedule", False)):
        return (
            float(args.lambda_future),
            float(args.lambda_q),
            float(args.lambda_intervention),
        )
    s1 = int(getattr(args, "stage1_epochs", 1))
    s2 = int(getattr(args, "stage2_epochs", 3))
    if epoch <= s1:
        return (1.0, 1.0, 0.1)
    if epoch <= s1 + s2:
        return (0.5, 1.0, 1.0)
    return (0.1, 0.5, 1.0)


def train_vq1(args):
    """Unified VQ1 trainer (shared h_t at <VQ_QUERY>, V-JEPA future loss,"""
    log(f"[vq1] variant={args.variant}  future_mode={args.future_latent_mode}  "
        f"vjepa_path={args.vjepa_path}")
    spec = get_variant(args.variant)
    log(f"[vq1] spec: future_head={spec.use_future_head} L_future={spec.use_future_loss} "
        f"Q_head={spec.use_q_head} L_Q={spec.use_q_loss} "
        f"int_head={spec.use_int_head} L_int={spec.use_int_loss} "
        f"q_input={spec.q_input} int_input_z={spec.int_input_z} int_use_q_cond={spec.int_use_q_cond}")
    itype = str(getattr(args, "intervention_head_type", "binary")).lower()
    if itype not in ("binary", "current_value", "tvr_history"):
        raise SystemExit(f"[vq1] unknown --intervention_head_type={itype}; choices: binary, current_value, tvr_history")
    log(f"[vq1] intervention_head_type={itype}")

    use_wandb = getattr(args, "use_wandb", False)
    if use_wandb:
        run_name = args.wandb_run_name or f"vq1_{args.variant}_{int(time.time())}"
        init_kwargs = {"project": args.wandb_project, "name": run_name, "config": vars(args)}
        if getattr(args, "wandb_entity", None):
            init_kwargs["entity"] = args.wandb_entity
        wandb.init(**init_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_kw = {
        "stage1_include_reward_label": False,
        "train_ratio": float(getattr(args, "train_ratio", 0.8)),
        "val_ratio": float(getattr(args, "val_ratio", 0.1)),
        "test_ratio": float(getattr(args, "test_ratio", 0.1)),
        "seed": int(getattr(args, "split_seed", 42)),
        "round2_retrieval_source": str(getattr(args, "round2_retrieval_source", "memory_bank")),
        "round2_memory_bank_path": str(getattr(args, "memory_bank_path", "")),
        "round2_memory_bank_router_path": str(getattr(args, "memory_bank_router_json", "")),
        "round2_task_route_mode": str(getattr(args, "round2_task_route_mode", "strict")),
        "round2_memory_top_k": int(getattr(args, "memory_top_k", 5)),
        "round2_memory_goal_aggregate": str(getattr(args, "memory_goal_aggregate", "top1")),
        "intervention_label_mode": str(getattr(args, "intervention_label_mode", "recover_mask")),
        "intervention_label_source": str(getattr(args, "intervention_label_source", "auto")),
        "intervention_label_key": str(getattr(args, "intervention_label_key", "vf_mining_intervention_label")),
        "intervention_expand_steps": int(getattr(args, "intervention_expand_steps", 0)),
        "task_aware_instr": bool(getattr(args, "task_aware_instr", True)),
        "round2_task_aware_filter": bool(getattr(args, "round2_task_aware_filter", True)),
        "enable_round2": False,  # VQ1 path: do not pull VQ2 data; keeps loader fast and clean.
        "round2_target_space": str(getattr(args, "round2_target_space", "state11_pose")),
        "state_dim": int(getattr(args, "state_dim", 11)),
        "state_clip_k": float(getattr(args, "state_clip_k", 4.0)),
        "use_external_wrist_keys": bool(getattr(args, "use_external_wrist_keys", False)),
    }
    ds_kw["recover_mining_config"] = RecoverTargetMiningConfig(
        value_key="vf_value_pred",
        value_normalization="minmax_episode",
        require_intervention_signal=False,
        decline_window=int(getattr(args, "decline_window", 8)),
        min_down_ratio=float(getattr(args, "min_down_ratio", 0.7)),
        min_drop=float(getattr(args, "min_drop", 0.05)),
        intervention_margin=int(getattr(args, "intervention_margin", 2)),
    )
    if getattr(args, "instr", None):
        ds_kw["instr"] = str(args.instr)
    req_vf = str(getattr(args, "required_vf_dir", "") or "").strip()
    ds_kw["required_vf_dir"] = req_vf if req_vf else None
    train_ds = HILVLMDataset(args.buffer_dir, split="train", **ds_kw)
    val_ds_kw = dict(ds_kw)
    val_ds_kw["state_dim"] = int(getattr(train_ds, "_state_dim", 11))
    val_ds_kw["state_clip_k"] = float(getattr(train_ds, "_state_clip_k", 4.0))
    val_ds   = HILVLMDataset(
        args.buffer_dir,
        split="val",
        action_min=train_ds.a_min,
        action_max=train_ds.a_max,
        state_mean=train_ds.s_mean,
        state_std=train_ds.s_std,
        **val_ds_kw,
    )
    log(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    ep_bounds_train = _build_episode_bounds_map(train_ds.episode_spans, len(train_ds.transitions))
    ep_bounds_val = _build_episode_bounds_map(val_ds.episode_spans, len(val_ds.transitions))

    w = np.ones(len(train_ds.transitions), dtype=float)
    boost = float(getattr(args, "stage1_yes_boost", 1.0))
    if boost > 1.0:
        m = np.array([bool(train_ds.intervention_label_at(i)) for i in range(len(train_ds))], dtype=bool)
        boosted, stats = apply_binary_mask_boost(w, m, boost)
        if boosted is not None:
            w = boosted
            log(f"[weight] yes-boost x{boost:.2f}  effective_yes_share={stats['effective_share']:.4f}")
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(w),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    model_cls, model_cls_name = _resolve_qwen_vl_model_cls(args.model_path)
    log(f"[model] model_cls={model_cls_name}")
    if getattr(args, "load_in_4bit", False):
        if device.type != "cuda":
            raise SystemExit("--load_in_4bit needs CUDA")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = model_cls.from_pretrained(
            args.model_path, quantization_config=bnb_config,
            device_map={"": torch.cuda.current_device()}, trust_remote_code=False,
        )
    else:
        model = model_cls.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=False,
        ).to(device)

    if bool(args.use_lora):
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        if getattr(args, "load_in_4bit", False):
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool(getattr(args, "gradient_checkpointing", False)),
            )
        target_modules = [t.strip() for t in str(args.lora_target_modules).split(",") if t.strip()]
        if bool(getattr(args, "lora_include_ffn", False)):
            for t in ("gate_proj", "up_proj", "down_proj"):
                if t not in target_modules:
                    target_modules.append(t)
        lc = LoraConfig(
            r=int(args.lora_r), lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout), bias="none",
            target_modules=target_modules, task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lc)
        model.enable_input_require_grads()
        model.print_trainable_parameters()
        log(f"[lora] target_modules={target_modules} r={args.lora_r} alpha={args.lora_alpha}")
    if bool(getattr(args, "gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    vq_token_id = add_vq_query_token(processor.tokenizer, model)
    log(f"[vq1] <VQ_QUERY> token_id={vq_token_id} vocab={len(processor.tokenizer)}")

    hidden_capture_mode = str(getattr(args, "hidden_state_capture", "hook")).lower()
    hook_target_module = None
    hook_target_path = "<unused>"
    if hidden_capture_mode == "hook":
        norm_mod, layers_mod, parent_path = _locate_text_decoder_norm(model)
        if norm_mod is not None:
            hook_target_module = norm_mod
            hook_target_path = f"{parent_path}.norm"
        elif layers_mod is not None:
            hook_target_module = layers_mod[-1]
            hook_target_path = f"{parent_path}.layers[-1]"
            log(
                "[vq1] WARN: text-decoder final norm not found; hooking last "
                "decoder layer instead. Under gradient_checkpointing the cached "
                "tensor may lack grad_fn — flip --hidden_state_capture=output_hidden_states "
                "if backward fails."
            )
        else:
            log(
                "[vq1] WARN: could not locate text decoder module; falling back "
                "to output_hidden_states=True"
            )
            hidden_capture_mode = "output_hidden_states"
    log(
        f"[vq1] hidden_state_capture={hidden_capture_mode}  "
        f"hook_target={hook_target_path}"
    )

    model.train()
    device = next(model.parameters()).device
    cfg_top = model.config
    cfg_text = getattr(cfg_top, "text_config", None)
    hidden_size = int(
        getattr(cfg_top, "hidden_size", None)
        or (getattr(cfg_text, "hidden_size", None) if cfg_text is not None else None)
        or 0
    )
    if hidden_size <= 0:
        raise SystemExit("[vq1] could not infer hidden_size from model.config")
    log(f"[vq1] hidden_size={hidden_size}")

    if spec.use_future_loss:
        future_target = FutureTargetEncoder(
            source=str(args.future_target_source),
            device=device, dtype=torch.bfloat16,
            vjepa_path=(str(args.vjepa_path) or None),
            latent_dir=(str(args.jepa_latent_dir) or None),
            latent_manifest=(str(args.jepa_latent_manifest) or None),
            latent_mode=str(args.future_latent_mode),
        )
    else:
        future_target = FutureTargetEncoder(source="online_vjepa2", device=device, vjepa_path=None)
    if spec.use_future_loss and not future_target.available:
        msg = (
            f"V-JEPA target unavailable: source={args.future_target_source} "
            f"vjepa_path={args.vjepa_path!r} jepa_latent_dir={args.jepa_latent_dir!r}"
        )
        if bool(getattr(args, "require_future_loss", False)) or args.variant == "full":
            raise SystemExit(
                "[vq1] L_future is REQUIRED for this variant (use --variant=no_future / "
                "no_future_loss to disable). " + msg
            )
        log(f"[vq1] WARN: {msg} — L_future will be skipped this run.")

    target_dim = int(args.vjepa_feature_dim)
    n_tokens = int(args.vjepa_n_tokens)
    if spec.use_future_head and future_target.available:
        probe_idx = 0
        if str(args.future_target_source) == "precomputed_jepa":
            stem = _episode_stem_for_idx(train_ds, probe_idx)
            md = train_ds.transitions[probe_idx].get("transition_metadata", {}) if isinstance(train_ds.transitions[probe_idx], dict) else {}
            step_in_ep = int((md or {}).get("step_in_episode", probe_idx - int(ep_bounds_train[probe_idx, 0])))
            z_probe = future_target.compute_target(
                mode=args.future_latent_mode,
                episode_stem=stem,
                step_in_ep=step_in_ep,
            )
        else:
            probe_views = [_next_obs_uint8_views(
                train_ds.transitions,
                probe_idx,
                use_external_wrist_keys=bool(getattr(args, "use_external_wrist_keys", False)),
            )]
            z_probe = future_target.compute_target(
                mode=args.future_latent_mode,
                views_per_sample=probe_views,
            )
        if z_probe is not None:
            if args.future_latent_mode == "pooled":
                target_dim = int(z_probe.shape[-1])
                n_tokens = 0
                log(f"[vq1] future target shape (pooled) = [B, D={target_dim}]  source={args.future_target_source}")
            else:
                n_tokens = int(z_probe.shape[1])
                target_dim = int(z_probe.shape[-1])
                log(f"[vq1] future target shape (dense)  = [B, N={n_tokens}, D={target_dim}]")

    future_head = None
    twin_q_head = None
    intervention_head = None
    value_history_encoder = None
    v_curr_head = None
    risk_head = None
    if spec.use_future_head:
        future_head = FutureHead(
            hidden_size=hidden_size, latent_dim=target_dim,
            mode=args.future_latent_mode, n_tokens=n_tokens,
        ).to(device=device, dtype=torch.float32)
    if spec.use_q_head:
        if spec.q_input == "z" and args.future_latent_mode == "dense":
            twin_q_head = TwinQHead(in_dim=target_dim, accepts_dense=True).to(device=device, dtype=torch.float32)
        elif spec.q_input == "z":
            twin_q_head = TwinQHead(in_dim=target_dim, accepts_dense=False).to(device=device, dtype=torch.float32)
        else:
            twin_q_head = TwinQHead(in_dim=hidden_size, accepts_dense=False).to(device=device, dtype=torch.float32)
    if spec.use_int_head:
        if spec.int_input_z == "z" and args.future_latent_mode == "dense":
            _int_in_dim = target_dim
            _int_accepts_dense = True
        elif spec.int_input_z == "z":
            _int_in_dim = target_dim
            _int_accepts_dense = False
        else:
            _int_in_dim = hidden_size
            _int_accepts_dense = False
        if itype == "binary":
            intervention_head = InterventionHead(
                in_dim=_int_in_dim, use_q_cond=spec.int_use_q_cond, accepts_dense=_int_accepts_dense,
            ).to(device=device, dtype=torch.float32)
        elif itype == "tvr_history":
            _K = int(getattr(args, "value_history_window", 8))
            _embed = int(getattr(args, "value_history_embed_dim", 32))
            value_history_encoder = ValueHistoryEncoder(K=_K, embed_dim=_embed).to(device=device, dtype=torch.float32)
            v_curr_head = VCurrHead(hidden_size=hidden_size).to(device=device, dtype=torch.float32)
            risk_head = RiskHead(z_dim=_int_in_dim, context_dim=_embed, accepts_dense=_int_accepts_dense).to(device=device, dtype=torch.float32)
            log(f"[vq1] tvr_history heads: K={_K} embed_dim={_embed} z_dim={_int_in_dim}")
        elif itype == "current_value":
            v_curr_head = VCurrHead(hidden_size=hidden_size).to(device=device, dtype=torch.float32)
            risk_head = RiskHead(z_dim=_int_in_dim, context_dim=1, accepts_dense=_int_accepts_dense).to(device=device, dtype=torch.float32)
            log(f"[vq1] current_value heads: z_dim={_int_in_dim} context_dim=1")

    resume_heads_dir = str(getattr(args, "resume_vq1_heads_path", "") or "").strip()
    if resume_heads_dir:
        heads_pt = os.path.join(resume_heads_dir, "vq1_heads.pt")
        if not os.path.isfile(heads_pt):
            raise SystemExit(f"[resume] vq1_heads.pt not found: {heads_pt}")
        saved = torch.load(heads_pt, map_location=device, weights_only=True)
        for head, key in [
            (future_head,           "future_head"),
            (twin_q_head,           "twin_q_head"),
            (intervention_head,     "intervention_head"),
            (value_history_encoder, "value_history_encoder"),
            (v_curr_head,           "v_curr_head"),
            (risk_head,             "risk_head"),
        ]:
            if head is not None and saved.get(key) is not None:
                head.load_state_dict(saved[key], strict=True)
                log(f"[resume] loaded {key} from {heads_pt}")
            elif head is not None:
                log(f"[resume] WARN: {key} not found in checkpoint, keeping random init")

    head_modules = [future_head, twin_q_head, intervention_head,
                    value_history_encoder, v_curr_head, risk_head]
    param_groups = _split_param_groups(model, head_modules,
                                       lr_backbone=float(args.lr),
                                       lr_heads=float(args.head_lr))
    n_backbone = int(sum(p.numel() for p in param_groups[0]["params"]))
    n_heads = int(sum(p.numel() for p in param_groups[1]["params"]))
    log(f"[opt] trainable: backbone/LoRA={n_backbone} heads={n_heads} total={n_backbone+n_heads}")

    if bool(getattr(args, "adam8bit", False)):
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(param_groups, lr=float(args.lr), weight_decay=0.01)
        except Exception as e:
            log(f"[opt] AdamW8bit unavailable ({e}); falling back to AdamW")
            optimizer = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=0.01)

    proxy_vf_provenance = _read_proxy_vf_provenance(args.buffer_dir, args=args)
    _log_proxy_vf_provenance(proxy_vf_provenance)
    if bool(getattr(args, "strict_proxy_vf", False)) and (
        str(proxy_vf_provenance.get("proxy_vf_source_name")) == "unknown"
        or str(proxy_vf_provenance.get("proxy_vf_checkpoint_path")) == "unknown"
    ):
        raise SystemExit(
            "[proxy_vf] --strict_proxy_vf is set but provenance is unverified. "
            "Inspect the buffer's vf_value_pred_manifest.json or pass --proxy_vf_* "
            "CLI flags to declare provenance explicitly."
        )

    latent_proxy_vf: "_LatentProxyVF | None" = None
    if bool(getattr(args, "proxy_vf_is_latent_based", False)):
        ckpt = str(getattr(args, "proxy_vf_checkpoint_path", "") or "")
        lat_dir = str(getattr(args, "proxy_vf_latent_dir", "") or
                      getattr(args, "jepa_latent_dir", "") or "")
        lat_manifest = str(getattr(args, "proxy_vf_latent_manifest", "") or
                           getattr(args, "jepa_latent_manifest", "") or "")
        if not ckpt:
            log("[proxy_vf] WARNING: proxy_vf.is_latent_based=true but no checkpoint_path set; "
                "falling back to buffer vf_value_pred for Q targets.")
        elif not lat_dir:
            log("[proxy_vf] WARNING: proxy_vf.is_latent_based=true but no latent_dir; "
                "falling back to buffer vf_value_pred for Q targets.")
        else:
            vf_cfg: dict = {}
            vf_dir = os.path.dirname(ckpt)
            cfg_path = os.path.join(vf_dir, "config.json")
            if os.path.isfile(cfg_path):
                import json as _json
                with open(cfg_path, encoding="utf-8") as _f:
                    vf_cfg = _json.load(_f)
            if getattr(args, "proxy_vf_language_model", ""):
                vf_cfg["language_model"] = str(args.proxy_vf_language_model)
            if getattr(args, "proxy_vf_prompt", ""):
                vf_cfg["prompt"] = str(args.proxy_vf_prompt)
            latent_proxy_vf = _LatentProxyVF(
                checkpoint_path=ckpt,
                vf_cfg=vf_cfg,
                device=device,
                latent_dir=lat_dir,
                latent_manifest=lat_manifest or None,
            )
            log(f"[proxy_vf] latent VF active — Q targets will use z_{{t+1}} → VF(z)")

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        import json as _json
        meta = {
            "variant": args.variant,
            "future_latent_mode": args.future_latent_mode,
            "vjepa_feature_dim": target_dim,
            "vjepa_n_tokens": n_tokens,
            "vjepa_path": args.vjepa_path,
            "lora_target_modules": args.lora_target_modules,
            "lambda_future": args.lambda_future,
            "lambda_q": args.lambda_q,
            "lambda_intervention": args.lambda_intervention,
            "intervention_threshold": args.intervention_threshold,
            "intervention_loss_type": getattr(args, "intervention_loss_type", "weighted_bce"),
            "focal_alpha": getattr(args, "focal_alpha", 0.75),
            "focal_gamma": getattr(args, "focal_gamma", 2.0),
            "intervention_head_type": itype,
            "value_history_window": int(getattr(args, "value_history_window", 8)),
            "value_history_gamma": float(getattr(args, "value_history_gamma", 0.9)),
            "value_history_epsilon": float(getattr(args, "value_history_epsilon", 0.005)),
            "value_history_embed_dim": int(getattr(args, "value_history_embed_dim", 32)),
            "lambda_tvr": float(getattr(args, "lambda_tvr", 1.0)),
            "lambda_v_curr": float(getattr(args, "lambda_v_curr", 1.0)),
            "head_lr": args.head_lr,
            "backbone_lr": args.lr,
            "action_min": [float(x) for x in train_ds.a_min.tolist()],
            "action_max": [float(x) for x in train_ds.a_max.tolist()],
            "n_bins_act": int(train_ds.n_bins_act),
            "vq_token": VQ_TOKEN,
            "proxy_vf_provenance": proxy_vf_provenance,
        }
        with open(os.path.join(args.output_dir, "vq1_config.json"), "w") as f:
            _json.dump(meta, f, indent=2)
    except Exception as e:
        log(f"[vq1] WARN: failed to save vq1_config.json: {e}")

    loss_type = str(getattr(args, "intervention_loss_type", "weighted_bce")).lower()
    if loss_type == "focal":
        pos_w = 0.0
        log(f"[vq1] intervention_loss_type=focal alpha={args.focal_alpha} gamma={args.focal_gamma} (pos_weight disabled)")
    else:
        pos_w = float(getattr(args, "intervention_pos_weight", 0.0))
        if pos_w <= 0.0:
            pos_w = _intervention_pos_weight(
                train_ds.transitions,
                mask_fn=lambda i: bool(train_ds.intervention_label_at(i)),
            )
        log(f"[vq1] intervention_loss_type=weighted_bce auto pos_weight={pos_w:.3f}")
    pos_w_t = torch.tensor([max(pos_w, 1e-9)], dtype=torch.float32, device=device)

    use_cuda_bf16_amp = device.type == "cuda" and not getattr(args, "no_cuda_bf16_autocast", False)
    if use_cuda_bf16_amp:
        log("[train] bf16 autocast enabled on CUDA")

    log("[align] ── per-sample debug (first N) ──")
    n_dbg = max(0, int(getattr(args, "vq1_debug_samples", 0)))
    n_pos = n_term = n_fallback = n_valid = 0
    v_t_arr, v_tp1_arr = [], []
    for i in range(len(train_ds)):
        sample_i = train_ds[i]
        flat_i = int(sample_i["flat_idx"])
        v_t = float(train_ds.transitions[flat_i].get("vf_value_pred", 0.0))
        v_tp1, terminal, fallback, nxt = _proxy_v_tp1(train_ds.transitions, ep_bounds_train, flat_i)
        v_t_arr.append(v_t); v_tp1_arr.append(v_tp1)
        if terminal: n_term += 1
        if fallback: n_fallback += 1
        if not terminal: n_valid += 1
        if bool(sample_i.get("intervened", False)): n_pos += 1
        if i < n_dbg:
            md = train_ds.transitions[flat_i].get("transition_metadata", {}) or {}
            md_nxt = train_ds.transitions[nxt].get("transition_metadata", {}) or {}
            stem, lo = train_ds.episode_stem_and_local_index(flat_i)
            act = np.asarray(train_ds.transitions[flat_i].get("actions", []), dtype=np.float32).reshape(-1)
            log(
                f"[align] i={i} stem={stem} step={md.get('step_in_episode', lo)} "
                f"ep_id={str(md.get('episode_id',''))[:30]} "
                f"-> next_idx={nxt} step_next={md_nxt.get('step_in_episode','?')} "
                f"same_ep={(str(md.get('episode_id','')) == str(md_nxt.get('episode_id','')))} "
                f"terminal={terminal} fallback={fallback} "
                f"a_t[norm:{(act.min() if act.size else 0):.3f}..{(act.max() if act.size else 0):.3f}] "
                f"V(t)={v_t:.4f} V(t+1)={v_tp1:.4f} "
                f"intervened={bool(sample_i.get('intervened', False))} "
                f"L_Q={'use' if not terminal else 'bootstrap_self'}"
            )
    if v_t_arr and v_tp1_arr:
        v_t_a = np.asarray(v_t_arr, dtype=np.float64)
        v_tp1_a = np.asarray(v_tp1_arr, dtype=np.float64)
        if v_t_a.std() > 1e-9 and v_tp1_a.std() > 1e-9:
            corr = float(np.corrcoef(v_t_a, v_tp1_a)[0, 1])
        else:
            corr = float("nan")
        log(
            f"[align] aggregate: total={len(train_ds)} valid_next={n_valid} "
            f"terminal={n_term} fallback={n_fallback} "
            f"yes={n_pos} ({100*n_pos/max(1,len(train_ds)):.1f}%) "
            f"V(t): mean={v_t_a.mean():.3f} std={v_t_a.std():.3f} "
            f"min={v_t_a.min():.3f} max={v_t_a.max():.3f}  "
            f"V(t+1): mean={v_tp1_a.mean():.3f} std={v_tp1_a.std():.3f} "
            f"min={v_tp1_a.min():.3f} max={v_tp1_a.max():.3f}  corr(V(t),V(t+1))={corr:.3f}"
        )
        log(f"[align] valid_v_tp1_pct = {100.0 * n_valid / max(1, len(train_ds)):.2f}%")

    global_step = 0
    stop_training = False
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for m in head_modules:
            if m is not None:
                m.train()

        lambda_future, lambda_q, lambda_intervention = _stage_lambdas(epoch, args)
        log(f"[epoch {epoch}] lambdas: future={lambda_future} q={lambda_q} int={lambda_intervention}")

        agg = {
            "L_future": 0.0, "n_future": 0, "future_cos": 0.0,
            "L_q": 0.0, "n_q": 0, "Q1": 0.0, "Q2": 0.0, "Qmin": 0.0, "Vtgt": 0.0,
            "L_int": 0.0, "n_int": 0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "p_int": 0.0, "p_int_min": 1.0, "p_int_max": 0.0,
            "L_tvr": 0.0, "n_tvr": 0,
            "L_vcurr": 0.0, "n_vcurr": 0,
            "risk_preds_raw": [], "tvr_targets_raw": [],
            "L_total": 0.0, "n_total": 0,
            "probs": [], "labels": [],
        }

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            B = len(batch)
            if B == 0:
                continue

            inputs = _build_vq1_inputs_batched(batch, processor, device)
            with _cuda_bf16_autocast(device, use_cuda_bf16_amp):
                if hidden_capture_mode == "hook" and hook_target_module is not None:
                    with _LastHiddenCapture(hook_target_module, label="vq1.train") as cap:
                        _ = model(**inputs, output_hidden_states=False)
                        hidden_last = cap.hidden                                    # [B, T, H]
                else:
                    out = model(**inputs, output_hidden_states=True)
                    hidden_last = out.hidden_states[-1]                             # [B, T, H]
                h_t = _h_t_from_hidden(
                    hidden_last, inputs["input_ids"], vq_token_id,
                    inputs.get("attention_mask"),
                )                                                                   # [B, H]
                h_t_f = h_t.float()

                if global_step == 0 and step == 0:
                    log(
                        f"[vq1] capture: mode={hidden_capture_mode} "
                        f"hidden_last.shape={tuple(hidden_last.shape)} "
                        f"hidden_last.dtype={hidden_last.dtype} "
                        f"hidden_last.requires_grad={bool(hidden_last.requires_grad)} "
                        f"h_t.shape={tuple(h_t.shape)}"
                    )

                z_future_pred = future_head(h_t_f) if future_head is not None else None  # [B, D] or [B, N, D]

                z_for_q = z_future_pred if spec.q_input == "z" else h_t_f
                z_for_int = z_future_pred if spec.int_input_z == "z" else h_t_f

                batch_loss = torch.zeros((), device=device, dtype=torch.float32)

                if spec.use_future_loss and z_future_pred is not None and future_target.available:
                    z_tgt, valid_mask = _gather_future_targets_batched(
                        batch, train_ds, ep_bounds_train, future_target, args,
                    )
                    if z_tgt is not None and valid_mask is not None and bool(valid_mask.any()):
                        valid_idx = valid_mask.nonzero(as_tuple=True)[0].to(z_future_pred.device)
                        z_pred_valid = z_future_pred.index_select(0, valid_idx)
                        z_tgt = z_tgt.to(device=z_pred_valid.device, dtype=z_pred_valid.dtype)
                        if args.future_latent_mode == "dense":
                            z_tgt = _align_future_target_dense(z_pred_valid, z_tgt)
                        B_f = int(z_pred_valid.shape[0])
                        L_future = normalized_mse(z_pred_valid, z_tgt)
                        batch_loss = batch_loss + lambda_future * L_future.to(batch_loss.dtype)
                        agg["L_future"] += float(L_future.detach().item()) * B_f
                        agg["future_cos"] += float(cosine_sim_mean(z_pred_valid, z_tgt).detach().item()) * B_f
                        agg["n_future"] += B_f

                Q_min = None
                if twin_q_head is not None and z_for_q is not None:
                    q1, q2 = twin_q_head(z_for_q)                                   # [B], [B]
                    Q_min = torch.minimum(q1, q2)
                    v_tp1_np, terminal_np = _gather_proxy_v_tp1_batched(
                        batch, train_ds.transitions, ep_bounds_train,
                        latent_proxy_vf=latent_proxy_vf, dataset=train_ds,
                    )
                    y_q = torch.from_numpy(v_tp1_np).to(device=q1.device, dtype=q1.dtype)
                    non_term = torch.from_numpy(~terminal_np).to(device=q1.device)
                    if spec.use_q_loss and bool(non_term.any()):
                        nt_idx = non_term.nonzero(as_tuple=True)[0]
                        q1_v = q1.index_select(0, nt_idx)
                        q2_v = q2.index_select(0, nt_idx)
                        y_v = y_q.index_select(0, nt_idx)
                        B_q = int(nt_idx.numel())
                        L_q = F.smooth_l1_loss(q1_v, y_v) + F.smooth_l1_loss(q2_v, y_v)
                        batch_loss = batch_loss + lambda_q * L_q.to(batch_loss.dtype)
                        agg["L_q"] += float(L_q.detach().item()) * B_q
                        agg["Q1"] += float(q1_v.detach().mean().item()) * B_q
                        agg["Q2"] += float(q2_v.detach().mean().item()) * B_q
                        agg["Qmin"] += float(Q_min.index_select(0, nt_idx).detach().mean().item()) * B_q
                        agg["Vtgt"] += float(y_v.detach().mean().item()) * B_q
                        agg["n_q"] += B_q

                if spec.use_int_loss and z_for_int is not None:
                    y_int = torch.tensor(
                        [1.0 if bool(s.get("intervened", False)) else 0.0 for s in batch],
                        dtype=torch.float32, device=device,
                    )                                                               # [B]
                    logit_or_risk = None
                    if itype == "binary" and intervention_head is not None:
                        q_cond = Q_min if (spec.int_use_q_cond and Q_min is not None) else None
                        logit_or_risk = intervention_head(z_for_int, q_cond)        # [B]
                        if loss_type == "focal":
                            L_int = binary_focal_loss_with_logits(
                                logit_or_risk, y_int.to(logit_or_risk.dtype),
                                alpha=float(args.focal_alpha),
                                gamma=float(args.focal_gamma),
                            )
                        else:
                            L_int = F.binary_cross_entropy_with_logits(
                                logit_or_risk, y_int.to(logit_or_risk.dtype),
                                pos_weight=pos_w_t.to(logit_or_risk.dtype),
                            )
                        batch_loss = batch_loss + lambda_intervention * L_int.to(batch_loss.dtype)
                        agg["L_int"] += float(L_int.detach().item()) * B
                    elif itype in ("current_value", "tvr_history") and risk_head is not None:
                        _K = int(getattr(args, "value_history_window", 8))
                        _gam = float(getattr(args, "value_history_gamma", 0.9))
                        _eps = float(getattr(args, "value_history_epsilon", 0.005))
                        _lam_tvr = float(getattr(args, "lambda_tvr", 1.0))
                        vals_b, delts_b = _gather_value_history_batched(
                            batch, train_ds.transitions, ep_bounds_train, _K,
                        )                                                           # [B,K+1], [B,K]
                        if v_curr_head is not None:
                            v_curr_pred = v_curr_head(h_t_f)                        # [B]
                            proxy_vt = _gather_proxy_v_t_batched(batch, train_ds.transitions)
                            y_vcurr = torch.from_numpy(proxy_vt).to(device=device, dtype=v_curr_pred.dtype)
                            L_vcurr = F.smooth_l1_loss(v_curr_pred, y_vcurr)
                            batch_loss = batch_loss + float(getattr(args, "lambda_v_curr", 1.0)) * L_vcurr.to(batch_loss.dtype)
                            agg["L_vcurr"] += float(L_vcurr.detach().item()) * B
                            agg["n_vcurr"] += B
                        if itype == "current_value" and v_curr_head is not None:
                            context = v_curr_pred.detach().unsqueeze(-1)            # [B, 1]
                        else:
                            hist_np = np.concatenate([vals_b, delts_b], axis=1)     # [B, 2K+1]
                            hist_t = torch.from_numpy(hist_np).to(device=device, dtype=torch.float32)
                            context = value_history_encoder(hist_t)                 # [B, embed_dim]
                        risk_pred = risk_head(z_for_int, context)                   # [B]
                        logit_or_risk = risk_pred
                        if itype == "tvr_history":
                            tvr_np = _compute_tvr_targets_batched(vals_b, _gam, _eps)
                            y_tvr = torch.from_numpy(tvr_np).to(device=device, dtype=risk_pred.dtype)
                            L_tvr = F.smooth_l1_loss(risk_pred, y_tvr)
                            batch_loss = batch_loss + _lam_tvr * L_tvr.to(batch_loss.dtype)
                            agg["L_tvr"] += float(L_tvr.detach().item()) * B
                            agg["n_tvr"] += B
                            agg["risk_preds_raw"].extend(risk_pred.detach().float().cpu().tolist())
                            agg["tvr_targets_raw"].extend(tvr_np.tolist())
                        if loss_type in ("focal", "weighted_bce"):
                            if loss_type == "focal":
                                L_int = binary_focal_loss_with_logits(
                                    risk_pred, y_int.to(risk_pred.dtype),
                                    alpha=float(getattr(args, "focal_alpha", 0.75)),
                                    gamma=float(getattr(args, "focal_gamma", 2.0)),
                                )
                            else:
                                L_int = F.binary_cross_entropy_with_logits(
                                    risk_pred, y_int.to(risk_pred.dtype),
                                    pos_weight=pos_w_t.to(risk_pred.dtype),
                                )
                            batch_loss = batch_loss + lambda_intervention * L_int.to(batch_loss.dtype)
                            agg["L_int"] += float(L_int.detach().item()) * B

                    if logit_or_risk is not None:
                        p = torch.sigmoid(logit_or_risk).detach()                   # [B]
                        pred = (p > float(args.intervention_threshold)).float()
                        agg["n_int"] += int(B)
                        agg["tp"] += int(((pred == 1) & (y_int == 1)).sum().item())
                        agg["fp"] += int(((pred == 1) & (y_int == 0)).sum().item())
                        agg["fn"] += int(((pred == 0) & (y_int == 1)).sum().item())
                        agg["tn"] += int(((pred == 0) & (y_int == 0)).sum().item())
                        agg["p_int"] += float(p.sum().item())
                        agg["probs"].extend(p.float().cpu().tolist())
                        agg["labels"].extend(y_int.cpu().tolist())
                        agg["p_int_min"] = min(agg["p_int_min"], float(p.min().item()))
                        agg["p_int_max"] = max(agg["p_int_max"], float(p.max().item()))

            if not torch.isfinite(batch_loss):
                log(f"[skip] non-finite batch_loss at step {step}")
                optimizer.zero_grad()
                continue

            if device.type == "cuda" and global_step == 0:
                fwd_peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
                torch.cuda.reset_peak_memory_stats(device)
            batch_loss.backward()
            if device.type == "cuda" and global_step == 0:
                bwd_peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
                log(
                    f"[vq1] peak_cuda_alloc  forward={fwd_peak_gb:.2f} GB  "
                    f"backward={bwd_peak_gb:.2f} GB  "
                    f"batch_size={B} grad_ckpt={bool(getattr(args, 'gradient_checkpointing', False))}"
                )
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]], 1.0,
            )
            optimizer.step()
            agg["L_total"] += float(batch_loss.detach().item())
            agg["n_total"] += 1
            global_step += 1

            if step % int(args.log_interval) == 0:
                _vq1_log_epoch_stats(epoch, step, len(train_loader), agg, args, prefix="train", wandb_use=use_wandb)
            if int(args.max_train_steps) > 0 and global_step >= int(args.max_train_steps):
                log(f"[train] max_train_steps={args.max_train_steps} reached — stopping after this epoch")
                stop_training = True
                break

        _vq1_log_epoch_stats(epoch, None, None, agg, args, prefix="train_epoch", wandb_use=use_wandb)

        if not bool(getattr(args, "skip_val", False)):
            model.eval()
            for m in head_modules:
                if m is not None:
                    m.eval()
            v_agg = {k: (v if not isinstance(v, float) else 0.0) for k, v in agg.items()}
            for k in v_agg:
                v_agg[k] = 0
            v_agg = {
                "L_future": 0.0, "n_future": 0, "future_cos": 0.0,
                "L_q": 0.0, "n_q": 0, "Q1": 0.0, "Q2": 0.0, "Qmin": 0.0, "Vtgt": 0.0,
                "L_int": 0.0, "n_int": 0,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                "p_int": 0.0, "p_int_min": 1.0, "p_int_max": 0.0,
                "L_tvr": 0.0, "n_tvr": 0,
                "L_vcurr": 0.0, "n_vcurr": 0,
                "risk_preds_raw": [], "tvr_targets_raw": [],
                "L_total": 0.0, "n_total": 0,
                "probs": [], "labels": [],
            }
            with torch.no_grad():
                for batch in val_loader:
                    B = len(batch)
                    if B == 0:
                        continue

                    inputs = _build_vq1_inputs_batched(batch, processor, device)
                    with _cuda_bf16_autocast(device, use_cuda_bf16_amp):
                        if hidden_capture_mode == "hook" and hook_target_module is not None:
                            with _LastHiddenCapture(hook_target_module, label="vq1.val") as cap:
                                _ = model(**inputs, output_hidden_states=False)
                                hidden_last = cap.hidden
                        else:
                            out = model(**inputs, output_hidden_states=True)
                            hidden_last = out.hidden_states[-1]
                        h_t = _h_t_from_hidden(
                            hidden_last, inputs["input_ids"], vq_token_id,
                            inputs.get("attention_mask"),
                        )                                                           # [B, H]
                        h_t_f = h_t.float()
                        z_future_pred = future_head(h_t_f) if future_head is not None else None
                        z_for_q = z_future_pred if spec.q_input == "z" else h_t_f
                        z_for_int = z_future_pred if spec.int_input_z == "z" else h_t_f

                        if spec.use_future_loss and z_future_pred is not None and future_target.available:
                            z_tgt, valid_mask = _gather_future_targets_batched(
                                batch, val_ds, ep_bounds_val, future_target, args,
                            )
                            if z_tgt is not None and valid_mask is not None and bool(valid_mask.any()):
                                valid_idx = valid_mask.nonzero(as_tuple=True)[0].to(z_future_pred.device)
                                z_pred_valid = z_future_pred.index_select(0, valid_idx)
                                z_tgt = z_tgt.to(device=z_pred_valid.device, dtype=z_pred_valid.dtype)
                                if args.future_latent_mode == "dense":
                                    z_tgt = _align_future_target_dense(z_pred_valid, z_tgt)
                                B_f = int(z_pred_valid.shape[0])
                                v_agg["L_future"] += float(normalized_mse(z_pred_valid, z_tgt).item()) * B_f
                                v_agg["future_cos"] += float(cosine_sim_mean(z_pred_valid, z_tgt).item()) * B_f
                                v_agg["n_future"] += B_f

                        Q_min = None
                        if twin_q_head is not None and z_for_q is not None:
                            q1, q2 = twin_q_head(z_for_q)
                            Q_min = torch.minimum(q1, q2)
                            v_tp1_np, terminal_np = _gather_proxy_v_tp1_batched(
                                batch, val_ds.transitions, ep_bounds_val,
                                latent_proxy_vf=latent_proxy_vf, dataset=val_ds,
                            )
                            y_q = torch.from_numpy(v_tp1_np).to(device=q1.device, dtype=q1.dtype)
                            non_term = torch.from_numpy(~terminal_np).to(device=q1.device)
                            if spec.use_q_loss and bool(non_term.any()):
                                nt_idx = non_term.nonzero(as_tuple=True)[0]
                                q1_v = q1.index_select(0, nt_idx)
                                q2_v = q2.index_select(0, nt_idx)
                                y_v = y_q.index_select(0, nt_idx)
                                B_q = int(nt_idx.numel())
                                L_q = F.smooth_l1_loss(q1_v, y_v) + F.smooth_l1_loss(q2_v, y_v)
                                v_agg["L_q"] += float(L_q.item()) * B_q
                                v_agg["Q1"] += float(q1_v.mean().item()) * B_q
                                v_agg["Q2"] += float(q2_v.mean().item()) * B_q
                                v_agg["Qmin"] += float(Q_min.index_select(0, nt_idx).mean().item()) * B_q
                                v_agg["Vtgt"] += float(y_v.mean().item()) * B_q
                                v_agg["n_q"] += B_q

                        if spec.use_int_loss and z_for_int is not None:
                            y_int = torch.tensor(
                                [1.0 if bool(s.get("intervened", False)) else 0.0 for s in batch],
                                dtype=torch.float32, device=device,
                            )
                            logit_or_risk = None
                            if itype == "binary" and intervention_head is not None:
                                q_cond = Q_min if (spec.int_use_q_cond and Q_min is not None) else None
                                logit_or_risk = intervention_head(z_for_int, q_cond)
                                if loss_type == "focal":
                                    L_int = binary_focal_loss_with_logits(
                                        logit_or_risk, y_int.to(logit_or_risk.dtype),
                                        alpha=float(args.focal_alpha),
                                        gamma=float(args.focal_gamma),
                                    )
                                else:
                                    L_int = F.binary_cross_entropy_with_logits(
                                        logit_or_risk, y_int.to(logit_or_risk.dtype),
                                        pos_weight=pos_w_t.to(logit_or_risk.dtype),
                                    )
                                v_agg["L_int"] += float(L_int.item()) * B
                            elif itype in ("current_value", "tvr_history") and risk_head is not None:
                                _K = int(getattr(args, "value_history_window", 8))
                                _gam = float(getattr(args, "value_history_gamma", 0.9))
                                _eps = float(getattr(args, "value_history_epsilon", 0.005))
                                vals_b, delts_b = _gather_value_history_batched(
                                    batch, val_ds.transitions, ep_bounds_val, _K,
                                )
                                v_curr_pred = None
                                if v_curr_head is not None:
                                    v_curr_pred = v_curr_head(h_t_f)
                                    proxy_vt = _gather_proxy_v_t_batched(batch, val_ds.transitions)
                                    y_vcurr = torch.from_numpy(proxy_vt).to(device=device, dtype=v_curr_pred.dtype)
                                    L_vcurr = F.smooth_l1_loss(v_curr_pred, y_vcurr)
                                    v_agg["L_vcurr"] += float(L_vcurr.item()) * B
                                    v_agg["n_vcurr"] += B
                                if itype == "current_value" and v_curr_pred is not None:
                                    context = v_curr_pred.detach().unsqueeze(-1)
                                else:
                                    hist_np = np.concatenate([vals_b, delts_b], axis=1)
                                    hist_t = torch.from_numpy(hist_np).to(device=device, dtype=torch.float32)
                                    context = value_history_encoder(hist_t)
                                risk_pred = risk_head(z_for_int, context)
                                logit_or_risk = risk_pred
                                if itype == "tvr_history":
                                    tvr_np = _compute_tvr_targets_batched(vals_b, _gam, _eps)
                                    y_tvr = torch.from_numpy(tvr_np).to(device=device, dtype=risk_pred.dtype)
                                    L_tvr = F.smooth_l1_loss(risk_pred, y_tvr)
                                    v_agg["L_tvr"] += float(L_tvr.item()) * B
                                    v_agg["n_tvr"] += B
                                    v_agg["risk_preds_raw"].extend(risk_pred.float().cpu().tolist())
                                    v_agg["tvr_targets_raw"].extend(tvr_np.tolist())
                                if loss_type in ("focal", "weighted_bce"):
                                    if loss_type == "focal":
                                        L_int = binary_focal_loss_with_logits(
                                            risk_pred, y_int.to(risk_pred.dtype),
                                            alpha=float(getattr(args, "focal_alpha", 0.75)),
                                            gamma=float(getattr(args, "focal_gamma", 2.0)),
                                        )
                                    else:
                                        L_int = F.binary_cross_entropy_with_logits(
                                            risk_pred, y_int.to(risk_pred.dtype),
                                            pos_weight=pos_w_t.to(risk_pred.dtype),
                                        )
                                    v_agg["L_int"] += float(L_int.item()) * B

                            if logit_or_risk is not None:
                                p = torch.sigmoid(logit_or_risk)
                                pred = (p > float(args.intervention_threshold)).float()
                                v_agg["n_int"] += int(B)
                                v_agg["tp"] += int(((pred == 1) & (y_int == 1)).sum().item())
                                v_agg["fp"] += int(((pred == 1) & (y_int == 0)).sum().item())
                                v_agg["fn"] += int(((pred == 0) & (y_int == 1)).sum().item())
                                v_agg["tn"] += int(((pred == 0) & (y_int == 0)).sum().item())
                                v_agg["p_int"] += float(p.sum().item())
                                v_agg["probs"].extend(p.float().cpu().tolist())
                                v_agg["labels"].extend(y_int.cpu().tolist())
                                v_agg["p_int_min"] = min(v_agg["p_int_min"], float(p.min().item()))
                                v_agg["p_int_max"] = max(v_agg["p_int_max"], float(p.max().item()))
            _vq1_log_epoch_stats(epoch, None, None, v_agg, args, prefix="val", wandb_use=use_wandb)

        ckpt_dir = os.path.join(args.output_dir, f"vq1_epoch{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)
        head_state = {
            "future_head": future_head.state_dict() if future_head is not None else None,
            "twin_q_head": twin_q_head.state_dict() if twin_q_head is not None else None,
            "intervention_head": intervention_head.state_dict() if intervention_head is not None else None,
            "value_history_encoder": value_history_encoder.state_dict() if value_history_encoder is not None else None,
            "v_curr_head": v_curr_head.state_dict() if v_curr_head is not None else None,
            "risk_head": risk_head.state_dict() if risk_head is not None else None,
            "variant": args.variant,
            "future_latent_mode": args.future_latent_mode,
            "vjepa_feature_dim": target_dim,
            "vjepa_n_tokens": n_tokens,
            "hidden_size": hidden_size,
            "intervention_threshold": float(args.intervention_threshold),
            "vq_token_id": int(vq_token_id),
            "proxy_vf_provenance": proxy_vf_provenance,
            "intervention_loss_type": getattr(args, "intervention_loss_type", "weighted_bce"),
            "focal_alpha": float(getattr(args, "focal_alpha", 0.75)),
            "focal_gamma": float(getattr(args, "focal_gamma", 2.0)),
            "intervention_head_type": itype,
            "value_history_window": int(getattr(args, "value_history_window", 8)),
            "value_history_gamma": float(getattr(args, "value_history_gamma", 0.9)),
            "value_history_epsilon": float(getattr(args, "value_history_epsilon", 0.005)),
            "value_history_embed_dim": int(getattr(args, "value_history_embed_dim", 32)),
        }
        torch.save(head_state, os.path.join(ckpt_dir, "vq1_heads.pt"))
        log(f"[saved] {ckpt_dir}")

        if stop_training:
            break

    log("[vq1] done")
    if use_wandb:
        wandb.finish()


def _vq1_log_epoch_stats(epoch, step, total_steps, agg: dict, args, prefix: str, wandb_use: bool):
    n_q = max(1, int(agg.get("n_q", 0)))
    n_f = max(1, int(agg.get("n_future", 0)))
    n_i = max(1, int(agg.get("n_int", 0)))
    n_tot = max(1, int(agg.get("n_total", 0)))
    tp = int(agg["tp"]); fp = int(agg["fp"]); fn = int(agg["fn"]); tn = int(agg["tn"])
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = (2 * prec * rec / max(1e-9, prec + rec)) if (prec + rec) > 0 else 0.0
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    trig = (tp + fp) / max(1, tp + fp + fn + tn)
    stats = {
        f"{prefix}/L_total": agg["L_total"] / n_tot,
        f"{prefix}/L_future": agg["L_future"] / n_f,
        f"{prefix}/L_q": agg["L_q"] / n_q,
        f"{prefix}/L_intervention": agg["L_int"] / n_i,
        f"{prefix}/future_cosine": agg["future_cos"] / n_f,
        f"{prefix}/Q1_mean": agg["Q1"] / n_q,
        f"{prefix}/Q2_mean": agg["Q2"] / n_q,
        f"{prefix}/Q_min_mean": agg["Qmin"] / n_q,
        f"{prefix}/V_tp1_mean": agg["Vtgt"] / n_q,
        f"{prefix}/p_int_mean": agg["p_int"] / n_i,
        f"{prefix}/p_int_min": float(agg["p_int_min"]),
        f"{prefix}/p_int_max": float(agg["p_int_max"]),
        f"{prefix}/precision": prec,
        f"{prefix}/recall": rec,
        f"{prefix}/f1": f1,
        f"{prefix}/fpr": fpr,
        f"{prefix}/fnr": fnr,
        f"{prefix}/trigger_rate": trig,
        f"{prefix}/n_future": int(agg["n_future"]),
        f"{prefix}/n_q": int(agg["n_q"]),
        f"{prefix}/n_int": int(agg["n_int"]),
    }
    n_tvr = max(1, int(agg.get("n_tvr", 0)))
    n_vcurr = max(1, int(agg.get("n_vcurr", 0)))
    stats[f"{prefix}/L_tvr"] = agg.get("L_tvr", 0.0) / n_tvr
    stats[f"{prefix}/L_vcurr"] = agg.get("L_vcurr", 0.0) / n_vcurr
    risk_preds_raw = agg.get("risk_preds_raw") or []
    tvr_targets_raw = agg.get("tvr_targets_raw") or []
    corr_risk_tvr = float("nan")
    if len(risk_preds_raw) >= 2 and len(risk_preds_raw) == len(tvr_targets_raw):
        rp = np.asarray(risk_preds_raw, dtype=np.float64)
        tt = np.asarray(tvr_targets_raw, dtype=np.float64)
        if rp.std() > 1e-9 and tt.std() > 1e-9:
            corr_risk_tvr = float(np.corrcoef(rp, tt)[0, 1])
    stats[f"{prefix}/corr_risk_tvr"] = corr_risk_tvr
    stats[f"{prefix}/risk_pred_mean"] = float(np.mean(risk_preds_raw)) if risk_preds_raw else 0.0
    stats[f"{prefix}/tvr_target_mean"] = float(np.mean(tvr_targets_raw)) if tvr_targets_raw else 0.0

    itype_log = str(getattr(args, "intervention_head_type", "binary")).lower()
    step_str = "" if step is None else f" step {step}/{total_steps}"
    log(f"[{prefix}] epoch {epoch}{step_str}  "
        f"L_total={stats[f'{prefix}/L_total']:.4f}  "
        f"L_future={stats[f'{prefix}/L_future']:.4f}  "
        f"L_q={stats[f'{prefix}/L_q']:.4f}  "
        f"L_int={stats[f'{prefix}/L_intervention']:.4f}  "
        f"fcos={stats[f'{prefix}/future_cosine']:.3f}  "
        f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} FPR={fpr:.3f} FNR={fnr:.3f} trig={trig:.3f}  "
        f"Qmin={stats[f'{prefix}/Q_min_mean']:.3f} Vtgt={stats[f'{prefix}/V_tp1_mean']:.3f}")
    if itype_log in ("current_value", "tvr_history") and int(agg.get("n_tvr", 0)) > 0:
        log(f"[{prefix}] TVR: L_tvr={stats[f'{prefix}/L_tvr']:.4f}  "
            f"L_vcurr={stats[f'{prefix}/L_vcurr']:.4f}  "
            f"risk_mean={stats[f'{prefix}/risk_pred_mean']:.4f}  "
            f"tvr_tgt_mean={stats[f'{prefix}/tvr_target_mean']:.4f}  "
            f"corr(risk,TVR)={corr_risk_tvr:.3f}  n_tvr={agg.get('n_tvr',0)}")
    probs = agg.get("probs", []) or []
    labels = agg.get("labels", []) or []
    is_epoch_or_final = (step is None) or (prefix.endswith("epoch")) or (prefix == "val")
    if is_epoch_or_final and len(probs) > 0 and len(labels) == len(probs):
        sweep = threshold_sweep_metrics(np.asarray(probs), np.asarray(labels))
        stats[f"{prefix}/best_f1"] = sweep["best_f1"]
        stats[f"{prefix}/best_threshold"] = sweep["best_threshold"]
        stats[f"{prefix}/precision_at_best"] = sweep["precision_at_best"]
        stats[f"{prefix}/recall_at_best"] = sweep["recall_at_best"]
        stats[f"{prefix}/trigger_at_best"] = sweep["trigger_at_best"]
        stats[f"{prefix}/auprc"] = sweep["auprc"]
        log(
            f"[{prefix}] sweep: best_F1={sweep['best_f1']:.3f} @ thr={sweep['best_threshold']:.2f}  "
            f"P@best={sweep['precision_at_best']:.3f} R@best={sweep['recall_at_best']:.3f} "
            f"trig@best={sweep['trigger_at_best']:.3f}  AUPRC={sweep['auprc']:.3f}  "
            f"pos={sweep['n_pos']}/{sweep['n_total']}"
        )
    if wandb_use:
        stats["epoch"] = epoch
        if step is not None:
            stats["step"] = step
        wandb.log(stats)




def _build_batch_stage1_inputs(samples, processor, device, use_indicator=False):
    """Pack several samples into one processor call; returns (inputs_dict, prompt_lens)."""
    texts, images_list = [], []
    for sample in samples:
        user_text = build_stage1_user_text(sample)
        msgs = [
            {"role": "system", "content": SYSTEM_MSG},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in sample["images"]],
                    {"type": "text", "text": user_text},
                ],
            },
            {"role": "assistant", "content": sample["target"]},
        ]
        texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))
        images_list.append(sample["images"] if sample["images"] else None)

    inputs = processor(
        text=texts,
        images=images_list,
        return_tensors="pt",
        padding=True,
    ).to(device)

    prompt_lens = []
    for sample in samples:
        prompt_lens.append(_stage1_prompt_len(processor, sample))

    return inputs, prompt_lens


def _build_batch_round2_inputs(samples, processor, device):
    """Pack several round2 samples into one processor call; returns inputs_dict."""
    texts, images_list = [], []
    for sample in samples:
        sys_msg, suffix = _round2_msg_suffix(sample)
        imgs = sample.get("round2_images") or sample["images"]
        user_body = f"{sample['instr']}{suffix}"
        goal_block = sample.get("round2_goal_user_block", "")
        if goal_block:
            user_body = f"{user_body}\n\n{goal_block}"
        msgs = [
            {"role": "system", "content": sys_msg},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in imgs],
                    {"type": "text", "text": user_body},
                ],
            },
        ]
        texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        images_list.append(imgs if imgs else None)

    return processor(
        text=texts,
        images=images_list,
        return_tensors="pt",
        padding=True,
    ).to(device)


def _minibatch_stage1_forward(
    samples, model, processor, device, use_cuda_bf16_amp,
    stage1_ce_enabled, use_indicator, batch_total, mini_bs=32,
):
    """Batched forward over stage1 samples; returns the scalar loss tensor added to batch_loss and statistics."""
    loss_contrib = 0.0
    s1_sum, s1_n = 0.0, 0
    if not stage1_ce_enabled or not samples:
        return loss_contrib, s1_sum, s1_n

    for i in range(0, len(samples), mini_bs):
        chunk = samples[i: i + mini_bs]
        inputs, prompt_lens = _build_batch_stage1_inputs(chunk, processor, device, use_indicator)
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        for j, plen in enumerate(prompt_lens):
            labels[j, :plen] = -100
        if "attention_mask" in inputs:
            labels[inputs["attention_mask"] == 0] = -100
        with _cuda_bf16_autocast(device, use_cuda_bf16_amp):
            outputs = model(**inputs, labels=labels, output_hidden_states=False)
        loss_contrib = loss_contrib + outputs.loss * len(chunk) / batch_total
        s1_sum += float(outputs.loss.detach().item()) * len(chunk)
        s1_n += len(chunk)

    return loss_contrib, s1_sum, s1_n


def _minibatch_round2_forward_v2(
    samples, model, fast_dec, fast_dec_tokenizer, processor, device,
    use_cuda_bf16_amp, args, train_ds, batch_total, mini_bs=16,
):
    """Official FAST BPE teacher-forcing; returns loss and token accuracy."""
    loss_contrib = 0.0
    fd_ce_sum, fd_acc_sum, fd_n = 0.0, 0.0, 0

    fd_w = float(getattr(args, "fast_decoder_loss_weight", 1.0))
    label_smoothing = float(getattr(args, "fast_decoder_v2_label_smoothing", 0.0))

    for i in range(0, len(samples), mini_bs):
        chunk = samples[i: i + mini_bs]

        gt_list, valid_chunk = [], []
        for sample in chunk:
            gt = build_fast_bpe_gt_tokens(
                sample, fast_dec_tokenizer,
                train_ds.a_min, train_ds.a_max, int(train_ds.n_bins_act), device,
                s_mean=getattr(train_ds, "s_mean", None),
                s_std=getattr(train_ds, "s_std", None),
                clip_k=float(getattr(train_ds, "_state_clip_k", 4.0)),
            )
            if gt is not None:
                gt_list.append(gt)
                valid_chunk.append(sample)

        if not valid_chunk:
            continue

        max_len = max(int(g.numel()) for g in gt_list)
        if max_len > int(fast_dec.max_tokens):
            raise RuntimeError(
                f"FAST BPE target has {max_len} tokens, decoder max is {fast_dec.max_tokens}; "
                "increase --fast_decoder_v2_max_tokens instead of truncating supervision"
            )
        gt_batch = torch.full(
            (len(valid_chunk), max_len), int(fast_dec_tokenizer.pad_id),
            dtype=torch.long, device=device,
        )
        for k, g in enumerate(gt_list):
            gt_batch[k, :g.numel()] = g

        inputs_r2 = _build_batch_round2_inputs(valid_chunk, processor, device)
        with _cuda_bf16_autocast(device, use_cuda_bf16_amp):
            fd_backbone = model(**inputs_r2, output_hidden_states=True)
        fd_hidden = _pool_hidden_from_outputs_with_mask(
            fd_backbone, inputs_r2.get("attention_mask")
        )  # [B, H]
        if fd_hidden is None:
            continue

        with _cuda_bf16_autocast(device, use_cuda_bf16_amp):
            ctx = fast_dec.encode_context(fd_hidden)             # [B, 1, d_dec]
            logits = fast_dec.decode(ctx, gt_batch)
            fd_losses = fast_bpe_ce_loss(
                logits, gt_batch,
                pad_id=fast_dec_tokenizer.pad_id,
                label_smoothing=label_smoothing,
            )

        loss_contrib = loss_contrib + fd_w * fd_losses["ce"] * len(valid_chunk) / batch_total
        fd_ce_sum += float(fd_losses["ce"].detach().item()) * len(valid_chunk)
        fd_acc_sum += float(fd_losses["acc"].detach().item()) * len(valid_chunk)
        fd_n += len(valid_chunk)

    return loss_contrib, fd_ce_sum, fd_acc_sum, fd_n


def _minibatch_round2_forward_v1(
    samples, model, fast_dec, processor, device,
    use_cuda_bf16_amp, args, train_ds, batch_total, mini_bs=16,
):
    """Batched forward over round2 samples (V1 fast decoder); returns (loss_contrib, fd_l1_sum, fd_end_sum, fd_n)."""
    loss_contrib = 0.0
    fd_l1_sum, fd_end_sum, fd_n = 0.0, 0.0, 0
    fd_w = float(getattr(args, "fast_decoder_loss_weight", 1.0))

    for i in range(0, len(samples), mini_bs):
        chunk = samples[i: i + mini_bs]

        gt_list, valid_chunk = [], []
        for sample in chunk:
            gt_fd = build_fast_decoder_gt(
                sample, train_ds.a_min, train_ds.a_max, int(train_ds.n_bins_act), device,
                s_mean=getattr(train_ds, "s_mean", None),
                s_std=getattr(train_ds, "s_std", None),
                clip_k=float(getattr(train_ds, "_state_clip_k", 4.0)),
            )
            if gt_fd is not None and int(gt_fd.shape[0]) > 0:
                gt_list.append(gt_fd)      # [T, D] float
                valid_chunk.append(sample)

        if not valid_chunk:
            continue

        max_T = max(g.shape[0] for g in gt_list)
        D = gt_list[0].shape[1]
        gt_batch = torch.zeros(len(valid_chunk), max_T, D, dtype=torch.float32, device=device)
        for k, g in enumerate(gt_list):
            gt_batch[k, :g.shape[0]] = g

        inputs_r2 = _build_batch_round2_inputs(valid_chunk, processor, device)
        with _cuda_bf16_autocast(device, use_cuda_bf16_amp):
            fd_backbone = model(**inputs_r2, output_hidden_states=True)
        fd_hidden = _pool_hidden_from_outputs_with_mask(
            fd_backbone, inputs_r2.get("attention_mask")
        )
        if fd_hidden is None:
            continue

        with _cuda_bf16_autocast(device, use_cuda_bf16_amp):
            ctx = fast_dec.encode_context(fd_hidden)
            pred_batch = fast_dec.decode(ctx, gt_batch)
            fd_losses = fast_decoder_losses(
                pred_batch, gt_batch,
                tail_k=int(getattr(args, "fast_decoder_tail_k", 3)),
                margin=float(getattr(args, "fast_decoder_margin", 0.0)),
                end_weight=float(getattr(args, "fast_decoder_end_weight", 0.15)),
                tail_weight=float(getattr(args, "fast_decoder_tail_weight", 0.05)),
            )
            fd_total = fast_decoder_total_loss(
                fd_losses,
                l1_weight=float(getattr(args, "fast_decoder_l1_weight", 1.0)),
                end_weight=float(getattr(args, "fast_decoder_end_weight", 0.15)),
                tail_weight=float(getattr(args, "fast_decoder_tail_weight", 0.05)),
            )

        loss_contrib = loss_contrib + fd_w * fd_total * len(valid_chunk) / batch_total
        fd_l1_sum += float(fd_losses["l1"].detach().item()) * len(valid_chunk)
        fd_end_sum += float(fd_losses["end"].detach().item()) * len(valid_chunk)
        fd_n += len(valid_chunk)

    return loss_contrib, fd_l1_sum, fd_end_sum, fd_n




def train(args):
    log(f"[train] buffer_dir={args.buffer_dir}  model={args.model_path}")
    log(f"        use_indicator={getattr(args, 'use_indicator', False)}")
    log(
        "        round2_extra="
        f"tail_steps_w={float(args.round2_tail_steps_weight):.3f},"
        f"tail_k={int(args.round2_tail_steps_k)},"
        f"goal_end_w={float(args.round2_goal_end_weight):.3f},"
        f"len_floor_ratio={float(args.round2_length_floor_ratio):.3f},"
        f"len_penalty_w={float(args.round2_length_penalty_weight):.3f}"
    )
    log(
        "        round2_geo="
        f"profile={str(getattr(args, 'train_profile', 'default'))},"
        f"end_w={float(args.round2_geo_end_weight):.3f},"
        f"tail_w={float(args.round2_geo_tail_weight):.3f},"
        f"tail_k={int(args.round2_geo_tail_k)},"
        f"margin={float(args.round2_geo_margin):.4f},"
        f"vgain_tau={float(args.round2_geo_vgain_tau):.4f},"
        f"warmup_epochs={int(args.round2_geo_warmup_epochs)}"
    )
    log(
        "        twin_aux="
        f"use={bool(getattr(args, 'use_twin_aux', False))} "
        f"lambda={float(getattr(args, 'twin_loss_weight', 0.0)):.3f} "
        f"w_td={float(getattr(args, 'twin_td_weight', 1.0)):.3f} "
        f"w_prog={float(getattr(args, 'twin_prog_weight', 1.0)):.3f} "
        f"w_mc={float(getattr(args, 'twin_mc_weight', 0.0)):.3f}"
    )
    stage1_ce_enabled = not bool(getattr(args, "disable_stage1_ce", False))
    log(f"        stage1_ce=enabled={stage1_ce_enabled}")

    use_wandb = getattr(args, "use_wandb", False)
    if use_wandb:
        run_name = args.wandb_run_name or f"uniintervene_{int(time.time())}"
        init_kwargs = {
            "project": args.wandb_project,
            "name": run_name,
            "config": vars(args),
        }
        if getattr(args, "wandb_entity", None):
            init_kwargs["entity"] = args.wandb_entity
        wandb.init(**init_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_kw = {
        "stage1_include_reward_label": bool(getattr(args, "stage1_include_reward_label", False)),
        "round2_bridge_steps": int(getattr(args, "round2_bridge_steps", 0)),
        "round2_retrieval_source": str(getattr(args, "round2_retrieval_source", "memory_bank")),
        "round2_memory_bank_path": str(getattr(args, "memory_bank_path", "")),
        "round2_memory_bank_router_path": str(getattr(args, "memory_bank_router_json", "")),
        "round2_task_route_mode": str(getattr(args, "round2_task_route_mode", "strict")),
        "round2_memory_top_k": int(getattr(args, "memory_top_k", 5)),
        "round2_memory_goal_aggregate": str(getattr(args, "memory_goal_aggregate", "top1")),
        "intervention_label_mode": str(getattr(args, "intervention_label_mode", "recover_mask")),
        "intervention_label_source": str(getattr(args, "intervention_label_source", "auto")),
        "intervention_label_key": str(getattr(args, "intervention_label_key", "vf_mining_intervention_label")),
        "intervention_expand_steps": int(getattr(args, "intervention_expand_steps", 0)),
        "task_aware_instr": bool(getattr(args, "task_aware_instr", True)),
        "round2_task_aware_filter": bool(getattr(args, "round2_task_aware_filter", True)),
        "enable_round2": not bool(getattr(args, "stage1_only", False)),
        "round2_target_space": str(getattr(args, "round2_target_space", "state11_pose")),
        "state_dim": int(getattr(args, "state_dim", 11)),
        "state_clip_k": float(getattr(args, "state_clip_k", 4.0)),
        "use_external_wrist_keys": bool(getattr(args, "use_external_wrist_keys", False)),
    }
    ds_recover_cfg = RecoverTargetMiningConfig(
        value_key="vf_value_pred",
        value_normalization="minmax_episode",
        require_intervention_signal=False,
        decline_window=int(getattr(args, "decline_window", 8)),
        min_down_ratio=float(getattr(args, "min_down_ratio", 0.7)),
        min_drop=float(getattr(args, "min_drop", 0.05)),
        intervention_margin=int(getattr(args, "intervention_margin", 2)),
    )
    ds_kw["recover_mining_config"] = ds_recover_cfg
    if getattr(args, "instr", None):
        ds_kw["instr"] = str(args.instr)
    req_vf = str(getattr(args, "required_vf_dir", "") or "").strip()
    ds_kw["required_vf_dir"] = req_vf if req_vf else None
    train_ds = HILVLMDataset(args.buffer_dir, split="train", **ds_kw)
    val_ds_kw = dict(ds_kw)
    val_ds_kw["state_dim"] = int(getattr(train_ds, "_state_dim", 11))
    val_ds_kw["state_clip_k"] = float(getattr(train_ds, "_state_clip_k", 4.0))
    val_ds   = HILVLMDataset(
        args.buffer_dir,
        split="val",
        action_min=train_ds.a_min,
        action_max=train_ds.a_max,
        state_mean=train_ds.s_mean,
        state_std=train_ds.s_std,
        **val_ds_kw,
    )
    log(f"[data] train={len(train_ds)}  val={len(val_ds)}")
    _register_train_aux_for_validation(train_ds, val_ds)
    geo_gate_mode = str(getattr(args, "round2_geo_gate_mode", "goal_vf_delta")).strip().lower()
    geo_gate_quantile = float(getattr(args, "round2_geo_gate_quantile", 0.3))
    geo_gate_quantile_tau = 0.0
    if geo_gate_mode == "goal_vf_delta_quantile":
        gate_vals: list[float] = []
        for i in range(len(train_ds)):
            sample_i = train_ds[i]
            if bool(sample_i.get("has_round2", False)):
                gate_vals.append(float(sample_i.get("round2_goal_vf_delta", 0.0)))
        if len(gate_vals) > 0:
            q = float(np.clip(geo_gate_quantile, 0.0, 1.0))
            geo_gate_quantile_tau = float(np.quantile(np.asarray(gate_vals, dtype=np.float32), q))
        log(
            f"[round2_geo] gate_mode=goal_vf_delta_quantile "
            f"q={geo_gate_quantile:.2f} tau={geo_gate_quantile_tau:.4f} n={len(gate_vals)}"
        )
    else:
        log(
            f"[round2_geo] gate_mode={geo_gate_mode} "
            f"vgain_tau={float(args.round2_geo_vgain_tau):.4f}"
        )

    if getattr(args, "use_indicator", False):
        log("[weight] using VF indicator sampling weights (good 2.0 / bad 0.5 / unscored 1.0)")
        log("         note: Intervention labels still depend only on human intervened, not on the indicator")
        w = make_indicator_weights(train_ds.transitions)
    else:
        log("[weight] using uniform sampling (default; NTTG bin balancing disabled)")
        w = np.ones(len(train_ds.transitions), dtype=float)

    stage1_yes_boost = float(getattr(args, "stage1_yes_boost", 1.0))
    if stage1_yes_boost > 1.0:
        yes_mask = np.array(
            [bool(train_ds.intervention_label_at(i)) for i in range(len(train_ds))],
            dtype=bool,
        )
        boosted, stats = apply_binary_mask_boost(w, yes_mask, stage1_yes_boost)
        if boosted is not None and stats is not None:
            w = boosted
            log(
                f"[weight] stage1 yes boost x{stage1_yes_boost:.2f}  "
                f"boosted={int(stats['n_hit'])}/{int(stats['n_total'])}  "
                f"before_range=[{stats['before_min']:.4f},{stats['before_max']:.4f}]  "
                f"after_range=[{stats['after_min']:.4f},{stats['after_max']:.4f}]  "
                f"effective_yes_share={stats['effective_share']:.4f}"
            )

    boost = float(getattr(args, "round2_sample_boost", 1.0))
    if boost > 1.0:
        m = np.array([bool(train_ds.intervention_label_at(i)) for i in range(len(train_ds))], dtype=bool)
        boosted, stats = apply_binary_mask_boost(w, m, boost)
        if boosted is not None and stats is not None:
            w = boosted
            log(
                f"[weight] round2/intervention boost x{boost:.2f}  "
                f"boosted={int(stats['n_hit'])}/{int(stats['n_total'])}  "
                f"before_range=[{stats['before_min']:.4f},{stats['before_max']:.4f}]  "
                f"after_range=[{stats['after_min']:.4f},{stats['after_max']:.4f}]  "
                f"effective_share={stats['effective_share']:.4f}"
            )

    if not stage1_ce_enabled:
        round2_indices = np.array(
            [i for i in range(len(train_ds)) if train_ds.intervention_label_at(i)],
            dtype=np.int64,
        )
        log(f"[data] stage1_ce=False → DataLoader filtered to round2-only: {len(round2_indices)}/{len(train_ds)} samples")
        w_eff = w[round2_indices]
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(w_eff),
            num_samples=len(round2_indices),
            replacement=True,
        )
        train_ds_eff = Subset(train_ds, round2_indices.tolist())
    else:
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(w),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_ds_eff = train_ds

    train_loader = DataLoader(
        train_ds_eff,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    ep_bounds_train = _build_episode_bounds_map(train_ds.episode_spans, len(train_ds.transitions))
    ep_bounds_val = _build_episode_bounds_map(val_ds.episode_spans, len(val_ds.transitions))

    log(f"[model] loading {args.model_path} ...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    if getattr(args, "load_in_4bit", False) and not getattr(args, "use_lora", False):
        raise SystemExit("--load_in_4bit requires --use_lora (QLoRA)")
    if getattr(args, "resume_lora_path", "") and not getattr(args, "use_lora", False):
        raise SystemExit("--resume_lora_path requires --use_lora")

    model_cls, model_cls_name = _resolve_qwen_vl_model_cls(args.model_path)
    log(f"[model] model_cls={model_cls_name}")

    if getattr(args, "load_in_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if device.type != "cuda":
            raise SystemExit("--load_in_4bit requires CUDA")
        _dev_ix = torch.cuda.current_device()
        model = model_cls.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map={"": _dev_ix},
            trust_remote_code=False,
        )
        log("[model] 4-bit quantized base model (QLoRA)")
    else:
        model = model_cls.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        ).to(device)

    if getattr(args, "use_lora", False):
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training

        if getattr(args, "load_in_4bit", False):
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool(
                    getattr(args, "gradient_checkpointing", False)
                ),
            )

        vq2_adapter_mode = str(getattr(args, "vq2_adapter_mode", "base_lora")).lower()
        vq1_shared_lora_path = str(getattr(args, "vq1_shared_lora_path", "") or "").strip()
        resume_lora = str(getattr(args, "resume_lora_path", "") or "").strip()

        if vq2_adapter_mode == "init_from_vq1":
            if not vq1_shared_lora_path or not os.path.isdir(vq1_shared_lora_path):
                raise SystemExit(
                    f"--vq2_adapter_mode=init_from_vq1 requires a valid --vq1_shared_lora_path "
                    f"(got {vq1_shared_lora_path!r})"
                )
            if resume_lora:
                if not os.path.isdir(resume_lora):
                    raise SystemExit(f"--resume_lora_path not found: {resume_lora}")
                model = PeftModel.from_pretrained(model, resume_lora, is_trainable=True)
                model.enable_input_require_grads()
                log(f"[vq2] adapter_mode=init_from_vq1  resumed from {resume_lora} (trainable)")
            else:
                model = PeftModel.from_pretrained(model, vq1_shared_lora_path, is_trainable=True)
                model.enable_input_require_grads()
                log(f"[vq2] adapter_mode=init_from_vq1  init from {vq1_shared_lora_path} (trainable)")
        elif vq2_adapter_mode == "shared_residual":
            if not vq1_shared_lora_path or not os.path.isdir(vq1_shared_lora_path):
                raise SystemExit(
                    f"--vq2_adapter_mode=shared_residual requires a valid --vq1_shared_lora_path "
                    f"(got {vq1_shared_lora_path!r})"
                )
            model = PeftModel.from_pretrained(
                model, vq1_shared_lora_path,
                adapter_name="vq1_shared",
                is_trainable=False,
            )
            if bool(getattr(args, "freeze_vq1_shared_lora_for_vq2", True)):
                for n, p in model.named_parameters():
                    if "vq1_shared" in n:
                        p.requires_grad = False
            r_resid = int(getattr(args, "vq2_residual_lora_r", args.lora_r))
            a_resid = int(getattr(args, "vq2_residual_lora_alpha", args.lora_alpha))
            d_resid = float(getattr(args, "vq2_residual_lora_dropout", 0.05))
            residual_cfg = LoraConfig(
                r=r_resid, lora_alpha=a_resid, lora_dropout=d_resid,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type=TaskType.CAUSAL_LM,
            )
            model.add_adapter("vq2_residual", residual_cfg)
            _adapters = ["vq1_shared", "vq2_residual"]
            # Activate both adapters: the PeftModel wrapper takes a list of
            # names, and the LoRA model underneath accepts one as well.
            try:
                model.set_adapter(_adapters)
            except TypeError:
                model.base_model.set_adapter(_adapters)
            _active = getattr(model, "active_adapters", None) or getattr(
                model.base_model, "active_adapters", None
            )
            _active = list(_active) if _active is not None else []
            if sorted(_active) != sorted(_adapters):
                raise RuntimeError(
                    "The installed PEFT version cannot compose the VQ1 shared and VQ2 "
                    f"residual adapters (active={_active}). Install a version that "
                    "supports activating both adapters, e.g. peft>=0.17."
                )
            log(f"[vq2] active adapters: {sorted(_active)}")
            for n, p in model.named_parameters():
                if "vq1_shared" in n:
                    p.requires_grad = False
            if resume_lora:
                if not os.path.isdir(resume_lora):
                    raise SystemExit(f"--resume_lora_path not found: {resume_lora}")
                cand_files = [
                    os.path.join(resume_lora, "adapter_model.safetensors"),
                    os.path.join(resume_lora, "adapter_model.bin"),
                    os.path.join(resume_lora, "pytorch_model.bin"),
                    os.path.join(resume_lora, "vq2_residual", "adapter_model.safetensors"),
                    os.path.join(resume_lora, "vq2_residual", "adapter_model.bin"),
                ]
                st_path = None
                for pth in cand_files:
                    if os.path.isfile(pth):
                        st_path = pth
                        break
                if st_path is None:
                    raise SystemExit(
                        f"--resume_lora_path has no adapter weight file (adapter_model.safetensors/bin): {resume_lora}"
                    )
                resume_state = None
                try:
                    if st_path.endswith(".safetensors"):
                        from safetensors.torch import load_file as _safe_load_file  # type: ignore
                        resume_state = _safe_load_file(st_path)
                    else:
                        resume_state = torch.load(st_path, map_location="cpu", weights_only=True)
                except Exception as exc:
                    raise SystemExit(f"failed to read --resume_lora_path weights: {st_path} ({exc})")
                if not isinstance(resume_state, dict) or len(resume_state) == 0:
                    raise SystemExit(f"--resume_lora_path weights are empty or malformed: {st_path}")
                try:
                    from peft import set_peft_model_state_dict
                    set_peft_model_state_dict(model, resume_state, adapter_name="vq2_residual")
                    log(f"[vq2] resumed residual adapter weights from {st_path}")
                except Exception as exc:
                    raise SystemExit(
                        f"failed to restore vq2_residual in shared_residual mode: {st_path} ({exc})"
                    )
                for n, p in model.named_parameters():
                    if "vq1_shared" in n:
                        p.requires_grad = False
            model.enable_input_require_grads()
            log(f"[vq2] active adapters: vq1_shared (frozen) + vq2_residual (trainable) "
                f"both_active=True")
            log(
                f"[vq2] adapter_mode=shared_residual  base + frozen 'vq1_shared' from "
                f"{vq1_shared_lora_path} + trainable 'vq2_residual' (r={r_resid},alpha={a_resid})"
            )
        elif resume_lora:
            if not os.path.isdir(resume_lora):
                raise SystemExit(f"--resume_lora_path not found: {resume_lora}")
            model = PeftModel.from_pretrained(model, resume_lora, is_trainable=True)
            model.enable_input_require_grads()
            log(f"[model] resuming LoRA training: {resume_lora}")
        else:
            lc = LoraConfig(
                r=int(args.lora_r),
                lora_alpha=int(args.lora_alpha),
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lc)
            model.enable_input_require_grads()
            log("[model] LoRA injected (training adapters only; same target_modules as the QwenActor default)")
        model.print_trainable_parameters()

    device = next(model.parameters()).device
    if getattr(args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        log("[model] gradient_checkpointing enabled (lower activation memory)")
    else:
        log("[model] gradient_checkpointing disabled (avoids recomputation overhead)")
        torch.backends.cudnn.benchmark = True
    model.train()

    twin_heads = None
    if bool(getattr(args, "use_twin_aux", False)):
        hsz = int(model.config.hidden_size)
        twin_heads = TwinVFHeads(hidden_size=hsz).to(device=device, dtype=torch.float32)
        log(f"[twin] initialized twin vf heads with hidden_size={hsz}")
        log("[twin] feature source=prompt-only deploy-style hidden (system+user, no assistant target)")
        resume_dir = str(getattr(args, "resume_twin_head_path", "") or "").strip()
        if resume_dir:
            head_path = os.path.join(resume_dir, "twin_heads.pt")
            if os.path.isfile(head_path):
                state = torch.load(head_path, map_location=device, weights_only=True)
                twin_heads.load_state_dict(state, strict=False)
                log(f"[twin] resumed twin heads from {head_path}")
            else:
                log(f"[twin] resume path missing twin_heads.pt: {head_path} (keep reinit)")

    fast_dec = None
    fast_dec_tokenizer = None  # only set when V2 is enabled
    fast_dec_is_v2 = bool(getattr(args, "use_fast_decoder_v2", False))
    if bool(getattr(args, "use_fast_decoder", False)) or fast_dec_is_v2:
        cfg_top = model.config
        cfg_text = getattr(cfg_top, "text_config", None)
        _hsz = int(
            getattr(cfg_top, "hidden_size", None)
            or (getattr(cfg_text, "hidden_size", None) if cfg_text is not None else None)
            or 2048
        )
        _r2_space = str(getattr(args, "round2_target_space", "state")).strip().lower()
        if _r2_space == "state11_pose":
            _fd_dim = 7
        elif _r2_space == "state":
            _fd_dim = int(getattr(train_ds, "_state_dim", 11))
        else:
            _fd_dim = int(train_ds.a_min.shape[0])

        if fast_dec_is_v2:
            _v2_max_T = int(getattr(args, "fast_decoder_v2_max_T", 8))
            fast_dec_tokenizer = FASTBPETokenizer(
                processor_path=str(getattr(args, "fast_tokenizer_path", "physical-intelligence/fast")),
                action_dim=_fd_dim,
                horizon=_v2_max_T,
            )
            fast_dec = FASTBPEActionDecoder(
                backbone_hidden=_hsz,
                vocab_size=fast_dec_tokenizer.model_vocab_size,
                pad_id=fast_dec_tokenizer.pad_id,
                eos_id=fast_dec_tokenizer.eos_id,
                max_tokens=int(getattr(args, "fast_decoder_v2_max_tokens", 128)),
                d_dec=int(getattr(args, "fast_decoder_v2_d_dec", 256)),
                n_layers=int(getattr(args, "fast_decoder_v2_n_layers", 4)),
                n_heads=int(getattr(args, "fast_decoder_v2_n_heads", 8)),
            ).to(device=device, dtype=torch.float32)
            n_fd = sum(p.numel() for p in fast_dec.parameters())
            log(f"[fast_dec_v2] official FAST initialized: processor={fast_dec_tokenizer.processor_path} "
                f"hidden={_hsz} dim={_fd_dim} H={_v2_max_T} vocab={fast_dec.vocab_size} "
                f"max_tokens={fast_dec.max_tokens} params={n_fd:,}")
        else:
            fast_dec = FastDecoder(
                backbone_hidden=_hsz,
                action_dim=_fd_dim,
                d_dec=int(getattr(args, "fast_decoder_d_dec", 256)),
                n_layers=int(getattr(args, "fast_decoder_n_layers", 2)),
                n_heads=int(getattr(args, "fast_decoder_n_heads", 4)),
                max_traj_len=int(getattr(args, "fast_decoder_max_traj_len", 128)),
            ).to(device=device, dtype=torch.float32)
            n_fd = sum(p.numel() for p in fast_dec.parameters())
            log(f"[fast_dec] initialized: hidden={_hsz} dim={_fd_dim} (space={_r2_space}) "
                f"d_dec={fast_dec.d_dec} n_layers={len(fast_dec.blocks)} params={n_fd:,}")

        _fd_resume = str(getattr(args, "resume_fast_decoder_path", "") or "").strip()
        if _fd_resume:
            _fd_pt = os.path.join(_fd_resume, "fast_decoder.pt")
            if os.path.isfile(_fd_pt):
                fast_dec.load_state_dict(torch.load(_fd_pt, map_location=device, weights_only=True))
                log(f"[fast_dec] resumed from {_fd_pt}")
            else:
                log(f"[fast_dec] resume path missing fast_decoder.pt: {_fd_pt} (keep reinit)")
            if fast_dec_is_v2:
                _tk_pt = os.path.join(_fd_resume, "fast_tokenizer.json")
                if os.path.isfile(_tk_pt):
                    import json as _json
                    with open(_tk_pt, "r") as _f:
                        _tk_sd = _json.load(_f)
                    fast_dec_tokenizer = FASTBPETokenizer.from_state_dict(
                        _tk_sd,
                        processor_path=str(getattr(args, "fast_tokenizer_path", "")),
                    )
                    log(f"[fast_dec_v2] tokenizer resumed from {_tk_pt}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    twin_trainable = []
    if twin_heads is not None:
        twin_trainable = [p for p in twin_heads.parameters() if p.requires_grad]
    fast_dec_trainable = []
    if fast_dec is not None:
        fast_dec_trainable = [p for p in fast_dec.parameters() if p.requires_grad]
    opt_params = trainable + twin_trainable + fast_dec_trainable
    if not trainable:
        raise RuntimeError("no parameters with requires_grad")
    if twin_heads is not None and not twin_trainable:
        raise RuntimeError("twin_heads enabled but has no trainable parameters")

    if getattr(args, "adam8bit", False):
        try:
            import bitsandbytes as bnb  # noqa: F401

            optimizer = bnb.optim.AdamW8bit(
                opt_params, lr=args.lr, weight_decay=0.01
            )
            log("[opt] AdamW8bit (bitsandbytes, trainable parameters only)")
        except Exception as e:
            log(f"[opt] AdamW8bit unavailable ({e}), falling back to AdamW")
            optimizer = torch.optim.AdamW(
                opt_params, lr=args.lr, weight_decay=0.01
            )
    else:
        optimizer = torch.optim.AdamW(
            opt_params, lr=args.lr, weight_decay=0.01
        )
    n_model_train = int(sum(p.numel() for p in trainable))
    n_twin_train = int(sum(p.numel() for p in twin_trainable))
    n_fast_train = int(sum(p.numel() for p in fast_dec_trainable))
    log(f"[opt] trainable summary: model={n_model_train}  twin_heads={n_twin_train}  fast_dec={n_fast_train}  total={n_model_train+n_twin_train+n_fast_train}")
    os.makedirs(args.output_dir, exist_ok=True)

    use_cuda_bf16_amp = device.type == "cuda" and not getattr(
        args, "no_cuda_bf16_autocast", False
    )
    if use_cuda_bf16_amp:
        log("[train] using bfloat16 autocast for the CUDA forward (avoids PEFT/cuBLASLt issues on some setups)")

    eos_token_id = int(processor.tokenizer.eos_token_id or -1)
    bin_token_ids, bin_token_values = _build_bin_token_table(processor.tokenizer, int(train_ds.n_bins_act))
    if bin_token_ids.numel() <= 0:
        log("[round2_geo] bin token table is empty; the geometric auxiliary loss will be skipped")
    else:
        log(f"[round2_geo] bin token coverage={int(bin_token_ids.numel())}/{int(train_ds.n_bins_act)+1}")

    global_step = 0
    stop_training = False
    retrieval_audit_dir = os.path.join(args.output_dir, "retrieval_audit")
    for epoch in range(1, args.epochs + 1):
        epoch_retrieval_rows: list[dict[str, object]] = []
        model.train()
        if twin_heads is not None:
            twin_heads.train()
        if fast_dec is not None:
            fast_dec.train()
        total_loss, n_steps = 0.0, 0
        epoch_s1_sum, epoch_s1_n = 0.0, 0
        epoch_r2ce_sum, epoch_r2ce_n = 0.0, 0
        epoch_geo_end_sum, epoch_geo_end_n = 0.0, 0
        epoch_geo_tail_sum, epoch_geo_tail_n = 0.0, 0
        epoch_twin_total_sum, epoch_twin_n = 0.0, 0
        epoch_twin_td_sum, epoch_twin_prog_sum, epoch_twin_mc_sum = 0.0, 0.0, 0.0
        epoch_twin_p1_sum, epoch_twin_p2_sum, epoch_twin_pmin_sum = 0.0, 0.0, 0.0
        epoch_twin_miss = 0
        epoch_fd_l1_sum, epoch_fd_end_sum, epoch_fd_n = 0.0, 0.0, 0
        geo_end_w, geo_tail_w = _geo_weights_for_epoch(args, epoch)

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch_loss = 0.0
            epoch_retrieval_rows.extend(_collect_round2_retrieval_rows(batch, epoch=epoch, split="train"))

            s1_contrib, s1_sum, s1_n = _minibatch_stage1_forward(
                batch, model, processor, device, use_cuda_bf16_amp,
                stage1_ce_enabled,
                use_indicator=getattr(args, "use_indicator", False),
                batch_total=len(batch),
                mini_bs=128,
            )
            batch_loss = batch_loss + s1_contrib
            epoch_s1_sum += s1_sum
            epoch_s1_n += s1_n

            if fast_dec is not None:
                round2_samples = [s for s in batch if s.get("has_round2")] if stage1_ce_enabled else batch
                if round2_samples:
                    if fast_dec_is_v2 and fast_dec_tokenizer is not None:
                        fd_contrib, fd_ce_sum, fd_acc_sum, fd_n_step = _minibatch_round2_forward_v2(
                            round2_samples, model, fast_dec, fast_dec_tokenizer,
                            processor, device, use_cuda_bf16_amp, args, train_ds,
                            batch_total=len(batch),
                            mini_bs=128,
                        )
                        batch_loss = batch_loss + fd_contrib
                        epoch_fd_l1_sum += fd_ce_sum
                        epoch_fd_end_sum += fd_acc_sum
                        epoch_fd_n += fd_n_step
                    else:
                        fd_contrib, fd_l1_sum, fd_end_sum, fd_n_step = _minibatch_round2_forward_v1(
                            round2_samples, model, fast_dec,
                            processor, device, use_cuda_bf16_amp, args, train_ds,
                            batch_total=len(batch),
                            mini_bs=128,
                        )
                        batch_loss = batch_loss + fd_contrib
                        epoch_fd_l1_sum += fd_l1_sum
                        epoch_fd_end_sum += fd_end_sum
                        epoch_fd_n += fd_n_step

            if twin_heads is not None:
                for sample in batch:
                    tgt = _twin_targets_from_sample(
                        sample=sample,
                        transitions=train_ds.transitions,
                        ep_bounds=ep_bounds_train,
                        device=device,
                        gamma=float(getattr(args, "twin_gamma", 0.99)),
                    )
                    prompt_inputs = build_prompt_only_input(sample, processor, SYSTEM_MSG, device)
                    with _cuda_bf16_autocast(device, use_cuda_bf16_amp):
                        twin_outputs = model(**prompt_inputs, output_hidden_states=True)
                    pooled = _pool_hidden_from_outputs_with_mask(
                        twin_outputs,
                        prompt_inputs.get("attention_mask"),
                    )
                    if tgt is None or pooled is None:
                        epoch_twin_miss += 1
                    else:
                        pred1, pred2 = twin_heads(pooled.float())
                        next_pred_min = torch.minimum(pred1.detach(), pred2.detach())
                        twin_loss_raw, twin_stats = _compute_twin_loss(pred1, pred2, next_pred_min, tgt, args)
                        twin_loss = float(args.twin_loss_weight) * twin_loss_raw
                        batch_loss = batch_loss + twin_loss / len(batch)
                        epoch_twin_total_sum += float(twin_loss.detach().item())
                        epoch_twin_td_sum += twin_stats["loss_td"]
                        epoch_twin_prog_sum += twin_stats["loss_prog"]
                        epoch_twin_mc_sum += twin_stats["loss_mc"]
                        epoch_twin_p1_sum += twin_stats["pred1"]
                        epoch_twin_p2_sum += twin_stats["pred2"]
                        epoch_twin_pmin_sum += twin_stats["pred_min"]
                        epoch_twin_n += 1

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            n_steps += 1
            global_step += 1
            if getattr(args, "max_train_steps", 0) and global_step >= int(
                args.max_train_steps
            ):
                log(
                    f"[train] max_train_steps={args.max_train_steps} reached; stopping after this epoch"
                )
                stop_training = True
                break

            if step % args.log_interval == 0:
                avg_step_loss = total_loss / max(n_steps, 1)
                s1_mean = epoch_s1_sum / max(1, epoch_s1_n)
                r2ce_mean = epoch_r2ce_sum / max(1, epoch_r2ce_n)
                lend_mean = epoch_geo_end_sum / max(1, epoch_geo_end_n)
                ltail_mean = epoch_geo_tail_sum / max(1, epoch_geo_tail_n)
                twin_mean = epoch_twin_total_sum / max(1, epoch_twin_n)
                twin_td_mean = epoch_twin_td_sum / max(1, epoch_twin_n)
                twin_prog_mean = epoch_twin_prog_sum / max(1, epoch_twin_n)
                twin_mc_mean = epoch_twin_mc_sum / max(1, epoch_twin_n)
                twin_pmin_mean = epoch_twin_pmin_sum / max(1, epoch_twin_n)
                fd_l1_mean = epoch_fd_l1_sum / max(1, epoch_fd_n)
                fd_end_mean = epoch_fd_end_sum / max(1, epoch_fd_n)
                log(
                    f"  epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss={avg_step_loss:.4f}  "
                    f"L_stage1_intervene={s1_mean:.4f}  "
                    f"L_round2_ce={r2ce_mean:.4f}  "
                    f"L_end={lend_mean:.4f}  "
                    f"L_tail={ltail_mean:.4f}  "
                    f"L_fd_l1={fd_l1_mean:.4f} L_fd_end={fd_end_mean:.4f} (n={epoch_fd_n})  "
                    f"L_twin={twin_mean:.4f} td={twin_td_mean:.4f} prog={twin_prog_mean:.4f} mc={twin_mc_mean:.4f} "
                    f"vf_min={twin_pmin_mean:.4f} miss={epoch_twin_miss}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": avg_step_loss,
                            "epoch": epoch,
                            "step": step,
                            "train/L_stage1_intervene": s1_mean,
                            "train/L_round2_ce": r2ce_mean,
                            "train/L_end": lend_mean,
                            "train/L_tail": ltail_mean,
                            "train/loss_twin_total": twin_mean,
                            "train/loss_td": twin_td_mean,
                            "train/loss_prog": twin_prog_mean,
                            "train/loss_mc": twin_mc_mean,
                            "train/vf_pred_min_mean": twin_pmin_mean,
                            "train/twin_missing_labels": epoch_twin_miss,
                        }
                    )
            if int(getattr(args, "save_every_epochs", 0)) <= 0 and args.save_every_steps and (step + 1) % args.save_every_steps == 0:
                latest_dir = os.path.join(args.output_dir, "model_latest")
                model.save_pretrained(latest_dir)
                processor.save_pretrained(latest_dir)
                if twin_heads is not None:
                    torch.save(twin_heads.state_dict(), os.path.join(latest_dir, "twin_heads.pt"))
                if fast_dec is not None:
                    torch.save(fast_dec.state_dict(), os.path.join(latest_dir, "fast_decoder.pt"))
                    if fast_dec_is_v2 and fast_dec_tokenizer is not None:
                        import json as _json
                        with open(os.path.join(latest_dir, "fast_tokenizer.json"), "w") as _f:
                            _json.dump(fast_dec_tokenizer.state_dict(), _f)
                log(f"  [saved] {latest_dir} (step {step+1})")

        avg_loss = total_loss / max(n_steps, 1)
        epoch_s1_mean = epoch_s1_sum / max(1, epoch_s1_n)
        epoch_r2ce_mean = epoch_r2ce_sum / max(1, epoch_r2ce_n)
        epoch_lend_mean = epoch_geo_end_sum / max(1, epoch_geo_end_n)
        epoch_ltail_mean = epoch_geo_tail_sum / max(1, epoch_geo_tail_n)
        epoch_twin_mean = epoch_twin_total_sum / max(1, epoch_twin_n)
        epoch_twin_td_mean = epoch_twin_td_sum / max(1, epoch_twin_n)
        epoch_twin_prog_mean = epoch_twin_prog_sum / max(1, epoch_twin_n)
        epoch_twin_mc_mean = epoch_twin_mc_sum / max(1, epoch_twin_n)
        epoch_twin_p1_mean = epoch_twin_p1_sum / max(1, epoch_twin_n)
        epoch_twin_p2_mean = epoch_twin_p2_sum / max(1, epoch_twin_n)
        epoch_twin_pmin_mean = epoch_twin_pmin_sum / max(1, epoch_twin_n)
        epoch_fd_l1_mean = epoch_fd_l1_sum / max(1, epoch_fd_n)
        epoch_fd_end_mean = epoch_fd_end_sum / max(1, epoch_fd_n)
        _fd_train_text = (
            f"FAST_token_CE={epoch_fd_l1_mean:.4f} FAST_token_acc={epoch_fd_end_mean:.4f}"
            if fast_dec_is_v2 else
            f"L_fd_l1={epoch_fd_l1_mean:.4f} L_fd_end={epoch_fd_end_mean:.4f}"
        )
        log(f"[epoch {epoch}] train_avg_loss={avg_loss:.4f}")
        log(
            f"[epoch {epoch}] train_branch: "
            f"L_stage1_intervene={epoch_s1_mean:.4f} (n={epoch_s1_n})  "
            f"L_round2_ce={epoch_r2ce_mean:.4f} (n={epoch_r2ce_n})  "
            f"L_end={epoch_lend_mean:.4f} (n={epoch_geo_end_n}, w={geo_end_w:.3f})  "
            f"L_tail={epoch_ltail_mean:.4f} (n={epoch_geo_tail_n}, w={geo_tail_w:.3f})  "
            f"{_fd_train_text} (n={epoch_fd_n})  "
            f"L_twin={epoch_twin_mean:.4f} (n={epoch_twin_n}, miss={epoch_twin_miss})  "
            f"td={epoch_twin_td_mean:.4f} prog={epoch_twin_prog_mean:.4f} mc={epoch_twin_mc_mean:.4f}  "
            f"vf_pred1={epoch_twin_p1_mean:.4f} vf_pred2={epoch_twin_p2_mean:.4f} vf_pred_min={epoch_twin_pmin_mean:.4f}"
        )
        if use_wandb:
            _train_wb = {
                    "train/epoch_loss": avg_loss,
                    "epoch": epoch,
                    "train/L_stage1_intervene_epoch": epoch_s1_mean,
                    "train/L_round2_ce_epoch": epoch_r2ce_mean,
                    "train/round2_geo_end_mean": epoch_lend_mean,
                    "train/round2_geo_tail_mean": epoch_ltail_mean,
                    "train/L_end_epoch": epoch_lend_mean,
                    "train/L_tail_epoch": epoch_ltail_mean,
                    "train/L_stage1_intervene_n": epoch_s1_n,
                    "train/L_round2_ce_n": epoch_r2ce_n,
                    "train/round2_geo_end_n": epoch_geo_end_n,
                    "train/round2_geo_tail_n": epoch_geo_tail_n,
                    "train/fd_n_epoch": epoch_fd_n,
                    "train/loss_twin_total_epoch": epoch_twin_mean,
                    "train/loss_td_epoch": epoch_twin_td_mean,
                    "train/loss_prog_epoch": epoch_twin_prog_mean,
                    "train/loss_mc_epoch": epoch_twin_mc_mean,
                    "train/vf_pred1_mean_epoch": epoch_twin_p1_mean,
                    "train/vf_pred2_mean_epoch": epoch_twin_p2_mean,
                    "train/vf_pred_min_mean_epoch": epoch_twin_pmin_mean,
                    "train/twin_n_epoch": epoch_twin_n,
                    "train/twin_missing_labels_epoch": epoch_twin_miss,
                }
            if fast_dec_is_v2:
                _train_wb["train/fast_token_ce"] = epoch_fd_l1_mean
                _train_wb["train/fast_token_accuracy"] = epoch_fd_end_mean
            else:
                _train_wb["train/L_fd_l1_epoch"] = epoch_fd_l1_mean
                _train_wb["train/L_fd_end_epoch"] = epoch_fd_end_mean
            wandb.log(_train_wb)

        if not getattr(args, "skip_val", False):
            model.eval()
            if twin_heads is not None:
                twin_heads.eval()
            if fast_dec is not None:
                fast_dec.eval()
            val_loss, val_n = 0.0, 0
            val_loss_r2, val_n_r2 = 0.0, 0
            val_s1_sum, val_s1_n = 0.0, 0
            val_r2ce_sum, val_r2ce_n = 0.0, 0
            val_geo_end_sum, val_geo_end_n = 0.0, 0
            val_geo_tail_sum, val_geo_tail_n = 0.0, 0
            val_twin_total_sum, val_twin_n = 0.0, 0
            val_twin_td_sum, val_twin_prog_sum, val_twin_mc_sum = 0.0, 0.0, 0.0
            val_twin_p1_sum, val_twin_p2_sum, val_twin_pmin_sum = 0.0, 0.0, 0.0
            val_twin_miss = 0
            val_fd_l1_sum, val_fd_end_sum, val_fd_n = 0.0, 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    epoch_retrieval_rows.extend(_collect_round2_retrieval_rows(batch, epoch=epoch, split="val"))

                    if stage1_ce_enabled:
                        _, _s1_sum, _s1_n = _minibatch_stage1_forward(
                            batch, model, processor, device, use_cuda_bf16_amp,
                            stage1_ce_enabled,
                            use_indicator=getattr(args, "use_indicator", False),
                            batch_total=len(batch),
                            mini_bs=128,
                        )
                        if _s1_n > 0:
                            val_loss += _s1_sum
                            val_n += _s1_n
                            val_s1_sum += _s1_sum
                            val_s1_n += _s1_n

                    if fast_dec is not None:
                        round2_samples_v = [s for s in batch if s.get("has_round2")] if stage1_ce_enabled else batch
                        if round2_samples_v:
                            if fast_dec_is_v2 and fast_dec_tokenizer is not None:
                                _args_no_smooth = type("_A", (), {"__getattr__": lambda s, k: (
                                    0.0 if k == "fast_decoder_v2_label_smoothing" else getattr(args, k)
                                )})()
                                _, _fd_ce, _fd_acc, _fd_n = _minibatch_round2_forward_v2(
                                    round2_samples_v, model, fast_dec, fast_dec_tokenizer,
                                    processor, device, use_cuda_bf16_amp,
                                    _args_no_smooth, train_ds, batch_total=len(batch),
                                    mini_bs=128,
                                )
                                val_fd_l1_sum += _fd_ce
                                val_fd_end_sum += _fd_acc
                                val_fd_n += _fd_n
                            else:
                                _, _fd_l1, _fd_end, _fd_n = _minibatch_round2_forward_v1(
                                    round2_samples_v, model, fast_dec,
                                    processor, device, use_cuda_bf16_amp,
                                    args, train_ds, batch_total=len(batch),
                                    mini_bs=128,
                                )
                                val_fd_l1_sum += _fd_l1
                                val_fd_end_sum += _fd_end
                                val_fd_n += _fd_n

                    if twin_heads is not None:
                        for sample in batch:
                            tgt = _twin_targets_from_sample(
                                sample=sample,
                                transitions=val_ds.transitions,
                                ep_bounds=ep_bounds_val,
                                device=device,
                                gamma=float(getattr(args, "twin_gamma", 0.99)),
                            )
                            prompt_inputs_v = build_prompt_only_input(sample, processor, SYSTEM_MSG, device)
                            with _cuda_bf16_autocast(device, use_cuda_bf16_amp):
                                twin_outputs_v = model(**prompt_inputs_v, output_hidden_states=True)
                            pooled_v = _pool_hidden_from_outputs_with_mask(
                                twin_outputs_v,
                                prompt_inputs_v.get("attention_mask"),
                            )
                            if tgt is None or pooled_v is None:
                                val_twin_miss += 1
                            else:
                                pred1, pred2 = twin_heads(pooled_v.float())
                                next_pred_min = torch.minimum(pred1.detach(), pred2.detach())
                                twin_loss_raw, twin_stats = _compute_twin_loss(pred1, pred2, next_pred_min, tgt, args)
                                twin_loss = float(args.twin_loss_weight) * twin_loss_raw
                                val_twin_total_sum += float(twin_loss.item())
                                val_twin_td_sum += twin_stats["loss_td"]
                                val_twin_prog_sum += twin_stats["loss_prog"]
                                val_twin_mc_sum += twin_stats["loss_mc"]
                                val_twin_p1_sum += twin_stats["pred1"]
                                val_twin_p2_sum += twin_stats["pred2"]
                                val_twin_pmin_sum += twin_stats["pred_min"]
                                val_twin_n += 1

            model.train()
            if twin_heads is not None:
                twin_heads.train()
            if fast_dec is not None:
                fast_dec.train()
            if fast_dec_is_v2 and val_fd_n > 0:
                _val_fast_ce = val_fd_l1_sum / val_fd_n
                _val_fast_acc = val_fd_end_sum / val_fd_n
                log(
                    f"[epoch {epoch}] FAST val: token_CE={_val_fast_ce:.4f} "
                    f"token_acc={_val_fast_acc:.4f} (n={val_fd_n})"
                )
                if use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "val/fast_token_ce": _val_fast_ce,
                        "val/fast_token_accuracy": _val_fast_acc,
                        "val/fast_n": val_fd_n,
                    })
            if val_n > 0:
                val_avg = val_loss / val_n
                extra_r2 = ""
                if val_n_r2 > 0:
                    extra_r2 = (
                        f"  val_avg_loss_r2={val_loss_r2 / val_n_r2:.4f}  "
                        f"(val_n_r2={val_n_r2})"
                    )
                log(
                    f"[epoch {epoch}] val_avg_loss={val_avg:.4f}  "
                    f"(val_n={val_n}){extra_r2}"
                )
                log(
                    f"[epoch {epoch}] geo_val: "
                    f"mean_L_end={(val_geo_end_sum / max(1, val_geo_end_n)):.4f} (n={val_geo_end_n})  "
                    f"mean_L_tail={(val_geo_tail_sum / max(1, val_geo_tail_n)):.4f} (n={val_geo_tail_n})"
                )
                log(
                    f"[epoch {epoch}] val_branch: "
                    f"L_stage1_intervene={(val_s1_sum / max(1, val_s1_n)):.4f} (n={val_s1_n})  "
                    f"L_round2_ce={(val_r2ce_sum / max(1, val_r2ce_n)):.4f} (n={val_r2ce_n})  "
                    f"L_end={(val_geo_end_sum / max(1, val_geo_end_n)):.4f} (n={val_geo_end_n})  "
                    f"L_tail={(val_geo_tail_sum / max(1, val_geo_tail_n)):.4f} (n={val_geo_tail_n})  "
                    f"L_fd_l1={(val_fd_l1_sum / max(1, val_fd_n)):.4f} L_fd_end={(val_fd_end_sum / max(1, val_fd_n)):.4f} (n={val_fd_n})  "
                    f"L_twin={(val_twin_total_sum / max(1, val_twin_n)):.4f} (n={val_twin_n}, miss={val_twin_miss})  "
                    f"td={(val_twin_td_sum / max(1, val_twin_n)):.4f} prog={(val_twin_prog_sum / max(1, val_twin_n)):.4f} mc={(val_twin_mc_sum / max(1, val_twin_n)):.4f}  "
                    f"vf_min={(val_twin_pmin_sum / max(1, val_twin_n)):.4f}"
                )
                if use_wandb:
                    wb = {
                        "val/loss": val_avg,
                        "epoch": epoch,
                        "val_n": val_n,
                    }
                    if val_n_r2 > 0:
                        wb["val/loss_r2"] = val_loss_r2 / val_n_r2
                        wb["val_n_r2"] = val_n_r2
                    wb["val/round2_geo_end_mean"] = val_geo_end_sum / max(1, val_geo_end_n)
                    wb["val/round2_geo_tail_mean"] = val_geo_tail_sum / max(1, val_geo_tail_n)
                    wb["val/round2_geo_end_n"] = val_geo_end_n
                    wb["val/round2_geo_tail_n"] = val_geo_tail_n
                    wb["val/L_stage1_intervene"] = val_s1_sum / max(1, val_s1_n)
                    wb["val/L_round2_ce"] = val_r2ce_sum / max(1, val_r2ce_n)
                    wb["val/L_end"] = val_geo_end_sum / max(1, val_geo_end_n)
                    wb["val/L_tail"] = val_geo_tail_sum / max(1, val_geo_tail_n)
                    wb["val/L_fd_l1"] = val_fd_l1_sum / max(1, val_fd_n)
                    wb["val/L_fd_end"] = val_fd_end_sum / max(1, val_fd_n)
                    wb["val/fd_n"] = val_fd_n
                    wb["val/loss_twin_total"] = val_twin_total_sum / max(1, val_twin_n)
                    wb["val/loss_td"] = val_twin_td_sum / max(1, val_twin_n)
                    wb["val/loss_prog"] = val_twin_prog_sum / max(1, val_twin_n)
                    wb["val/loss_mc"] = val_twin_mc_sum / max(1, val_twin_n)
                    wb["val/vf_pred1_mean"] = val_twin_p1_sum / max(1, val_twin_n)
                    wb["val/vf_pred2_mean"] = val_twin_p2_sum / max(1, val_twin_n)
                    wb["val/vf_pred_min_mean"] = val_twin_pmin_sum / max(1, val_twin_n)
                    wb["val/twin_n"] = val_twin_n
                    wb["val/twin_missing_labels"] = val_twin_miss
                    wandb.log(wb)
        else:
            log(f"[epoch {epoch}] skip_val=True, skipping validation")

        _write_round2_retrieval_audit(
            epoch_retrieval_rows,
            os.path.join(retrieval_audit_dir, f"epoch_{epoch:03d}.csv"),
        )
        save_every_epochs = int(getattr(args, "save_every_epochs", 0))
        if save_every_epochs <= 0 or epoch % save_every_epochs == 0 or epoch == int(args.epochs) or stop_training:
            ckpt_dir = os.path.join(args.output_dir, f"model_epoch{epoch}")
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            if twin_heads is not None:
                torch.save(twin_heads.state_dict(), os.path.join(ckpt_dir, "twin_heads.pt"))
            if fast_dec is not None:
                torch.save(fast_dec.state_dict(), os.path.join(ckpt_dir, "fast_decoder.pt"))
                if fast_dec_is_v2 and fast_dec_tokenizer is not None:
                    import json as _json
                    with open(os.path.join(ckpt_dir, "fast_tokenizer.json"), "w") as _f:
                        _json.dump(fast_dec_tokenizer.state_dict(), _f)
            log(f"[saved] {ckpt_dir}")

        if stop_training:
            break

    log("[train] done")
    if use_wandb:
        wandb.finish()


def _apply_train_profile(args):
    profile = str(getattr(args, "train_profile", "default"))
    if profile == "epoch7_equiv":
        args.round2_tail_steps_weight = 0.0
        args.round2_goal_end_weight = 0.0
        args.round2_length_floor_ratio = 0.0
        args.round2_length_penalty_weight = 0.0
    if bool(getattr(args, "round2_ce_only", False)):
        args.use_twin_aux = False
        args.round2_last_token_weight = 0.0
        args.round2_tail_steps_weight = 0.0
        args.round2_goal_end_weight = 0.0
        args.round2_length_penalty_weight = 0.0
        args.round2_geo_end_weight = 0.0
        args.round2_geo_tail_weight = 0.0
        args.round2_quality_weight = 0.0
        args.round2_quality_cap = 1.0
        args.round2_loss_weight = 1.0
    if bool(getattr(args, "stage1_only", False)):
        args.round2_loss_weight = 0.0
        args.round2_last_token_weight = 0.0
        args.round2_tail_steps_weight = 0.0
        args.round2_goal_end_weight = 0.0
        args.round2_length_penalty_weight = 0.0
        args.round2_geo_end_weight = 0.0
        args.round2_geo_tail_weight = 0.0
        args.round2_quality_weight = 0.0
        args.round2_quality_cap = 1.0
        args.round2_smoothness_weight = 0.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train_profile",
        type=str,
        default="default",
        choices=["default", "epoch7_equiv"],
        help="Training profile.",
    )
    p.add_argument(
        "--round2_ce_only",
        action="store_true",
        help="Round2 uses only token CE (log-prob/NLL) supervision; disables the tail/goal/length/geo/quality auxiliary terms",
    )
    p.add_argument(
        "--stage1_only",
        action="store_true",
        help="Train only Round1 Intervention; disables Round2 data and all Round2 losses.",
    )
    p.add_argument(
        "--disable_stage1_ce",
        action="store_true",
        help="Disable stage1 text CE in VQ2/VLA training and keep only round2/FAST supervision.",
    )
    p.add_argument("--buffer_dir", default="buffer_data/buffer_nttg")
    p.add_argument(
        "--instr",
        type=str,
        default=None,
        help="Override the HILVLMDataset task instruction (should match the instruction used at eval time).",
    )
    p.add_argument(
        "--stage1_include_reward_label",
        action="store_true",
        help="Keep RewardLabel in the stage1 target (default off; train Intervention only)",
    )
    p.add_argument(
        "--round2_bridge_steps",
        type=int,
        default=0,
        help="Number of synthetic bridge actions prepended to a recovery target.",
    )
    p.add_argument(
        "--round2_retrieval_source",
        type=str,
        default="memory_bank",
        choices=["memory_bank"],
        help="Round2 training retrieval source: only memory_bank is supported (strict mode).",
    )
    p.add_argument(
        "--memory_bank_path",
        type=str,
        default="",
        help="Default memory bank for Round2 training retrieval (fallback on route miss; usually combined with --memory_bank_router_json)",
    )
    p.add_argument(
        "--memory_bank_router_json",
        type=str,
        default="",
        help="Round2 multi-task memory bank routing JSON: task_key -> bank_path",
    )
    p.add_argument(
        "--round2_task_route_mode",
        type=str,
        default="strict",
        choices=["strict", "fallback"],
        help="Multi-task bank routing mode: strict raises on a miss; fallback uses --memory_bank_path",
    )
    p.add_argument(
        "--intervention_label_mode",
        type=str,
        default="recover_mask",
        choices=["recover_mask", "decline_only"],
        help="Intervention label mode.",
    )
    p.add_argument("--decline_window", type=int, default=8, help="Decline window for decline_only / recover mining")
    p.add_argument("--min_down_ratio", type=float, default=0.7, help="Minimum down-step ratio for decline_only / recover mining")
    p.add_argument("--min_drop", type=float, default=0.05, help="Minimum drop for decline_only / recover mining")
    p.add_argument("--intervention_margin", type=int, default=2, help="Intervention neighborhood margin for recover mining")
    p.add_argument("--intervention_expand_steps", type=int, default=0, help="Number of steps to expand each positive Intervention label on both sides")
    p.add_argument(
        "--memory_top_k",
        type=int,
        default=5,
        help="Number of memory candidates retrieved for Round2.",
    )
    p.add_argument(
        "--round2_target_space",
        type=str,
        default="state11_pose",
        choices=["state11_pose", "state", "action"],
        help="Round2 target representation.",
    )
    p.add_argument(
        "--state_dim",
        type=int,
        default=11,
        help="Number of supervised state dimensions (default 11, the full robot state).",
    )
    p.add_argument(
        "--state_clip_k",
        type=float,
        default=4.0,
        help="Clip threshold k ([-k,k]) applied to state z-scores before quantization.",
    )
    p.add_argument(
        "--memory_goal_aggregate",
        type=str,
        default="top1",
        choices=["top1", "topk_mean"],
        help="Memory-goal aggregation mode.",
    )
    p.add_argument("--train_ratio", type=float, default=0.8, help="Dataset train split ratio (stratified by task).")
    p.add_argument("--val_ratio", type=float, default=0.1, help="Dataset val split ratio (stratified by task).")
    p.add_argument("--test_ratio", type=float, default=0.1, help="Dataset test split ratio (stratified by task).")
    p.add_argument("--split_seed", type=int, default=42, help="Deterministic seed for stratified task split.")
    p.add_argument(
        "--use_external_wrist_keys",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use observations keys external/wrist instead of side_policy_256/wrist_1.",
    )
    p.add_argument(
        "--intervention_label_source",
        type=str,
        default="auto",
        choices=["auto", "offline", "recover_mask"],
        help="Intervention label source: auto prefers offline mining label key if present, offline enforces it, recover_mask forces online mining.",
    )
    p.add_argument(
        "--intervention_label_key",
        type=str,
        default="vf_mining_intervention_label",
        help="Offline intervention label field name in transitions, used when intervention_label_source=auto/offline.",
    )
    p.add_argument(
        "--task_aware_instr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use per-sample task text (task_name/task_text) as prompt instruction when available.",
    )
    p.add_argument(
        "--round2_task_aware_filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter Round2 retrieval candidates by same task tag (strict: raise if empty).",
    )
    p.add_argument(
        "--round2_sample_boost",
        type=float,
        default=3.0,
        help="Sampling multiplier for intervention/round2 samples (>1 increases goal supervision density)",
    )
    p.add_argument(
        "--model_path",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    p.add_argument("--output_dir", default="runs/uniintervene")
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory (Round2 runs two forwards per step)",
    )
    p.add_argument(
        "--hidden_state_capture",
        type=str,
        default="hook",
        choices=["hook", "output_hidden_states"],
        help="VQ1 hidden-state capture mode.",
    )
    p.add_argument(
        "--adam8bit",
        action="store_true",
        help="Use bitsandbytes AdamW8bit to reduce optimizer state memory",
    )
    p.add_argument(
        "--use_lora",
        action="store_true",
        help="Train LoRA adapters only (PEFT) instead of full fine-tuning",
    )
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument(
        "--resume_lora_path",
        type=str,
        default="",
        help="Directory of an existing LoRA adapter to resume from (with adapter_config.json / adapter_model.safetensors)",
    )
    p.add_argument(
        "--resume_vq1_heads_path",
        type=str,
        default="",
        help="Directory of VQ1 heads to resume from (with vq1_heads.pt); usually the same vq1_epochN directory as --resume_lora_path.",
    )
    p.add_argument(
        "--vq2_adapter_mode",
        type=str,
        default="base_lora",
        choices=["base_lora", "init_from_vq1", "shared_residual"],
        help="VQ2 backbone parameterization: base_lora (default), init_from_vq1, or "
             "shared_residual (frozen VQ1 LoRA + trainable VQ2 residual LoRA).",
    )
    p.add_argument(
        "--vq1_shared_lora_path",
        type=str,
        default="",
        help="Path to a VQ1 LoRA adapter dir to share into VQ2 (required for init_from_vq1 / shared_residual).",
    )
    p.add_argument(
        "--freeze_vq1_shared_lora_for_vq2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When True (default), VQ1 shared adapter is frozen during VQ2 training (only effective for shared_residual).",
    )
    p.add_argument("--vq2_residual_lora_r", type=int, default=16,
                   help="Rank of the VQ2 residual LoRA (shared_residual mode).")
    p.add_argument("--vq2_residual_lora_alpha", type=int, default=32,
                   help="Alpha of the VQ2 residual LoRA.")
    p.add_argument("--vq2_residual_lora_dropout", type=float, default=0.05,
                   help="Dropout of the VQ2 residual LoRA.")
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the backbone in 4-bit mode.",
    )
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--log_interval",      type=int, default=50)
    p.add_argument("--save_every_steps",  type=int, default=500,
                   help="Save model_latest every N steps; 0 disables it")
    p.add_argument(
        "--save_every_epochs",
        type=int,
        default=0,
        help="Save a model_epochN checkpoint every N epochs; 0 saves every epoch.",
    )
    p.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help=">0 stops training once the global optimizer step reaches this value (smoke tests)",
    )
    p.add_argument(
        "--skip_val",
        action="store_true",
        help="Skip per-epoch validation",
    )
    p.add_argument(
        "--round2_loss_weight",
        type=float,
        default=1.4,
        help="Global Round2 loss weight (>1 strengthens goal-conditioned trajectory supervision)",
    )
    p.add_argument(
        "--round2_last_tokens",
        type=int,
        default=28,
        help="Round2 auxiliary endpoint supervision: number of trailing supervised tokens (0 disables)",
    )
    p.add_argument(
        "--round2_last_token_weight",
        type=float,
        default=0.7,
        help="Weight of the Round2 trailing-token auxiliary CE",
    )
    p.add_argument(
        "--round2_quality_weight",
        type=float,
        default=0.8,
        help="Coefficient scaling the Round2 loss by round2_quality_score",
    )
    p.add_argument(
        "--round2_quality_cap",
        type=float,
        default=2.5,
        help="Upper bound of the round2 quality weight, so extreme samples do not dominate",
    )
    p.add_argument(
        "--round2_tail_steps_weight",
        type=float,
        default=0.8,
        help="Round2 tail emphasis: weight on the main CE of tail tokens (tail_steps_k mapped to tokens)",
    )
    p.add_argument(
        "--round2_tail_steps_k",
        type=int,
        default=3,
        help="Number of Round2 tail supervision steps (k)",
    )
    p.add_argument(
        "--round2_goal_end_weight",
        type=float,
        default=0.5,
        help="Weight of the Round2 final-step goal alignment loss (final-step CE scaled by the action-space alignment coefficient)",
    )
    p.add_argument(
        "--round2_length_floor_ratio",
        type=float,
        default=0.8,
        help="Round2 minimum length ratio relative to GT steps; suppresses early EOS",
    )
    p.add_argument(
        "--round2_length_penalty_weight",
        type=float,
        default=0.3,
        help="Round2 length penalty weight (early-EOS penalty)",
    )
    p.add_argument(
        "--round2_geo_end_weight",
        type=float,
        default=0.15,
        help="Weight of the Round2 final-step geometric loss (L_end)",
    )
    p.add_argument(
        "--round2_geo_tail_weight",
        type=float,
        default=0.05,
        help="Weight of the Round2 monotone tail convergence loss (L_tail)",
    )
    p.add_argument(
        "--round2_smoothness_weight",
        type=float,
        default=0.0,
        help="Weight of the Round2 smoothness term (mean ||a_{t+1}-a_t||); small values such as 0.01-0.05",
    )
    p.add_argument(
        "--round2_geo_tail_k",
        type=int,
        default=3,
        help="Number of Round2 tail convergence steps (k)",
    )
    p.add_argument(
        "--round2_geo_margin",
        type=float,
        default=0.0,
        help="Margin of the Round2 monotone tail convergence (max(0, d_t+1-d_t+margin))",
    )
    p.add_argument(
        "--round2_geo_vgain_tau",
        type=float,
        default=0.05,
        help="Geometric gate threshold; used when gate_mode is goal_vf_delta or round2_improvement",
    )
    p.add_argument(
        "--round2_geo_gate_mode",
        type=str,
        default="goal_vf_delta",
        choices=["goal_vf_delta", "round2_improvement", "goal_vf_delta_quantile", "none"],
        help="Geometric gate mode; default uses goal_vf-current_vf (same as the retrieval filter)",
    )
    p.add_argument(
        "--round2_geo_gate_quantile",
        type=float,
        default=0.3,
        help="Quantile q for gate_mode=goal_vf_delta_quantile (tau=quantile(delta_v,q))",
    )
    p.add_argument(
        "--round2_geo_warmup_epochs",
        type=int,
        default=0,
        help="Disable geometric terms for the first warmup epochs, use a reduced weight at warmup+1, then the target weight. Default 0 activates them from epoch 1",
    )
    p.add_argument(
        "--no_cuda_bf16_autocast",
        action="store_true",
        help="Disable bfloat16 autocast for the CUDA forward (enabled by default)",
    )
    p.add_argument(
        "--use_fast_decoder",
        action="store_true",
        help="Enable the FAST+ decoder head: autoregressively decode continuous action trajectories from the Qwen pooled hidden state",
    )
    p.add_argument("--fast_decoder_action_dim", type=int, default=7,
                   help="FAST decoder output action dimension (default 7)")
    p.add_argument("--fast_decoder_d_dec", type=int, default=256,
                   help="FAST decoder hidden width (default 256)")
    p.add_argument("--fast_decoder_n_layers", type=int, default=2,
                   help="Number of FAST decoder causal Transformer layers (default 2)")
    p.add_argument("--fast_decoder_n_heads", type=int, default=4,
                   help="FAST decoder attention heads (default 4)")
    p.add_argument("--fast_decoder_max_traj_len", type=int, default=128,
                   help="FAST decoder maximum trajectory length (default 128)")
    p.add_argument("--fast_decoder_loss_weight", type=float, default=1.0,
                   help="FAST decoder total loss weight (default 1.0)")
    p.add_argument("--fast_decoder_l1_weight", type=float, default=1.0,
                   help="FAST decoder main L1 loss weight")
    p.add_argument("--fast_decoder_end_weight", type=float, default=0.15,
                   help="FAST decoder final-step alignment loss weight (L_fd_end)")
    p.add_argument("--fast_decoder_tail_weight", type=float, default=0.05,
                   help="FAST decoder monotone tail convergence loss weight (L_fd_tail)")
    p.add_argument("--fast_decoder_tail_k", type=int, default=3,
                   help="FAST decoder tail steps k")
    p.add_argument("--fast_decoder_margin", type=float, default=0.0,
                   help="FAST decoder tail hinge margin")
    p.add_argument("--resume_fast_decoder_path", type=str, default="",
                   help="Checkpoint directory to resume the FAST decoder from (reads fast_decoder.pt)")
    p.add_argument(
        "--use_fast_decoder_v2",
        action="store_true",
        help="Enable the official FAST processor (DCT + quantization + BPE) with a per-token CE decoder.",
    )
    p.add_argument("--fast_tokenizer_path", type=str, default="",
                   help="Local directory containing the reviewed FAST tokenizer assets.")
    p.add_argument("--fast_decoder_v2_max_tokens", type=int, default=128,
                   help="Maximum number of FAST BPE tokens per H-step action chunk (including EOS).")
    p.add_argument("--fast_decoder_v2_max_T", type=int, default=8,
                   help="FAST action horizon (default: 8).")
    p.add_argument("--fast_decoder_v2_n_layers", type=int, default=4,
                   help="Number of FAST+ V2 transformer layers (default 4)")
    p.add_argument("--fast_decoder_v2_n_heads", type=int, default=8,
                   help="FAST+ V2 attention heads (default 8)")
    p.add_argument("--fast_decoder_v2_d_dec", type=int, default=256,
                   help="FAST+ V2 transformer width (default 256)")
    p.add_argument("--fast_decoder_v2_label_smoothing", type=float, default=0.0,
                   help="FAST per-token CE label smoothing (default 0, plain NLL).")
    p.add_argument(
        "--use_twin_aux",
        action="store_true",
        help="Enable Unified+Twin auxiliary training (shared trunk + twin VF heads optimized jointly)",
    )
    p.add_argument("--twin_loss_weight", type=float, default=1.0, help="L_total += twin_loss_weight * L_twin")
    p.add_argument("--twin_td_weight", type=float, default=1.0, help="Weight of the TD term in L_twin")
    p.add_argument("--twin_prog_weight", type=float, default=1.0, help="Weight of the progress term in L_twin")
    p.add_argument("--twin_mc_weight", type=float, default=0.0, help="Weight of the MC term in L_twin (disabled by default)")
    p.add_argument("--twin_gamma", type=float, default=0.99, help="Gamma used to build twin TD rewards/targets")
    p.add_argument(
        "--resume_twin_head_path",
        type=str,
        default="",
        help="Directory to resume twin heads from (reads twin_heads.pt)",
    )
    p.add_argument(
        "--stage1_yes_boost",
        type=float,
        default=1.0,
        help="Sampling multiplier for Round1 Intervention=yes samples; >1 increases their share",
    )
    p.add_argument(
        "--use_indicator",
        action="store_true",
        help="Enable VF indicator sampling weights (good 2.0 / bad 0.5); "
             "Intervention labels still depend only on human intervened",
    )
    p.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging (requires WANDB_API_KEY)",
    )
    p.add_argument(
        "--wandb_project",
        type=str,
        default="uniintervene",
        help="wandb project name",
    )
    p.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="wandb entity / team name",
    )
    p.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="wandb run name (default includes a timestamp)",
    )

    p.add_argument(
        "--variant",
        type=str,
        default="vq2_fast",
        help=(
            "Architecture variant. 'vq2_fast' selects goal-conditioned FAST recovery. "
            "Use 'full' / 'no_future' / 'no_value_cond' "
            "/ 'no_future_value_cond' / 'no_intervention' / 'no_future_loss' "
            "to enable the unified VQ1 path."
        ),
    )
    p.add_argument(
        "--future_latent_mode",
        type=str,
        default="pooled",
        choices=["pooled", "dense"],
        help="V-JEPA target shape: pooled [B,D] or dense [B,N,D].",
    )
    p.add_argument(
        "--vjepa_path",
        type=str,
        default="",
        help="HF-style path or hub id of the frozen V-JEPA2 encoder for online mode.",
    )
    p.add_argument(
        "--future_target_source",
        type=str,
        default="online_vjepa2",
        choices=["online_vjepa2", "precomputed_jepa"],
        help="Backend for V-JEPA target. online_vjepa2 encodes next_observations on the fly; "
             "precomputed_jepa reads latent_<episode_stem>.npz from --jepa_latent_dir.",
    )
    p.add_argument(
        "--jepa_latent_dir",
        type=str,
        default="",
        help="Directory of precomputed latent_<episode_stem>.npz files (precomputed_jepa source).",
    )
    p.add_argument(
        "--jepa_latent_manifest",
        type=str,
        default="",
        help="Optional manifest.json mapping episode_file -> latent_file.",
    )
    p.add_argument(
        "--vq1_debug_samples",
        type=int,
        default=8,
        help="Number of first samples to print alignment debug info for in VQ1 path.",
    )
    p.add_argument(
        "--require_future_loss",
        action="store_true",
        help="Fail the run if V-JEPA target is unavailable for the 'full' variant.",
    )
    p.add_argument(
        "--required_vf_dir",
        type=str,
        default="",
        help="If non-empty, override the dataset's required vf_dir guard. Empty disables the guard.",
    )
    p.add_argument(
        "--vjepa_feature_dim",
        type=int,
        default=1024,
        help="Fallback dimension if V-JEPA shape can't be probed.",
    )
    p.add_argument(
        "--vjepa_n_tokens",
        type=int,
        default=256,
        help="Fallback token count for dense mode.",
    )
    p.add_argument("--lambda_future", type=float, default=1.0)
    p.add_argument("--lambda_q", type=float, default=1.0)
    p.add_argument("--lambda_intervention", type=float, default=1.0)
    p.add_argument(
        "--stage_schedule",
        action="store_true",
        help="Enable the 3-stage lambda schedule (warm-up / joint / calibration).",
    )
    p.add_argument("--stage1_epochs", type=int, default=1)
    p.add_argument("--stage2_epochs", type=int, default=3)
    p.add_argument(
        "--head_lr",
        type=float,
        default=1e-4,
        help="Learning rate for future_head / twin_Q / intervention_head; "
             "backbone+LoRA still use --lr.",
    )
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA target modules. Single shared adapter.",
    )
    p.add_argument(
        "--lora_include_ffn",
        action="store_true",
        help="Also LoRA-tune gate_proj/up_proj/down_proj.",
    )
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--intervention_pos_weight",
        type=float,
        default=0.0,
        help="pos_weight for BCEWithLogitsLoss; 0 = auto from training set imbalance.",
    )
    p.add_argument(
        "--intervention_threshold",
        type=float,
        default=0.5,
        help="Decision threshold for precision/recall/F1 metrics and inference.",
    )
    p.add_argument(
        "--intervention_loss_type",
        type=str,
        default="weighted_bce",
        choices=["weighted_bce", "focal"],
        help="VQ1 intervention loss: weighted BCE (default) or binary focal.",
    )
    p.add_argument("--focal_alpha", type=float, default=0.75,
                   help="Focal-loss alpha (positive-class weight). Used when "
                        "--intervention_loss_type=focal.")
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="Focal-loss gamma (modulation exponent).")

    p.add_argument("--proxy_vf_source_name", type=str, default="",
                   help="Override: human-readable name of the proxy VF (e.g. "
                        "'stitched_siglip_gemma_vf' or 'latent_gemma_vjepa2_vitl_vf').")
    p.add_argument("--proxy_vf_checkpoint_path", type=str, default="",
                   help="Override: path to the VF checkpoint that produced vf_value_pred.")
    p.add_argument("--proxy_vf_training_script", type=str, default="",
                   help="Override: training script that produced the VF checkpoint.")
    p.add_argument("--proxy_vf_score_buffer_dir", type=str, default="",
                   help="Override: the *input* buffer dir scored to produce the current pkls.")
    p.add_argument("--proxy_vf_score_manifest_path", type=str, default="",
                   help="Override: path to vf_value_pred_manifest.json.")
    p.add_argument("--proxy_vf_is_latent_based", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Override: whether the proxy VF is latent-based (V-JEPA z) "
                        "or non-latent (e.g. SigLIP+Gemma stitched). If unset, the "
                        "trainer auto-detects from the VF run's config.json training_mode.")
    p.add_argument("--strict_proxy_vf", action="store_true",
                   help="Fail the run if proxy VF provenance cannot be verified.")
    p.add_argument("--proxy_vf_latent_dir", type=str, default="",
                   help="Latent npz directory for is_latent_based proxy VF. "
                        "Defaults to --jepa_latent_dir if unset.")
    p.add_argument("--proxy_vf_latent_manifest", type=str, default="",
                   help="Manifest JSON for latent proxy VF. "
                        "Defaults to --jepa_latent_manifest if unset.")
    p.add_argument("--proxy_vf_language_model", type=str, default="",
                   help="Override language model path for the latent proxy VF "
                        "(default: read from VF run config.json).")
    p.add_argument("--proxy_vf_prompt", type=str, default="",
                   help="Override prompt template for the latent proxy VF "
                        "(default: read from VF run config.json).")

    p.add_argument(
        "--intervention_head_type",
        type=str,
        default="binary",
        choices=["binary", "current_value", "tvr_history"],
        help=(
            "Intervention head formulation: "
            "'binary' (existing BCE/focal on intervention label), "
            "'current_value' (TVR regression conditioned on predicted V_t from h_t), "
            "'tvr_history' (TVR regression conditioned on ValueHistoryEncoder([V_{t-K:t}, ΔV]))."
        ),
    )
    p.add_argument(
        "--value_history_window",
        type=int,
        default=8,
        help="K: number of past steps to include in the value history (tvr_history / current_value modes).",
    )
    p.add_argument(
        "--value_history_gamma",
        type=float,
        default=0.9,
        help="Discount factor γ for the TVR target sum Σ γ^i (ε - δ_i).",
    )
    p.add_argument(
        "--value_history_epsilon",
        type=float,
        default=0.005,
        help="ε threshold in TVR target: intervention triggered when δ_i < ε (i.e. value not rising).",
    )
    p.add_argument(
        "--value_history_embed_dim",
        type=int,
        default=32,
        help="Output embedding dimension of ValueHistoryEncoder (tvr_history mode).",
    )
    p.add_argument(
        "--lambda_tvr",
        type=float,
        default=1.0,
        help="Loss weight for L_TVR (TVR regression) in current_value and tvr_history modes.",
    )
    p.add_argument(
        "--lambda_v_curr",
        type=float,
        default=1.0,
        help="Loss weight for L_V_curr (auxiliary current-value prediction) in current_value mode.",
    )

    p.add_argument(
        "--config",
        type=str,
        default="",
        help=(
            "Path to a YAML config file (e.g. configs/vq1/full_tvr_history.yaml). "
            "When set, YAML values are applied as defaults; explicit CLI args override them. "
            "Use --override for highest-priority key=value patches."
        ),
    )
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "One or more 'key=value' overrides applied after the YAML config. "
            "Keys can be YAML dot-paths (e.g. training.batch_size=2) or "
            "direct argparse dest names (e.g. batch_size=2)."
        ),
    )
    return p


def main():
    import argparse as _argparse

    _pre = _argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", type=str, default="")
    _pre.add_argument("--override", nargs="*", default=[])
    _pre_args, _ = _pre.parse_known_args()

    p = build_parser()

    _config_path = _pre_args.config.strip() if _pre_args.config else ""

    if _config_path:
        from config_loader import (
            load_yaml_config,
            parse_overrides,
            apply_to_args,
            validate_config,
            log_resolved_config,
            save_resolved_config,
        )
        _yaml_defaults = load_yaml_config(_config_path)
        print(f"[config] loaded from: {_config_path}", flush=True)
        p.set_defaults(**_yaml_defaults)

    args = p.parse_args()

    if _config_path:
        if _pre_args.override:
            from config_loader import parse_overrides, apply_to_args
            _ovr = parse_overrides(_pre_args.override, p)
            apply_to_args(args, _ovr, source="--override")

        from config_loader import validate_config, log_resolved_config, save_resolved_config
        validate_config(args)
        log_resolved_config(args, _config_path)
        save_resolved_config(args, _config_path)

    variant = str(getattr(args, "variant", "vq2_fast")).strip().lower()
    if variant != "vq2_fast":
        if variant not in VARIANT_SPEC:
            raise SystemExit(
                f"--variant={args.variant} is not recognized. "
                f"Use 'vq2_fast' or one of: {sorted(VARIANT_SPEC.keys())}"
            )
        train_vq1(args)
        return
    _apply_train_profile(args)
    train(args)


if __name__ == "__main__":
    main()
