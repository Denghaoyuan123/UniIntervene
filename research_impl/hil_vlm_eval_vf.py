"""Shared HIL-VLM inference utilities: Qwen3-VL + LoRA loading, Round1/Round2 inputs, memory action de-quantization."""

from __future__ import annotations

import glob
import json
import os
import pickle
import re
from typing import Any, Callable, Optional

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from hil_vlm_train import (
    SYSTEM_MSG,
    SYSTEM_MSG_ROUND2,
    build_stage1_user_text,
)


def _resolve_qwen_vl_model_cls(model_path: str | None = None):
    """Mirror hil_vlm_train's model-class resolution so inference matches the checkpoint."""
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
        except Exception:
            pass

    def _try(name: str):
        try:
            import importlib
            mod = importlib.import_module("transformers")
            return getattr(mod, name, None)
        except Exception:
            return None

    if arch_hint:
        cls = _try(arch_hint)
        if cls is not None:
            return cls
    if model_type_hint == "qwen2_5_vl":
        cls = _try("Qwen2_5_VLForConditionalGeneration")
        if cls is not None:
            return cls
    if model_type_hint == "qwen3_vl":
        cls = _try("Qwen3VLForConditionalGeneration")
        if cls is not None:
            return cls
    for name in ("Qwen3VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration"):
        cls = _try(name)
        if cls is not None:
            return cls
    raise ImportError(
        "Neither Qwen3VLForConditionalGeneration nor Qwen2_5_VLForConditionalGeneration is available."
    )


def build_eval_inputs(sample: dict, processor, device: torch.device):
    """Round1: system + user (images + instruction + current action bins)."""
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
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return processor(
        text=[text],
        images=sample["images"] or None,
        return_tensors="pt",
        padding=True,
    ).to(device)


def build_retrieval_inputs(sample: dict, processor, device: torch.device):
    """Observation-and-instruction input used for recovery-memory retrieval."""
    msgs = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in sample["images"]],
                {"type": "text", "text": f"Task: {sample['instr']}"},
            ],
        },
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return processor(
        text=[text],
        images=sample["images"] or None,
        return_tensors="pt",
        padding=True,
    ).to(device)


def build_eval_inputs_round2(
    images: list,
    user_text: str,
    processor,
    device: torch.device,
    target_space: str = "action",
):
    """Round2: A+B images + pre-built user text (instruction, ROUND2 suffix, single-frame goal state bins)."""
    from hil_vlm_train import SYSTEM_MSG_ROUND2_STATE, SYSTEM_MSG_ROUND2_ACTION
    sys_msg = SYSTEM_MSG_ROUND2_STATE if str(target_space).strip().lower() == "state" else SYSTEM_MSG_ROUND2_ACTION
    msgs = [
        {"role": "system", "content": sys_msg},
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": user_text},
            ],
        },
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return processor(
        text=[text],
        images=images or None,
        return_tensors="pt",
        padding=True,
    ).to(device)


def compute_failure_embedding(model, model_inputs: dict) -> torch.Tensor:
    """Hidden state of the last valid token in the final layer, shape (1, H)."""
    with torch.no_grad():
        out = model(**model_inputs, output_hidden_states=True)
    last_h = out.hidden_states[-1]
    mask = model_inputs["attention_mask"].long()
    seq_lens = mask.size(1) - 1 - torch.flip(mask, dims=[1]).argmax(dim=1)
    b = torch.arange(last_h.size(0), device=last_h.device, dtype=torch.long)
    return last_h[b, seq_lens].float()


def parse_intervention(text: str) -> Optional[str]:
    m = re.search(r"Intervention:\s*(yes|no)\b", text, re.IGNORECASE)
    return m.group(1).lower() if m else None


def parse_reward_label(text: str) -> Optional[int]:
    m = re.search(r"RewardLabel:\s*(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


def _primary_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def model_hidden_size(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    candidates = (
        config,
        getattr(config, "text_config", None),
        getattr(config, "language_config", None),
    )
    for cfg in candidates:
        hidden_size = getattr(cfg, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
    raise AttributeError("model config does not expose hidden_size or text_config.hidden_size")


def _patch_greedy_generation(model: torch.nn.Module) -> None:
    for m in model.modules():
        gc = getattr(m, "generation_config", None)
        if gc is None:
            continue
        gc.do_sample = False
        gc.temperature = 1.0
        if hasattr(gc, "top_p"):
            gc.top_p = 1.0


def _cast_lora_params_fp32(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name.lower() and p.is_floating_point():
                p.data = p.data.float()


def _cast_lora_params_dtype(model: torch.nn.Module, dtype: torch.dtype) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name.lower() and p.is_floating_point():
                p.data = p.data.to(dtype=dtype)


def _read_adapter_config(path: str) -> dict[str, Any] | None:
    cfg_path = os.path.join(path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _resolve_peft_adapter_paths(lora_path: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (base_model_path, [(adapter_name, adapter_dir), ...])."""
    lp = os.path.abspath(str(lora_path))
    nested: list[tuple[str, str]] = []
    for name in ("vq1_shared", "vq2_residual"):
        cand = os.path.join(lp, name)
        if os.path.isfile(os.path.join(cand, "adapter_config.json")):
            nested.append((name, cand))

    if nested:
        cfg = _read_adapter_config(nested[0][1]) or {}
        base_model_path = str(cfg.get("base_model_name_or_path") or "").strip() or lp
        return base_model_path, nested

    cfg = _read_adapter_config(lp) or {}
    base_model_path = str(cfg.get("base_model_name_or_path") or "").strip() or lp
    return base_model_path, [("default", lp)]


def load_model(
    base_model_path: str,
    lora_path: Optional[str],
    device: torch.device,
    merge_lora: bool,
    load_in_4bit: bool,
    load_in_8bit: bool,
    device_map_auto: bool,
    lora_fp32: bool,
):
    if merge_lora and (load_in_4bit or load_in_8bit):
        raise ValueError("merge_lora cannot be combined with load_in_4bit/8bit (do not merge into a quantized base)")
    if load_in_4bit and load_in_8bit:
        raise ValueError("load_in_4bit and load_in_8bit are mutually exclusive")

    adapter_specs: list[tuple[str, str]] = []
    resolved_base_model_path = base_model_path
    if lora_path:
        try:
            resolved_base_model_path, adapter_specs = _resolve_peft_adapter_paths(lora_path)
        except Exception:
            adapter_specs = [("default", os.path.abspath(str(lora_path)))]
            resolved_base_model_path = base_model_path
    processor_path = resolved_base_model_path
    if lora_path and os.path.isfile(os.path.join(str(lora_path), "tokenizer_config.json")):
        processor_path = str(lora_path)
    processor = AutoProcessor.from_pretrained(
        processor_path,
        trust_remote_code=False,
    )
    try:
        _tok = getattr(processor, "tokenizer", None)
        if _tok is not None and (not hasattr(_tok, "tokenizer")) and hasattr(_tok, "_tokenizer"):
            setattr(_tok, "tokenizer", getattr(_tok, "_tokenizer"))
    except Exception:
        pass
    common_kw = dict(trust_remote_code=False)

    if load_in_4bit or load_in_8bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise ImportError("Quantized inference requires bitsandbytes: pip install bitsandbytes") from e
        compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        if load_in_4bit:
            qconf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            qconf = BitsAndBytesConfig(load_in_8bit=True)
        if device.type == "cuda" and not device_map_auto:
            dm: dict | str = {"": 0}
        else:
            dm = "auto"
        model_cls = _resolve_qwen_vl_model_cls(resolved_base_model_path)
        model = model_cls.from_pretrained(
            resolved_base_model_path,
            quantization_config=qconf,
            device_map=dm,
            **common_kw,
        )
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model_cls = _resolve_qwen_vl_model_cls(resolved_base_model_path)
        model = model_cls.from_pretrained(
            resolved_base_model_path,
            torch_dtype=dtype,
            **common_kw,
        )

    tokenizer_size = len(processor.tokenizer)
    embedding_size = int(model.get_input_embeddings().weight.shape[0])
    if tokenizer_size != embedding_size:
        model.resize_token_embeddings(tokenizer_size)

    if lora_path:
        from peft import PeftModel

        if len(adapter_specs) <= 1:
            _, adapter_dir = adapter_specs[0] if adapter_specs else ("default", os.path.abspath(str(lora_path)))
            model = PeftModel.from_pretrained(
                model, adapter_dir, autocast_adapter_dtype=False
            )
            if lora_fp32 and (load_in_4bit or load_in_8bit):
                _cast_lora_params_fp32(model)
            if merge_lora:
                model = model.merge_and_unload()
                model.to(device)
        else:
            if merge_lora:
                raise ValueError("merge_lora does not support loading multiple LoRA adapters")
            first_name, first_dir = adapter_specs[0]
            model = PeftModel.from_pretrained(
                model, first_dir, adapter_name=first_name, is_trainable=False, autocast_adapter_dtype=False
            )
            for adapter_name, adapter_dir in adapter_specs[1:]:
                model.load_adapter(
                    adapter_dir, adapter_name=adapter_name, is_trainable=False, autocast_adapter_dtype=False
                )
            try:
                model.set_adapter([name for name, _ in adapter_specs])
            except Exception as exc:
                raise RuntimeError(
                    "The installed PEFT version cannot compose all checkpoint adapters."
                ) from exc
            if lora_fp32 and (load_in_4bit or load_in_8bit):
                _cast_lora_params_fp32(model)

    if not (load_in_4bit or load_in_8bit):
        model.to(device)
        if lora_path and device.type == "cuda":
            _cast_lora_params_dtype(model, torch.bfloat16)

    model.eval()
    tensor_device = _primary_device(model)
    return model, processor, tensor_device


def load_memory_bank(path: str) -> Any:
    try:
        from memory.memory_bank import MemoryBank

        bank = MemoryBank.load(path, map_location="cpu")
        print(f"[hil_vlm_eval_vf] loaded memory bank: {path}  items={len(bank)}")
        return bank
    except Exception as e:
        print(f"[hil_vlm_eval_vf] warning: failed to load memory bank ({e})")
        return None


def _memory_ref_trajectory_tensor(mem_item: Any) -> Optional[torch.Tensor]:
    t = getattr(mem_item, "recovery_action_trajectory", None)
    if t is not None and t.numel() > 0:
        return t
    c = getattr(mem_item, "action_chunk", None)
    if c is not None and c.numel() > 0:
        return c
    return None


def aggregate_memory_goal_actions(
    items: list[Any],
    *,
    mode: str = "topk_mean",
    current_vf: Optional[float] = None,
    goal_vf_getter: Optional[Callable[[Any], Optional[float]]] = None,
    action_dim: Optional[int] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Any], int]:
    """Aggregate the Round2 goal from retrieval results; returns (goal_action, start_action, representative_item, n_valid)."""
    valid_items: list[Any] = []
    starts: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    for item in items or []:
        traj_t = _memory_ref_trajectory_tensor(item)
        if traj_t is None or int(traj_t.shape[0]) <= 0:
            continue
        traj_np = np.asarray(traj_t.detach().cpu().numpy(), dtype=np.float64)
        if traj_np.ndim == 1:
            traj_np = traj_np.reshape(1, -1)
        if action_dim is not None and int(action_dim) > 0 and int(traj_np.shape[-1]) != int(action_dim):
            continue

        if current_vf is not None and goal_vf_getter is not None:
            goal_vf = goal_vf_getter(item)
            if goal_vf is None or not np.isfinite(goal_vf) or float(goal_vf) <= float(current_vf):
                continue

        valid_items.append(item)
        starts.append(np.asarray(traj_np[0], dtype=np.float64))
        goals.append(np.asarray(traj_np[-1], dtype=np.float64))

    if not valid_items:
        return None, None, None, 0

    if str(mode) == "topk_mean":
        goal_action = np.mean(np.stack(goals, axis=0), axis=0)
        start_action = np.mean(np.stack(starts, axis=0), axis=0)
    else:
        goal_action = goals[0]
        start_action = starts[0]
    return goal_action.astype(np.float64), start_action.astype(np.float64), valid_items[0], len(valid_items)


def trajectory_bins_to_float_matrix(
    bins: np.ndarray,
    a_min: np.ndarray,
    a_max: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """(T,7) integer bins -> float actions, matching the dataset de-quantization."""
    b = np.asarray(bins, dtype=np.float64)
    a_lo = np.asarray(a_min, dtype=np.float64).reshape(1, -1)
    a_hi = np.asarray(a_max, dtype=np.float64).reshape(1, -1)
    x = b / float(max(n_bins, 1))
    return x * (a_hi - a_lo) + a_lo


def trajectory_mean_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    t = min(int(pred.shape[0]), int(gt.shape[0]))
    if t <= 0:
        return 0.0
    return float(np.mean(np.abs(pred[:t] - gt[:t])))


def trajectory_float_to_bins_matrix(
    actions: np.ndarray,
    a_min: np.ndarray,
    a_max: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """(T,7) float actions -> integer bins, matching the dataset discretization (round to nearest)."""
    x = np.asarray(actions, dtype=np.float64)
    a_lo = np.asarray(a_min, dtype=np.float64).reshape(1, -1)
    a_hi = np.asarray(a_max, dtype=np.float64).reshape(1, -1)
    denom = np.maximum(a_hi - a_lo, 1e-8)
    r = (x - a_lo) / denom
    b = np.rint(r * float(max(n_bins, 1)))
    b = np.clip(b, 0, int(max(n_bins, 1)))
    return b.astype(np.int64)


def postprocess_round2_pred_bins_anchor(
    pred_bins: Optional[np.ndarray],
    *,
    current_action: np.ndarray,
    green_start_action: Optional[np.ndarray],
    goal_end_action: Optional[np.ndarray] = None,
    a_min: np.ndarray,
    a_max: np.ndarray,
    n_bins: int,
    anchor_steps: int = 0,
    end_closure_steps: int = 0,
    min_length_steps: int = 0,
    target_length_steps: int = 0,
) -> Optional[np.ndarray]:
    """Optional Round2 post-processing: start anchoring, length padding and tail convergence."""
    if pred_bins is None:
        return None
    b = np.asarray(pred_bins, dtype=np.int64)
    if b.ndim != 2 or b.shape[0] <= 0:
        return b

    pred_f = trajectory_bins_to_float_matrix(b, a_min, a_max, n_bins)
    cur = np.asarray(current_action, dtype=np.float64).reshape(-1)
    if cur.shape[0] != pred_f.shape[1]:
        return b

    n_anchor = max(0, int(anchor_steps))
    target = None
    if green_start_action is not None:
        gs = np.asarray(green_start_action, dtype=np.float64).reshape(-1)
        if gs.shape[0] == pred_f.shape[1]:
            target = gs
    if target is None and n_anchor > 0:
        target = pred_f[min(pred_f.shape[0] - 1, max(0, n_anchor - 1))]

    if n_anchor > 0:
        t = min(int(pred_f.shape[0]), int(max(1, n_anchor)))
        if t <= 1:
            pred_f[0] = cur
        else:
            for i in range(t):
                alpha = float(i) / float(t - 1)
                pred_f[i] = (1.0 - alpha) * cur + alpha * target

    goal = None
    if goal_end_action is not None:
        g = np.asarray(goal_end_action, dtype=np.float64).reshape(-1)
        if g.shape[0] == pred_f.shape[1]:
            goal = g
    if goal is None:
        goal = pred_f[-1].copy()

    desired_len = max(int(pred_f.shape[0]), int(max(0, min_length_steps)), int(max(0, target_length_steps)))
    if desired_len > int(pred_f.shape[0]):
        start = pred_f[-1].copy()
        need = int(desired_len - pred_f.shape[0])
        ext = np.zeros((need, pred_f.shape[1]), dtype=np.float64)
        for i in range(need):
            alpha = float(i + 1) / float(max(need, 1))
            ext[i] = (1.0 - alpha) * start + alpha * goal
        pred_f = np.concatenate([pred_f, ext], axis=0)

    k = max(0, int(end_closure_steps))
    if k > 0 and pred_f.shape[0] > 0:
        t = min(int(pred_f.shape[0]), k)
        s = int(pred_f.shape[0] - t)
        head = pred_f[s].copy()
        if t <= 1:
            pred_f[-1] = goal
        else:
            for i in range(t):
                alpha = float(i + 1) / float(t)
                pred_f[s + i] = (1.0 - alpha) * head + alpha * goal

    return trajectory_float_to_bins_matrix(pred_f, a_min, a_max, n_bins)


def _load_episode_by_stem(buffer_dir: str, stem: str) -> Optional[list]:
    """Load a full episode by episode_id (pkl stem); used to fetch the B-frame image for Round2."""
    if not stem:
        return None
    direct = os.path.join(buffer_dir, f"{stem}.pkl")
    if os.path.isfile(direct):
        try:
            with open(direct, "rb") as f:
                ep = pickle.load(f)
            return ep if isinstance(ep, list) and ep else None
        except Exception:
            pass
    for p in sorted(glob.glob(os.path.join(buffer_dir, "**", f"{stem}.pkl"), recursive=True)):
        try:
            with open(p, "rb") as f:
                ep = pickle.load(f)
            if isinstance(ep, list) and ep:
                return ep
        except Exception:
            continue
    return None
