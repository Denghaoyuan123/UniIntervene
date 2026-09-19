"""config_loader.py — YAML config system for hil_vlm_train.py."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import yaml


YAML_PATH_TO_ARG: dict[str, str] = {
    "model.backbone.model_path":            "model_path",
    "model.backbone.use_lora":              "use_lora",
    "model.backbone.lora.r":               "lora_r",
    "model.backbone.lora.alpha":           "lora_alpha",
    "model.backbone.lora.dropout":         "lora_dropout",
    "model.backbone.lora.target_modules":  "lora_target_modules",
    "model.backbone.lora.include_ffn":     "lora_include_ffn",
    "model.variant":                        "variant",
    "model.future_latent_mode":             "future_latent_mode",
    "model.future_target.source":               "future_target_source",
    "model.future_target.vjepa_path":           "vjepa_path",
    "model.future_target.require_future_loss":  "require_future_loss",
    "model.future_target.vjepa_feature_dim":    "vjepa_feature_dim",
    "model.future_target.vjepa_n_tokens":       "vjepa_n_tokens",
    "model.future_target.jepa_latent_dir":      "jepa_latent_dir",
    "model.future_target.jepa_latent_manifest": "jepa_latent_manifest",
    "model.intervention.head_type":                      "intervention_head_type",
    "model.intervention.loss_type":                      "intervention_loss_type",
    "model.intervention.threshold":                      "intervention_threshold",
    "model.intervention.pos_weight":                     "intervention_pos_weight",
    "model.intervention.focal.alpha":                    "focal_alpha",
    "model.intervention.focal.gamma":                    "focal_gamma",
    "model.intervention.tvr.window":                     "value_history_window",
    "model.intervention.tvr.gamma":                      "value_history_gamma",
    "model.intervention.tvr.epsilon":                    "value_history_epsilon",
    "model.intervention.tvr.lambda_tvr":                 "lambda_tvr",
    "model.intervention.tvr.value_history_embed_dim":    "value_history_embed_dim",
    "model.intervention.current_value.lambda_v_curr":    "lambda_v_curr",
    "data.buffer_dir":                      "buffer_dir",
    "data.memory_bank_path":                "memory_bank_path",
    "data.use_external_wrist_keys":         "use_external_wrist_keys",
    "data.proxy_vf.source_name":            "proxy_vf_source_name",
    "data.proxy_vf.is_latent_based":        "proxy_vf_is_latent_based",
    "data.proxy_vf.checkpoint_path":        "proxy_vf_checkpoint_path",
    "data.proxy_vf.strict":                 "strict_proxy_vf",
    "data.proxy_vf.score_manifest_path":    "proxy_vf_score_manifest_path",
    "data.proxy_vf.score_buffer_dir":       "proxy_vf_score_buffer_dir",
    "data.proxy_vf.latent_dir":             "proxy_vf_latent_dir",
    "data.proxy_vf.latent_manifest":        "proxy_vf_latent_manifest",
    "data.proxy_vf.language_model":         "proxy_vf_language_model",
    "data.proxy_vf.prompt":                 "proxy_vf_prompt",
    "training.output_dir":                  "output_dir",
    "training.epochs":                      "epochs",
    "training.batch_size":                  "batch_size",
    "training.max_train_steps":             "max_train_steps",
    "training.skip_val":                    "skip_val",
    "training.lr.backbone":                 "lr",
    "training.lr.heads":                    "head_lr",
    "training.loss_weights.future":         "lambda_future",
    "training.loss_weights.q":              "lambda_q",
    "training.loss_weights.intervention":   "lambda_intervention",
    "training.loss_weights.tvr":            "lambda_tvr",
    "training.loss_weights.v_curr":         "lambda_v_curr",
    "training.stage_schedule.enabled":      "stage_schedule",
    "training.stage_schedule.stage1_epochs": "stage1_epochs",
    "training.stage_schedule.stage2_epochs": "stage2_epochs",
    "training.gradient_checkpointing":      "gradient_checkpointing",
    "training.hidden_state_capture":        "hidden_state_capture",
    "training.adam8bit":                    "adam8bit",
    "training.log_interval":               "log_interval",
    "training.save_every_steps":           "save_every_steps",
    "training.save_every_epochs":          "save_every_epochs",
    "training.wandb.enabled":              "use_wandb",
    "training.wandb.project":              "wandb_project",
    "training.wandb.entity":               "wandb_entity",
    "training.wandb.run_name":             "wandb_run_name",
    "evaluation.debug_samples":             "vq1_debug_samples",
    "vq2.adapter_mode":                     "vq2_adapter_mode",
    "vq2.vq1_shared_lora_path":             "vq1_shared_lora_path",
    "vq2.residual_lora.r":                  "vq2_residual_lora_r",
    "vq2.residual_lora.alpha":              "vq2_residual_lora_alpha",
    "vq2.residual_lora.dropout":            "vq2_residual_lora_dropout",
}

_ARG_TO_YAML_PATH: dict[str, str] = {v: k for k, v in YAML_PATH_TO_ARG.items()}



def _flatten_yaml(d: dict, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            result.update(_flatten_yaml(v, key))
        else:
            result[key] = v
    return result


def _coerce(val_str: str, dest: str, parser) -> Any:
    """Convert a string override value to the right Python type using parser metadata."""
    import argparse
    type_fn = None
    is_bool_action = False
    for action in parser._actions:
        if action.dest == dest:
            type_fn = action.type
            is_bool_action = isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                                                  argparse.BooleanOptionalAction))
            break

    if is_bool_action or type_fn is None and val_str.lower() in ("true", "false", "yes", "no", "1", "0"):
        return val_str.lower() in ("true", "yes", "1")

    if type_fn is not None:
        try:
            return type_fn(val_str)
        except (ValueError, TypeError):
            pass

    import ast
    try:
        return ast.literal_eval(val_str)
    except (ValueError, SyntaxError):
        return val_str


def _nested_set(d: dict, path: str, value: Any) -> None:
    """Set d[k1][k2]...[kn] = value, creating intermediate dicts."""
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value



def load_yaml_config(path: str) -> dict[str, Any]:
    """Load a nested YAML config file and return flat {argparse_dest: value}."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    flat = _flatten_yaml(raw)
    result: dict[str, Any] = {}
    unknown: list[str] = []
    for yaml_key, value in flat.items():
        if yaml_key in YAML_PATH_TO_ARG:
            result[YAML_PATH_TO_ARG[yaml_key]] = value
        else:
            unknown.append(yaml_key)
    if unknown:
        print(f"[config] WARNING: unknown YAML keys (ignored): {unknown}", flush=True)
    return result


def parse_overrides(override_list: list[str], parser) -> dict[str, Any]:
    """Parse ['key=value', ...] where key is a YAML dot-path or argparse dest."""
    result: dict[str, Any] = {}
    for item in (override_list or []):
        if "=" not in item:
            raise ValueError(f"[config] --override item must be 'key=value', got: {item!r}")
        key, _, val_str = item.partition("=")
        key = key.strip()
        val_str = val_str.strip()
        dest = YAML_PATH_TO_ARG.get(key, key)
        result[dest] = _coerce(val_str, dest, parser)
    return result


def apply_to_args(args, flat_dict: dict[str, Any], source: str = "config") -> None:
    """Apply {dest: value} onto an argparse.Namespace, logging each change."""
    for dest, value in flat_dict.items():
        old = getattr(args, dest, "<unset>")
        setattr(args, dest, value)
        print(f"[config] override ({source}): {dest} = {value!r}  (was {old!r})", flush=True)


def validate_config(args) -> None:
    """Raise ValueError for invalid config combinations."""
    variant = str(getattr(args, "variant", "vq2_fast")).lower()
    itype = str(getattr(args, "intervention_head_type", "binary")).lower()
    adapter_mode = str(getattr(args, "vq2_adapter_mode", "base_lora")).lower()

    if variant == "no_future" and getattr(args, "require_future_loss", False):
        raise ValueError(
            "[config] variant=no_future is incompatible with require_future_loss=True. "
            "Use model.future_target.require_future_loss: false in your config."
        )

    if variant == "no_intervention" and getattr(args, "lambda_intervention", 1.0) > 0:
        print(
            "[config] WARNING: variant=no_intervention but lambda_intervention > 0. "
            "The variant spec disables the intervention head; lambda_intervention is ignored.",
            flush=True,
        )

    if itype == "tvr_history":
        window = int(getattr(args, "value_history_window", 8))
        if window <= 0:
            raise ValueError(
                f"[config] intervention.head_type=tvr_history requires value_history_window > 0, "
                f"got {window}."
            )

    if itype == "current_value":
        lvc = float(getattr(args, "lambda_v_curr", 1.0))
        if lvc <= 0.0:
            raise ValueError(
                f"[config] intervention.head_type=current_value requires lambda_v_curr > 0, "
                f"got {lvc}. Set model.intervention.current_value.lambda_v_curr in your config."
            )

    if itype == "binary" and float(getattr(args, "lambda_tvr", 1.0)) != 1.0:
        print(
            "[config] WARNING: intervention.head_type=binary does not use lambda_tvr; "
            "the value is ignored.",
            flush=True,
        )

    if adapter_mode == "shared_residual":
        p = str(getattr(args, "vq1_shared_lora_path", "") or "")
        if not p or not os.path.exists(p):
            raise ValueError(
                f"[config] vq2.adapter_mode=shared_residual requires a valid "
                f"vq2.vq1_shared_lora_path, got: {p!r}. "
                "Set vq2.vq1_shared_lora_path in your config or --override."
            )

    print("[config] validation passed.", flush=True)


def log_resolved_config(args, config_path: str) -> None:
    """Print the fully resolved configuration in a structured format."""
    print(f"[config] ── resolved config ──────────────────────────", flush=True)
    print(f"[config] loaded from: {config_path}", flush=True)
    print(f"[config] resolved variant:            {getattr(args, 'variant', '?')}", flush=True)
    print(f"[config] resolved future_latent_mode: {getattr(args, 'future_latent_mode', '?')}", flush=True)
    print(f"[config] resolved intervention_head:  {getattr(args, 'intervention_head_type', '?')}", flush=True)
    print(f"[config] resolved batch_size:         {getattr(args, 'batch_size', '?')}", flush=True)
    print(f"[config] resolved epochs:             {getattr(args, 'epochs', '?')}", flush=True)
    print(f"[config] resolved output_dir:         {getattr(args, 'output_dir', '?')}", flush=True)
    print(f"[config] resolved lr (backbone):      {getattr(args, 'lr', '?')}", flush=True)
    print(f"[config] resolved lr (heads):         {getattr(args, 'head_lr', '?')}", flush=True)
    print(f"[config] resolved lambda_future:      {getattr(args, 'lambda_future', '?')}", flush=True)
    print(f"[config] resolved lambda_q:           {getattr(args, 'lambda_q', '?')}", flush=True)
    print(f"[config] resolved lambda_intervention:{getattr(args, 'lambda_intervention', '?')}", flush=True)
    print(f"[config] ─────────────────────────────────────────────", flush=True)


def save_resolved_config(args, config_path: str) -> None:
    """Save resolved config as YAML and JSON to args.output_dir."""
    output_dir = str(getattr(args, "output_dir", "") or "")
    if not output_dir:
        print("[config] WARNING: output_dir not set; skipping resolved config save.", flush=True)
        return
    os.makedirs(output_dir, exist_ok=True)

    nested: dict[str, Any] = {}
    for dest, yaml_path in _ARG_TO_YAML_PATH.items():
        if hasattr(args, dest):
            _nested_set(nested, yaml_path, getattr(args, dest))

    nested["_meta"] = {
        "config_file": os.path.abspath(config_path),
        "python_argv": sys.argv,
    }

    yaml_out = os.path.join(output_dir, "resolved_config.yaml")
    json_out = os.path.join(output_dir, "resolved_config.json")

    with open(yaml_out, "w") as f:
        yaml.dump(nested, f, allow_unicode=True, sort_keys=True)
    with open(json_out, "w") as f:
        json.dump(nested, f, indent=2, ensure_ascii=False, default=str)

    print(f"[config] saved resolved_config.yaml → {yaml_out}", flush=True)
    print(f"[config] saved resolved_config.json → {json_out}", flush=True)
