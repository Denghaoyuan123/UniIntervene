#!/usr/bin/env python3
"""Run a trained stitched VF (SigLIP feature cache + Gemma value head) over a whole buffer and write per-transition vf_value_pred."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

JEPA_ROOT = Path(__file__).resolve().parents[1] / "JEPA"
sys.path.insert(0, str(JEPA_ROOT))
import train_vf_stitched as tvs  # noqa: E402


def _research_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(p: str, root: Path) -> str:
    if os.path.isabs(p):
        return p
    return str((root / p).resolve())


def load_vf_cfg(vf_dir: Path) -> dict:
    cfg_path = vf_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config.json: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_vf_model(
    vf_dir: Path,
    language_model: str,
    prompt: str,
    vision_dim: int,
    hidden_dim: int,
    device: torch.device,
    use_action_cond: bool,
    action_feature_dim: int,
    action_hidden_dim: int,
) -> torch.nn.Module:
    model = tvs.RobotValueModel(
        language_model,
        prompt,
        vision_dim=vision_dim,
        hidden_dim=hidden_dim,
        use_action_cond=bool(use_action_cond),
        action_feature_dim=int(action_feature_dim),
        action_hidden_dim=int(action_hidden_dim),
    ).to(device)
    full_ckpt = vf_dir / "stitched_value_model_best.pt"
    if full_ckpt.exists():
        state = torch.load(full_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    else:
        proj = vf_dir / "projector_best.pt"
        vh = vf_dir / "value_head_best.pt"
        if not proj.is_file() or not vh.is_file():
            raise FileNotFoundError(
                f"need stitched_value_model_best.pt or projector_best.pt+value_head_best.pt under {vf_dir}"
            )
        projector_sd = torch.load(proj, map_location="cpu", weights_only=True)
        value_head_sd = torch.load(vh, map_location="cpu", weights_only=True)
        model.projector.load_state_dict(projector_sd)
        model.value_head1.load_state_dict(value_head_sd, strict=False)
        model.value_head2.load_state_dict(value_head_sd, strict=False)
    model.eval()
    return model


def build_basename_index(cache: dict) -> dict[str, tuple[dict, dict]]:
    """basename -> (split_data, episode_meta row)"""
    out: dict[str, tuple[dict, dict]] = {}
    for key in ("train_data", "val_data", "test_data"):
        if key not in cache:
            continue
        data = cache[key]
        meta_list = data.get("episode_meta") or []
        for meta in meta_list:
            bn = os.path.basename(str(meta["path"]))
            if bn in out:
                raise RuntimeError(
                    f"duplicate basename in cache: {bn!r} (train/val overlap or repeated entry)"
                )
            out[bn] = (data, meta)
    return out


def _cache_episode_meta_map(cache: dict) -> dict[str, tuple[dict, dict]]:
    return build_basename_index(cache)


def _normalize_episode_basename(name: str) -> str:
    bn = os.path.basename(str(name))
    if bn.endswith("_nttg_vf.pkl"):
        bn = bn[:-len("_vf.pkl")]
    elif bn.endswith("_nttg.pkl"):
        bn = bn[:-len(".pkl")]
    elif bn.endswith(".pkl"):
        bn = bn[:-len(".pkl")]
    return bn


def _obs_to_pil_with_fallback(obs: dict, primary_keys: tuple[str, str], fallback_keys: tuple[str, str] | None = None) -> list["Image.Image"]:
    for side_key, wrist_key in [primary_keys] + ([fallback_keys] if fallback_keys is not None else []):
        if side_key in obs or wrist_key in obs:
            imgs = tvs.obs_to_pils(obs, use_external_wrist_keys=(side_key == tvs.ALT_SIDE_KEY and wrist_key == tvs.ALT_WRIST_KEY))
            if imgs:
                return imgs
    return []


def _cuda_batch_retryable(exc: BaseException) -> bool:
    msg = str(exc)
    return "CUBLAS" in msg or "CUDA error" in msg or "cuda runtime" in msg.lower()


def build_action_feature_tensor(
    transitions: list[dict],
    action_dim: int,
    use_delta_action: bool,
    device: torch.device,
) -> torch.Tensor:
    curr_actions, _ = tvs.build_action_features(
        transitions,
        action_dim=(int(action_dim) if int(action_dim) > 0 else None),
        use_delta_action=bool(use_delta_action),
    )
    return torch.tensor(curr_actions, dtype=torch.float32, device=device)


def _infer_task_prompt_from_episode(transitions: list[dict], default_prompt: str) -> str:
    for tr in transitions:
        for k in ("task_prompt", "task_description", "instruction", "instr", "prompt"):
            v = tr.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        infos = tr.get("infos")
        if isinstance(infos, dict):
            for k in ("task_prompt", "instruction", "instr", "prompt"):
                v = infos.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return default_prompt


def _encode_episode_raw(
    transitions: list[dict],
    vision_tower,
    image_processor,
    use_action_cond: bool,
    action_feature_dim: int,
    action_hidden_dim: int,
    use_external_wrist_keys: bool,
    device: torch.device,
):
    obs_feats = []
    next_obs_feats = []
    for i, tr in enumerate(transitions):
        imgs = tvs.obs_to_pils(tr["observations"], use_external_wrist_keys=use_external_wrist_keys)
        if not imgs:
            raise KeyError(f"transition[{i}] has no usable image key in observations")
        proc = image_processor(images=imgs, return_tensors="pt")
        pixel_values = proc.pixel_values.to(device)
        out = vision_tower(pixel_values=pixel_values)
        pooled = out.last_hidden_state.mean(dim=1).float().cpu()  # [N_img, D]
        if pooled.ndim == 2:
            pooled = pooled.mean(dim=0)  # average multiple views into a single vector
        obs_feats.append(pooled)

        nimgs = tvs.obs_to_pils(tr["next_observations"], use_external_wrist_keys=use_external_wrist_keys)
        if not nimgs:
            raise KeyError(f"transition[{i}] has no usable image key in next_observations")
        proc2 = image_processor(images=nimgs, return_tensors="pt")
        pixel_values2 = proc2.pixel_values.to(device)
        out2 = vision_tower(pixel_values=pixel_values2)
        pooled2 = out2.last_hidden_state.mean(dim=1).float().cpu()
        if pooled2.ndim == 2:
            pooled2 = pooled2.mean(dim=0)
        next_obs_feats.append(pooled2)

    obs_feats = torch.stack(obs_feats, dim=0)
    next_obs_feats = torch.stack(next_obs_feats, dim=0)
    action_feats = None
    if use_action_cond:
        action_feats = build_action_feature_tensor(
            transitions,
            action_dim=(action_feature_dim if action_feature_dim > 0 else 0),
            use_delta_action=False,
            device=device,
        )
        if int(action_hidden_dim) <= 0:
            raise ValueError("invalid action_hidden_dim")
    return obs_feats, next_obs_feats, action_feats


@torch.no_grad()
def score_episode(
    model: torch.nn.Module,
    obs_feats: torch.Tensor,
    prompt_texts: list[str] | None,
    action_feats: torch.Tensor | None,
    start: int,
    end: int,
    device: torch.device,
    batch_size: int,
    cuda_batch_cap: list[int] | None = None,
) -> list[float]:
    sl = slice(start, end)
    feats = obs_feats[sl].to(device)
    texts = prompt_texts[sl] if prompt_texts is not None else None
    acts = action_feats[sl] if action_feats is not None else None
    n = feats.shape[0]
    scores: list[float] = []
    cap = min(cuda_batch_cap[0], batch_size) if cuda_batch_cap else batch_size
    s = 0
    while s < n:
        chunk = min(cap, n - s)
        while chunk >= 1:
            try:
                batch_f = feats[s : s + chunk]
                batch_t = None if texts is None else texts[s : s + chunk]
                batch_a = acts[s : s + chunk] if acts is not None else None
                v = model.predict_value_from_siglip_feats(batch_f, prompt_texts=batch_t, actions=batch_a)
                scores.extend(v.float().cpu().numpy().tolist())
                s += chunk
                break
            except RuntimeError as ex:
                if (
                    chunk <= 1
                    or cuda_batch_cap is None
                    or device.type != "cuda"
                    or not _cuda_batch_retryable(ex)
                ):
                    raise
                chunk //= 2
                cap = chunk
                cuda_batch_cap[0] = cap
                print(f"[annotate] cuda batch cap -> {cap} ({type(ex).__name__})", flush=True)
    if len(scores) != n:
        raise RuntimeError("score_episode output length does not match the window")
    return scores


def main() -> None:
    root = _research_root()
    p = argparse.ArgumentParser(description="Write vf_value_pred into an nttg buffer and output *_nttg_vf.pkl")
    p.add_argument(
        "--buffer_dir",
        type=str,
        required=True,
        help="Directory containing transitions_*_nttg.pkl",
    )
    p.add_argument(
        "--vf_dir",
        type=str,
        required=True,
        help="VF run directory (with config.json, stitched_value_model_best.pt, etc.)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory (receives transitions_*_nttg_vf.pkl)",
    )
    p.add_argument("--cache_file", type=str, default=None, help="Override the cache path in config.json")
    p.add_argument("--allow_raw_fallback", action="store_true", help="If the cache cannot be aligned, extract features online from the raw episode and score it")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--use_action_cond", type=int, default=None, help="Override the config action switch; defaults to the VF config.")
    p.add_argument("--use_delta_action", type=int, default=None, help="Override the config delta_action switch; defaults to the VF config.")
    p.add_argument("--action_dim", type=int, default=None, help="Override the config action_dim; defaults to the VF config.")
    p.add_argument("--action_hidden_dim", type=int, default=None, help="Override the config action_hidden_dim; defaults to the VF config.")
    args = p.parse_args()

    buffer_dir = os.path.abspath(args.buffer_dir)
    vf_dir = Path(os.path.abspath(args.vf_dir))
    output_dir = Path(os.path.abspath(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_vf_cfg(vf_dir)
    cache_rel = args.cache_file or cfg.get("cache_file")
    if not cache_rel:
        raise ValueError("config.json has no cache_file and --cache_file was not given")
    cache_path = _resolve_path(str(cache_rel), root)
    if not os.path.isfile(cache_path):
        if not args.allow_raw_fallback:
            raise FileNotFoundError(f"cache not found: {cache_path}")
        cache_path = ""

    language_model = str(cfg["language_model"])
    prompt = str(cfg.get("prompt") or tvs.DEFAULT_PROMPT)
    vision_dim = int(cfg.get("proj_in", tvs.PROJ_IN))
    hidden_dim = int(cfg.get("proj_out", tvs.HIDDEN_DIM))
    cfg_use_action_cond = bool(cfg.get("use_action_cond", False))
    cfg_use_delta_action = bool(cfg.get("use_delta_action", False))
    cfg_action_dim = int(cfg.get("action_dim", 0))
    cfg_action_feature_dim = int(cfg.get("action_feature_dim", 0))
    cfg_action_hidden_dim = int(cfg.get("action_hidden_dim", 256))

    use_action_cond = bool(cfg_use_action_cond if args.use_action_cond is None else int(args.use_action_cond))
    use_delta_action = bool(cfg_use_delta_action if args.use_delta_action is None else int(args.use_delta_action))
    action_dim = int(cfg_action_dim if args.action_dim is None else args.action_dim)
    action_hidden_dim = int(cfg_action_hidden_dim if args.action_hidden_dim is None else args.action_hidden_dim)
    action_feature_dim = int(cfg_action_feature_dim)
    if use_action_cond and action_feature_dim <= 0:
        raise ValueError("use_action_cond=1 but config has invalid action_feature_dim")

    print(f"[annotate] cache={cache_path}", flush=True)
    print(f"[annotate] vf_dir={vf_dir}  vision_dim={vision_dim}  hidden_dim={hidden_dim}", flush=True)
    print(
        f"[annotate] action_cond={int(use_action_cond)} delta_action={int(use_delta_action)} "
        f"action_dim={action_dim} action_feature_dim={action_feature_dim} action_hidden_dim={action_hidden_dim}",
        flush=True,
    )

    cache = None
    bn_index: dict[str, tuple[dict, dict]] = {}
    cache_episode_names: set[str] = set()
    if cache_path:
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        bn_index = build_basename_index(cache)
        cache_episode_names = {os.path.basename(str(meta["path"])) for k in ("train_data", "val_data", "test_data") for meta in ((cache.get(k) or {}).get("episode_meta") or [])}

    device = torch.device(args.device)
    model = load_vf_model(
        vf_dir,
        language_model,
        prompt,
        vision_dim,
        hidden_dim,
        device,
        use_action_cond=use_action_cond,
        action_feature_dim=action_feature_dim,
        action_hidden_dim=action_hidden_dim,
    )
    vision_tower = None
    image_processor = None
    if args.allow_raw_fallback:
        cfg_v = load_vf_cfg(vf_dir)
        vision_model = str(cfg_v["vision_model"])
        print(f"[annotate] raw fallback enabled -> loading vision tower {vision_model}", flush=True)
        raw_vision = tvs.AutoModel.from_pretrained(vision_model, torch_dtype=torch.bfloat16)
        vision_tower = raw_vision.vision_model if hasattr(raw_vision, "vision_model") else raw_vision
        vision_tower = vision_tower.to(device)
        for p in vision_tower.parameters():
            p.requires_grad = False
        vision_tower.eval()
        image_processor = tvs.AutoImageProcessor.from_pretrained(vision_model)
    cuda_batch_cap: list[int] | None = [int(args.batch_size)] if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.init()
        idx = device.index if device.index is not None else 0
        torch.cuda.set_device(idx)
        with torch.inference_mode():
            dummy = torch.zeros(1, vision_dim, device=device, dtype=torch.float32)
            warm_action = torch.zeros(1, action_feature_dim, device=device, dtype=torch.float32) if use_action_cond else None
            _ = model.predict_value_from_siglip_feats(dummy, prompt_texts=None, actions=warm_action)
        torch.cuda.synchronize()
        print("[annotate] cuda warmup ok (batch=1)", flush=True)

    pkls = sorted(Path(buffer_dir).rglob("transitions_*_nttg.pkl"))
    if not pkls:
        pkls = sorted(Path(buffer_dir).rglob("transitions_*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"no transitions_*.pkl found: {buffer_dir}")

    for i, pkl_path in enumerate(pkls):
        bn = pkl_path.name
        bn_key = bn
        if bn_key not in bn_index:
            bn_key = _normalize_episode_basename(bn)
        if bn_key in bn_index:
            data, meta = bn_index[bn_key]
            start, end = int(meta["start"]), int(meta["end"])
            obs_feats = data["obs_feats"]
            prompt_texts = data.get("prompt_texts")
            if prompt_texts is not None and len(prompt_texts) != int(obs_feats.shape[0]):
                raise ValueError(
                    f"prompt_texts length {len(prompt_texts)} does not match obs_feats rows {obs_feats.shape[0]}: {bn!r}"
                )

            with open(pkl_path, "rb") as f:
                transitions = pickle.load(f)
            if not isinstance(transitions, list) or not transitions:
                raise ValueError(f"empty or non-list episode: {pkl_path}")
            n = len(transitions)
            if end - start != n:
                raise ValueError(
                    f"length mismatch {pkl_path}: transitions={n} cache_window={end - start} "
                    f"(start={start} end={end})"
                )
            action_feats = None
            if use_action_cond:
                action_feats = build_action_feature_tensor(
                    transitions,
                    action_dim=(action_dim if int(action_dim) > 0 else (action_feature_dim // (2 if use_delta_action else 1))),
                    use_delta_action=use_delta_action,
                    device=device,
                )
                if action_feats.shape[1] != action_feature_dim:
                    raise ValueError(
                        f"action feature dim mismatch for {bn}: got {action_feats.shape[1]} expected {action_feature_dim}"
                    )
            scores = score_episode(
                model,
                obs_feats,
                prompt_texts,
                action_feats,
                start,
                end,
                device,
                args.batch_size,
                cuda_batch_cap=cuda_batch_cap,
            )
        elif args.allow_raw_fallback:
            with open(pkl_path, "rb") as f:
                transitions = pickle.load(f)
            if not isinstance(transitions, list) or not transitions:
                raise ValueError(f"empty or non-list episode: {pkl_path}")
            prompt_text = _infer_task_prompt_from_episode(transitions, prompt)
            use_external_wrist_keys = any(k in (transitions[0].get("observations") or {}) for k in (tvs.ALT_SIDE_KEY, tvs.ALT_WRIST_KEY))
            obs_feats, _next_obs_feats, action_feats = _encode_episode_raw(
                transitions,
                vision_tower,
                image_processor,
                use_action_cond,
                action_feature_dim,
                action_hidden_dim,
                use_external_wrist_keys,
                device,
            )
            prompt_texts = [prompt_text] * len(transitions)
            scores = score_episode(
                model,
                obs_feats,
                prompt_texts,
                action_feats,
                0,
                len(transitions),
                device,
                args.batch_size,
                cuda_batch_cap=cuda_batch_cap,
            )
        else:
            raise KeyError(
                f"no episode_meta entry for {bn!r} in cache (buffer differs from the one used to build the cache?)"
            )
        for tr, sc in zip(transitions, scores, strict=True):
            tr["vf_value_pred"] = float(sc)

        out_name = bn.replace("_nttg.pkl", "_nttg_vf.pkl")
        if out_name == bn:
            out_name = bn[:-4] + "_vf.pkl"
        out_path = output_dir / out_name
        with open(out_path, "wb") as f:
            pickle.dump(transitions, f)

        if (i + 1) % 20 == 0 or (i + 1) == len(pkls):
            print(f"  [{i+1}/{len(pkls)}] -> {out_path.name}", flush=True)

    manifest = {
        "vf_dir": str(vf_dir),
        "cache_file": cache_path,
        "buffer_dir": buffer_dir,
        "output_dir": str(output_dir),
        "n_episodes": len(pkls),
        "language_model": language_model,
        "prompt": prompt,
        "use_action_cond": bool(use_action_cond),
        "use_delta_action": bool(use_delta_action),
        "action_dim": int(action_dim),
        "action_feature_dim": int(action_feature_dim),
        "action_hidden_dim": int(action_hidden_dim),
        "prompt_texts_source": "cache" if cache is not None and any(
            (cache.get(k) or {}).get("prompt_texts") is not None for k in ("train_data", "val_data")
        ) else ("raw_episode" if args.allow_raw_fallback else "vf_config_default"),
    }
    man_path = output_dir / "vf_value_pred_manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[annotate] done. manifest -> {man_path}", flush=True)


if __name__ == "__main__":
    main()
