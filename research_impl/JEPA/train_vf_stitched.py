from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    import wandb
except ImportError:
    wandb = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIDE_KEY = "side_policy_256"
WRIST_KEY = "wrist_1"
ALT_SIDE_KEY = "external"
ALT_WRIST_KEY = "wrist"
DEFAULT_PROMPT = "Press the button to complete the task."
DEFAULT_VISION_MODEL = "google/siglip-so400m-patch14-384"
DEFAULT_LANGUAGE_MODEL = "google/gemma-3-270m-it"
DEFAULT_VQA_DATASET = "HuggingFaceM4/VQAv2"
CACHE_VERSION = 7
PROJ_IN = 1152
HIDDEN_DIM = 640


def log(msg: str) -> None:
    print(msg, flush=True)


def infer_task_name(buffer_dir: str) -> str:
    path = Path(buffer_dir)
    name = path.name
    if name.startswith("buffer") and path.parent.name:
        name = path.parent.name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return name or "dataset"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _infer_episode_task_name_from_path(pkl_path: str) -> str:
    """Read the task name from the episode and fall back to its parent directory."""
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            t = obj[0].get("task_name")
            if t:
                return str(t)
        if isinstance(obj, dict):
            t = obj.get("task_name")
            if t:
                return str(t)
    except Exception:
        pass

    parent = Path(pkl_path).parent.name
    return str(parent or "unknown_task")


def _split_counts(n: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    """Per-task split counts (train/val/test), favoring 8:1:1 when possible."""
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    n_val = max(1, int(round(n * float(val_ratio))))
    n_test = max(1, int(round(n * float(test_ratio))))
    if n_val + n_test >= n:
        overflow = (n_val + n_test) - (n - 1)
        cut_test = min(overflow, max(0, n_test - 1))
        n_test -= cut_test
        overflow -= cut_test
        if overflow > 0:
            cut_val = min(overflow, max(0, n_val - 1))
            n_val -= cut_val

    n_train = n - n_val - n_test
    if n_train <= 0:
        n_train = 1
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1

    return n_train, n_val, n_test


def stratified_split_pkls(
    pkls: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str], dict[str, dict[str, int]]]:
    """Task-stratified split: for each task, shuffle independently and split into"""
    if len(pkls) <= 1:
        return list(pkls), [], [], {}

    task_to_pkls: dict[str, list[str]] = defaultdict(list)
    for p in pkls:
        task_to_pkls[_infer_episode_task_name_from_path(p)].append(p)

    rng = random.Random(seed)
    train_pkls: list[str] = []
    val_pkls: list[str] = []
    test_pkls: list[str] = []
    stats: dict[str, dict[str, int]] = {}

    for task, arr in sorted(task_to_pkls.items()):
        local = list(arr)
        rng.shuffle(local)
        n = len(local)
        n_train, n_val, n_test = _split_counts(n, val_ratio=val_ratio, test_ratio=test_ratio)
        train_part = local[:n_train]
        val_part = local[n_train:n_train + n_val]
        test_part = local[n_train + n_val:n_train + n_val + n_test]

        train_pkls.extend(train_part)
        val_pkls.extend(val_part)
        test_pkls.extend(test_part)
        stats[task] = {
            "total": n,
            "train": len(train_part),
            "val": len(val_part),
            "test": len(test_part),
        }

    rng.shuffle(train_pkls)
    rng.shuffle(val_pkls)
    rng.shuffle(test_pkls)
    return train_pkls, val_pkls, test_pkls, stats


def _build_episode_index(cached: dict[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    index: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for split_key in ("train_data", "val_data", "test_data"):
        data = cached.get(split_key)
        if data is None:
            continue
        for meta in data.get("episode_meta", []):
            bn = os.path.basename(str(meta["path"]))
            if bn in index:
                raise RuntimeError(f"duplicate cache episode basename: {bn}")
            index[bn] = (data, meta)
    return index


def _rebuild_subset_from_cache(index: dict[str, tuple[dict[str, Any], dict[str, Any]]], paths: list[str]) -> dict[str, Any]:
    if not paths:
        raise ValueError("cannot rebuild a cache subset from an empty path list")

    obs_parts = []
    next_obs_parts = []
    rewards_parts = []
    dones_parts = []
    mc_parts = []
    progress_parts = []
    success_parts = []
    action_parts = []
    next_action_parts = []
    prompt_texts: list[str] = []
    task_names: list[str] = []
    traj_ids: list[int] = []
    step_ids: list[int] = []
    traj_lens: list[int] = []
    episode_meta: list[dict[str, Any]] = []

    cursor = 0
    for traj_id, path in enumerate(paths):
        bn = os.path.basename(path)
        if bn not in index:
            raise KeyError(f"cache is missing episode: {bn}")
        data, meta = index[bn]
        start = int(meta["start"])
        end = int(meta["end"])
        T = end - start
        if T <= 0:
            continue

        obs_parts.append(data["obs_feats"][start:end])
        next_obs_parts.append(data["next_obs_feats"][start:end])
        rewards_parts.append(data["rewards"][start:end])
        dones_parts.append(data["dones"][start:end])
        mc_parts.append(data["mc_returns"][start:end])
        progress_parts.append(data["progress"][start:end])
        success_parts.append(data["success_mask"][start:end])
        if data.get("actions") is not None:
            action_parts.append(data["actions"][start:end])
        if data.get("next_actions") is not None:
            next_action_parts.append(data["next_actions"][start:end])

        prompt_texts.extend(data["prompt_texts"][start:end])
        task_names.extend(data["task_names"][start:end])
        traj_ids.extend([traj_id] * T)
        step_ids.extend(list(range(T)))
        traj_lens.extend([T] * T)

        episode_meta.append({
            "path": str(meta["path"]),
            "success": bool(meta["success"]),
            "start": cursor,
            "end": cursor + T,
            "length": T,
        })
        cursor += T

    if not obs_parts:
        raise ValueError("rebuilt cache subset is empty")

    out = {
        "obs_feats": torch.cat(obs_parts, dim=0),
        "next_obs_feats": torch.cat(next_obs_parts, dim=0),
        "rewards": torch.cat(rewards_parts, dim=0),
        "dones": torch.cat(dones_parts, dim=0),
        "mc_returns": torch.cat(mc_parts, dim=0),
        "progress": torch.cat(progress_parts, dim=0),
        "success_mask": torch.cat(success_parts, dim=0),
        "prompt_texts": prompt_texts,
        "task_names": task_names,
        "traj_ids": torch.tensor(traj_ids, dtype=torch.long),
        "step_ids": torch.tensor(step_ids, dtype=torch.long),
        "traj_lens": torch.tensor(traj_lens, dtype=torch.long),
        "episode_meta": episode_meta,
    }
    if action_parts:
        out["actions"] = torch.cat(action_parts, dim=0)
    if next_action_parts:
        out["next_actions"] = torch.cat(next_action_parts, dim=0)
    return out


def _as_action_vec(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty action")
    return arr


def _extract_actions_for_episode(transitions: list[dict], action_dim: int | None = None) -> np.ndarray:
    vecs: list[np.ndarray] = []
    for i, tr in enumerate(transitions):
        if "actions" not in tr:
            raise KeyError(f"transition[{i}] missing actions")
        vecs.append(_as_action_vec(tr["actions"]))
    if not vecs:
        return np.zeros((0, 0), dtype=np.float32)

    dim = int(action_dim) if (action_dim is not None and int(action_dim) > 0) else max(v.size for v in vecs)
    out = np.zeros((len(vecs), dim), dtype=np.float32)
    for i, v in enumerate(vecs):
        n = min(dim, v.size)
        out[i, :n] = v[:n]
    return out


def build_action_features(
    transitions: list[dict],
    action_dim: int | None = None,
    use_delta_action: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-step action features and next-step action features."""
    actions = _extract_actions_for_episode(transitions, action_dim=action_dim)
    if actions.shape[0] == 0:
        return actions, actions

    feats = actions
    if use_delta_action:
        prev = np.concatenate([actions[:1], actions[:-1]], axis=0)
        delta = actions - prev
        feats = np.concatenate([actions, delta], axis=1)
    next_feats = np.concatenate([feats[1:], feats[-1:]], axis=0)
    return feats.astype(np.float32), next_feats.astype(np.float32)


def init_wandb(args):
    if not args.use_wandb:
        return None
    if wandb is None:
        raise ImportError("wandb is not installed but --use_wandb is set")
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or os.path.basename(os.path.abspath(args.output_dir)),
        config={k: v for k, v in vars(args).items() if k != "device"},
        dir=args.output_dir,
    )


def log_wandb(metrics: dict[str, float], step: int | None = None) -> None:
    if wandb is None or wandb.run is None:
        return
    if step is None:
        wandb.log(metrics)
    else:
        wandb.log(metrics, step=step)


def finish_wandb(summary: dict[str, float], image_path: str | None = None) -> None:
    if wandb is None or wandb.run is None:
        return
    for key, value in summary.items():
        wandb.run.summary[key] = value
    if image_path and os.path.exists(image_path):
        wandb.log({"plots/trajectory_value_mc_progress": wandb.Image(image_path)})
    wandb.finish()


def obs_to_pils(obs: dict, *, use_external_wrist_keys: bool = False) -> list[Image.Image]:
    imgs = []
    keys = [ALT_SIDE_KEY, ALT_WRIST_KEY] if use_external_wrist_keys else [SIDE_KEY, WRIST_KEY]
    for key in keys:
        if key in obs:
            arr = np.array(obs[key])
            if arr.ndim == 4:
                arr = arr[0]
            imgs.append(Image.fromarray(arr.astype(np.uint8)))
    return imgs


def infer_success(transitions: list[dict]) -> bool:
    if not transitions:
        return False
    if "is_success" in transitions[-1]:
        return bool(transitions[-1]["is_success"])
    if float(transitions[-1].get("raw_value", -500.0)) == 0.0:
        return True
    return any(float(tr.get("rewards", 0.0)) > 0.0 for tr in transitions)


def compute_discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = float(rewards[idx]) + gamma * running
        out[idx] = running
    return out


def preprocess_episode(transitions: list[dict], gamma: float, terminal_success_reward: float = 1.0, terminal_failure_reward: float = 0.0) -> dict[str, np.ndarray | bool]:
    T = len(transitions)
    success = infer_success(transitions)
    default_rewards = np.zeros(T, dtype=np.float32)
    if T > 0:
        default_rewards[-1] = terminal_success_reward if success else terminal_failure_reward
    dones = np.asarray(
        [bool(tr.get("dones", False)) or (idx == T - 1) for idx, tr in enumerate(transitions)],
        dtype=np.float32,
    )
    denom = max(T - 1, 1)
    if all("progress" in tr for tr in transitions):
        progress = np.asarray([float(tr["progress"]) for tr in transitions], dtype=np.float32)
    else:
        progress = np.asarray([float(idx) / float(denom) for idx in range(T)], dtype=np.float32)
    actions = _extract_actions_for_episode(transitions, action_dim=None)
    rewards = default_rewards
    mc_returns = compute_discounted_returns(rewards, gamma)
    return {
        "rewards": rewards,
        "dones": dones,
        "progress": progress,
        "actions": actions,
        "mc_returns": mc_returns,
        "success": success,
    }


@dataclass
class EpisodeMeta:
    path: str
    success: bool
    start: int
    end: int
    length: int


@torch.no_grad()
def encode_image_batch(imgs: list[Image.Image], vision_tower, image_processor, device: str, batch_size: int) -> torch.Tensor:
    pooled_batches = []
    for start in range(0, len(imgs), batch_size):
        batch_imgs = imgs[start:start + batch_size]
        proc = image_processor(images=batch_imgs, return_tensors="pt")
        pixel_values = proc.pixel_values.to(device)
        out = vision_tower(pixel_values=pixel_values)
        pooled = out.last_hidden_state.mean(dim=1).float().cpu()
        pooled_batches.append(pooled)
    return torch.cat(pooled_batches, dim=0)


@torch.no_grad()
def encode_episode_observations(observations: list[dict], vision_tower, image_processor, device: str, feat_dim: int, batch_size: int, *, use_external_wrist_keys: bool = False) -> list[torch.Tensor]:
    all_imgs = []
    owners = []
    for obs_idx, obs in enumerate(observations):
        for img in obs_to_pils(obs, use_external_wrist_keys=use_external_wrist_keys):
            all_imgs.append(img)
            owners.append(obs_idx)
    if not all_imgs:
        return [torch.zeros(feat_dim, dtype=torch.float32) for _ in observations]
    pooled = encode_image_batch(all_imgs, vision_tower, image_processor, device, batch_size)
    per_obs = [[] for _ in observations]
    for feat, obs_idx in zip(pooled, owners):
        per_obs[obs_idx].append(feat)
    outputs = []
    for feats in per_obs:
        if feats:
            outputs.append(torch.stack(feats).mean(dim=0))
        else:
            outputs.append(torch.zeros(feat_dim, dtype=torch.float32))
    return outputs


@torch.no_grad()
def cache_transition_features(pkls: list[str], vision_tower, image_processor, device: str, gamma: float, feat_dim: int, phase1_batch_size: int, default_prompt: str, args) -> dict[str, Any]:
    obs_feats = []
    next_obs_feats = []
    rewards = []
    dones = []
    mc_returns = []
    progress = []
    success_mask = []
    actions = []
    next_actions = []
    prompt_texts = []
    task_names = []
    traj_ids = []
    step_ids = []
    traj_lens = []
    episode_meta: list[EpisodeMeta] = []

    n_success, n_fail = 0, 0
    for traj_id, path in enumerate(pkls):
        with open(path, "rb") as f:
            transitions = pickle.load(f)
        if not isinstance(transitions, list) or len(transitions) == 0:
            continue
        episode = preprocess_episode(transitions, gamma, args.terminal_success_reward, args.terminal_failure_reward)
        curr_actions, next_step_actions = build_action_features(
            transitions,
            action_dim=(int(args.action_dim) if int(getattr(args, "action_dim", 0)) > 0 else None),
            use_delta_action=bool(int(getattr(args, "use_delta_action", 0))),
        )
        T = len(transitions)
        success = bool(episode["success"])
        n_success += int(success)
        n_fail += int(not success)

        curr_feats = encode_episode_observations(
            [tr["observations"] for tr in transitions],
            vision_tower,
            image_processor,
            device,
            feat_dim,
            phase1_batch_size,
            use_external_wrist_keys=bool(int(getattr(args, "use_external_wrist_keys", 0))),
        )
        final_next_feat = encode_episode_observations(
            [transitions[-1]["next_observations"]],
            vision_tower,
            image_processor,
            device,
            feat_dim,
            phase1_batch_size,
            use_external_wrist_keys=bool(int(getattr(args, "use_external_wrist_keys", 0))),
        )[0]

        episode_prompt = str(transitions[0].get("task_prompt", default_prompt))
        episode_task_name = str(transitions[0].get("task_name", "default"))
        start = len(obs_feats)
        for step_idx in range(T):
            next_feat = curr_feats[step_idx + 1] if step_idx < T - 1 else final_next_feat
            obs_feats.append(curr_feats[step_idx])
            next_obs_feats.append(next_feat)
            rewards.append(float(episode["rewards"][step_idx]))
            dones.append(float(episode["dones"][step_idx]))
            mc_returns.append(float(episode["mc_returns"][step_idx]))
            progress.append(float(episode["progress"][step_idx]))
            success_mask.append(float(success))
            actions.append(curr_actions[step_idx])
            next_actions.append(next_step_actions[step_idx])
            prompt_texts.append(str(transitions[step_idx].get("task_prompt", episode_prompt)))
            task_names.append(str(transitions[step_idx].get("task_name", episode_task_name)))
            traj_ids.append(traj_id)
            step_ids.append(step_idx)
            traj_lens.append(T)
        end = len(obs_feats)
        episode_meta.append(EpisodeMeta(path, success, start, end, T))
        if (traj_id + 1) % 10 == 0 or (traj_id + 1) == len(pkls):
            log(f"  [cache] {traj_id+1}/{len(pkls)} success={n_success} fail={n_fail}")

    return {
        "obs_feats": torch.stack(obs_feats),
        "next_obs_feats": torch.stack(next_obs_feats),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "dones": torch.tensor(dones, dtype=torch.float32),
        "mc_returns": torch.tensor(mc_returns, dtype=torch.float32),
        "progress": torch.tensor(progress, dtype=torch.float32),
        "success_mask": torch.tensor(success_mask, dtype=torch.float32),
        "actions": torch.tensor(np.asarray(actions), dtype=torch.float32),
        "next_actions": torch.tensor(np.asarray(next_actions), dtype=torch.float32),
        "prompt_texts": prompt_texts,
        "task_names": task_names,
        "traj_ids": torch.tensor(traj_ids, dtype=torch.long),
        "step_ids": torch.tensor(step_ids, dtype=torch.long),
        "traj_lens": torch.tensor(traj_lens, dtype=torch.long),
        "episode_meta": [meta.__dict__ for meta in episode_meta],
    }


class RobotValueModel(nn.Module):
    def __init__(
        self,
        language_model_name: str,
        prompt: str,
        vision_dim: int = PROJ_IN,
        hidden_dim: int = HIDDEN_DIM,
        projector_micro_batch_size: int = 0,
        use_action_cond: bool = False,
        action_feature_dim: int = 0,
        action_hidden_dim: int = 256,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, vision_dim),
            nn.GELU(),
            nn.Linear(vision_dim, hidden_dim),
        )
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.default_prompt = prompt
        self.use_action_cond = bool(use_action_cond)
        self.action_feature_dim = int(action_feature_dim)
        self.action_hidden_dim = int(action_hidden_dim)
        fused_dim = hidden_dim
        if self.use_action_cond:
            if self.action_feature_dim <= 0:
                raise ValueError("use_action_cond=1 requires action_feature_dim > 0")
            self.action_encoder = nn.Sequential(
                nn.Linear(self.action_feature_dim, self.action_hidden_dim),
                nn.GELU(),
                nn.Linear(self.action_hidden_dim, hidden_dim),
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
            )
            fused_dim = hidden_dim
        self.value_head1 = nn.Linear(fused_dim, 1)
        self.value_head2 = nn.Linear(fused_dim, 1)
        self.hidden_dim = hidden_dim
        self.projector_micro_batch_size = max(int(projector_micro_batch_size), 0)

    def backbone(self):
        return getattr(self.language_model, "model", self.language_model)

    def _tokenize_prompts(self, prompt_texts: list[str] | None, device: torch.device | str):
        prompts = prompt_texts or [self.default_prompt]
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        return tok.input_ids.to(device), tok.attention_mask.to(device)

    def _project_siglip_feats(self, siglip_feats: torch.Tensor) -> torch.Tensor:
        chunk = self.projector_micro_batch_size
        if chunk <= 0 or siglip_feats.shape[0] <= chunk:
            return self.projector(siglip_feats)
        outputs = []
        for start in range(0, siglip_feats.shape[0], chunk):
            outputs.append(self.projector(siglip_feats[start:start + chunk]))
        return torch.cat(outputs, dim=0)

    def encode_from_siglip_feats(self, siglip_feats: torch.Tensor, prompt_texts: list[str] | None = None) -> torch.Tensor:
        device = siglip_feats.device
        device_type = device.type if isinstance(device, torch.device) else str(device).split(":", 1)[0]
        batch_size = siglip_feats.shape[0]
        prompt_ids, prompt_mask = self._tokenize_prompts(prompt_texts, device)
        if prompt_ids.shape[0] == 1 and batch_size > 1:
            prompt_ids = prompt_ids.expand(batch_size, -1)
            prompt_mask = prompt_mask.expand(batch_size, -1)
        prompt_embeds = self.language_model.get_input_embeddings()(prompt_ids)
        img_token = self._project_siglip_feats(siglip_feats).to(prompt_embeds.dtype).unsqueeze(1)
        inputs_embeds = torch.cat([img_token, prompt_embeds], dim=1)
        attn = torch.cat([torch.ones(batch_size, 1, device=device, dtype=prompt_mask.dtype), prompt_mask], dim=1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device_type == "cuda"):
            outputs = self.backbone()(inputs_embeds=inputs_embeds, attention_mask=attn, return_dict=True)
        last_hidden = outputs.last_hidden_state
        seq_lens = attn.sum(dim=1) - 1
        batch_idx = torch.arange(batch_size, device=device)
        return last_hidden[batch_idx, seq_lens].float()

    def _fuse_obs_action(self, obs_hidden: torch.Tensor, actions: torch.Tensor | None) -> torch.Tensor:
        if not self.use_action_cond:
            return obs_hidden
        if actions is None:
            raise ValueError("action-conditioned VF requires action tensor")
        if actions.dim() != 2:
            raise ValueError(f"actions must be rank-2 [B,D], got shape={tuple(actions.shape)}")
        if actions.shape[-1] != self.action_feature_dim:
            raise ValueError(f"action dim mismatch: expected {self.action_feature_dim}, got {actions.shape[-1]}")
        act_hidden = self.action_encoder(actions.to(obs_hidden.dtype))
        fused = self.fusion(torch.cat([obs_hidden, act_hidden], dim=-1))
        return fused

    def predict_value_pair_from_siglip_feats(
        self,
        siglip_feats: torch.Tensor,
        prompt_texts: list[str] | None = None,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_hidden = self.encode_from_siglip_feats(siglip_feats, prompt_texts=prompt_texts)
        hidden = self._fuse_obs_action(obs_hidden, actions=actions)
        value1 = self.value_head1(hidden).squeeze(-1)
        value2 = self.value_head2(hidden).squeeze(-1)
        return value1, value2

    def predict_value_from_siglip_feats(
        self,
        siglip_feats: torch.Tensor,
        prompt_texts: list[str] | None = None,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value1, value2 = self.predict_value_pair_from_siglip_feats(siglip_feats, prompt_texts=prompt_texts, actions=actions)
        return torch.minimum(value1, value2)

    def vqa_loss(self, siglip_feats: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = siglip_feats.device
        device_type = device.type if isinstance(device, torch.device) else str(device).split(":", 1)[0]
        img_token = self._project_siglip_feats(siglip_feats).unsqueeze(1)
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        text_embeds = text_embeds.to(img_token.dtype)
        inputs_embeds = torch.cat([img_token, text_embeds], dim=1)
        prefix = torch.full((labels.size(0), 1), -100, dtype=labels.dtype, device=device)
        expanded_labels = torch.cat([prefix, labels], dim=1)
        expanded_mask = torch.cat(
            [torch.ones(labels.size(0), 1, dtype=attention_mask.dtype, device=device), attention_mask],
            dim=1,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device_type == "cuda"):
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=expanded_mask,
                labels=expanded_labels,
                return_dict=True,
            )
        return outputs.loss.float()


class LocalVQADataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


class HFDatasetWrapper(Dataset):
    def __init__(self, dataset, max_samples: int):
        if max_samples > 0:
            self.dataset = dataset.select(range(min(max_samples, len(dataset))))
        else:
            self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[idx]


class StreamingHFDataset(Dataset):
    def __init__(self, stream, max_samples: int):
        self.records = []
        for record in stream:
            self.records.append(record)
            if max_samples > 0 and len(self.records) >= max_samples:
                break
        if not self.records:
            raise RuntimeError('no samples were read from the streaming VQA data')

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


class VQANormalizedDataset(Dataset):
    def __init__(self, source: Dataset):
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> dict:
        record = self.source[idx]
        image = extract_vqa_image(record)
        question, answer = extract_vqa_qa(record)
        return {
            "image": ensure_pil_image(image),
            "question": question,
            "answer": answer,
        }


def ensure_pil_image(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, np.ndarray):
        return Image.fromarray(image_obj.astype(np.uint8)).convert("RGB")
    if isinstance(image_obj, dict):
        if "path" in image_obj and image_obj["path"]:
            return Image.open(image_obj["path"]).convert("RGB")
        if "bytes" in image_obj and image_obj["bytes"] is not None:
            from io import BytesIO
            return Image.open(BytesIO(image_obj["bytes"])) .convert("RGB")
    if isinstance(image_obj, str):
        return Image.open(image_obj).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image_obj)}")


def extract_vqa_image(record: dict) -> Any:
    for key in ["image", "images", "img", "image_path"]:
        if key in record:
            value = record[key]
            if isinstance(value, list):
                return value[0]
            return value
    raise KeyError("VQA sample is missing the image/image_path field")


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("value") or item.get("content")
                if text:
                    chunks.append(str(text))
            elif isinstance(item, str):
                chunks.append(item)
        return " ".join(chunks).strip()
    if isinstance(content, dict):
        return str(content.get("text") or content.get("value") or content.get("content") or "")
    return str(content)


def extract_vqa_qa(record: dict) -> tuple[str, str]:
    if "conversations" in record:
        human = None
        assistant = None
        for turn in record["conversations"]:
            who = str(turn.get("from", "")).lower()
            value = _flatten_content(turn.get("value", ""))
            if human is None and who in {"human", "user"} and value:
                human = value
            elif assistant is None and who in {"gpt", "assistant"} and value:
                assistant = value
            if human and assistant:
                return human, assistant
    if "messages" in record:
        human = None
        assistant = None
        for turn in record["messages"]:
            role = str(turn.get("role", "")).lower()
            value = _flatten_content(turn.get("content", ""))
            if human is None and role == "user" and value:
                human = value
            elif assistant is None and role == "assistant" and value:
                assistant = value
            if human and assistant:
                return human, assistant
    for q_key, a_key in [
        ("question", "answer"),
        ("instruction", "response"),
        ("instruction", "output"),
        ("prompt", "completion"),
    ]:
        if q_key in record and a_key in record:
            return str(record[q_key]).strip(), str(record[a_key]).strip()
    if "question" in record and "answers" in record:
        answers = record["answers"]
        if isinstance(answers, list) and answers:
            return str(record["question"]).strip(), str(answers[0]).strip()
    if "caption" in record:
        return "Describe the image.", str(record["caption"]).strip()
    raise KeyError("cannot parse question/answer from VQA sample")


def build_vqa_loader(args, image_processor, tokenizer) -> DataLoader | None:
    if args.lambda_vqa <= 0:
        return None
    try:
        if args.vqa_manifest:
            manifest_path = Path(args.vqa_manifest)
            if not manifest_path.exists():
                raise FileNotFoundError(f"VQA manifest not found: {manifest_path}")
            if manifest_path.suffix == ".jsonl":
                records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                records = payload if isinstance(payload, list) else payload["data"]
            base_dataset = LocalVQADataset(records[: args.vqa_num_samples] if args.vqa_num_samples > 0 else records)
            log(f"[vqa] loaded local manifest {manifest_path} with {len(base_dataset)} samples")
        else:
            if load_dataset is None:
                raise ImportError("the datasets package is required to load Hugging Face VQA data")
            use_streaming = bool(args.vqa_streaming)
            trust_remote_code = bool(args.vqa_trust_remote_code)
            dataset = load_dataset(
                args.vqa_dataset,
                split=args.vqa_split,
                streaming=use_streaming,
                trust_remote_code=trust_remote_code,
            )
            if use_streaming:
                base_dataset = StreamingHFDataset(dataset, args.vqa_num_samples)
            else:
                base_dataset = HFDatasetWrapper(dataset, args.vqa_num_samples)
            log(
                f"[vqa] loaded HF dataset {args.vqa_dataset}:{args.vqa_split} with {len(base_dataset)} samples "
                f"(streaming={use_streaming} trust_remote_code={trust_remote_code})"
            )
    except Exception as exc:
        if not args.allow_vqa_fallback:
            raise
        log(f"[vqa] disabled due to dataset load failure: {exc}")
        log("[vqa] continuing with robot value learning only; set --allow_vqa_fallback=0 to make this fatal")
        return None
    normalized = VQANormalizedDataset(base_dataset)

    def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        proc = image_processor(images=images, return_tensors="pt")
        prompts = [f"Question: {item['question']}\nAnswer:" for item in batch]
        full_texts = [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in batch]
        tok_full = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.vqa_max_length)
        tok_prompt = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.vqa_max_length)
        labels = tok_full.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        prompt_lens = tok_prompt.attention_mask.sum(dim=1)
        for row, plen in enumerate(prompt_lens.tolist()):
            labels[row, :plen] = -100
        return {
            "pixel_values": proc.pixel_values,
            "input_ids": tok_full.input_ids,
            "attention_mask": tok_full.attention_mask,
            "labels": labels,
        }

    return DataLoader(
        normalized,
        batch_size=args.vqa_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )


def tensor_batch(data: dict[str, Any], idx: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    return {
        "obs_feats": data["obs_feats"][idx].to(device).contiguous(),
        "next_obs_feats": data["next_obs_feats"][idx].to(device).contiguous(),
        "rewards": data["rewards"][idx].to(device),
        "dones": data["dones"][idx].to(device),
        "mc_returns": data["mc_returns"][idx].to(device),
        "progress": data["progress"][idx].to(device),
        "success_mask": data["success_mask"][idx].to(device),
        "actions": (data["actions"][idx].to(device).contiguous() if data.get("actions") is not None else None),
        "next_actions": (data["next_actions"][idx].to(device).contiguous() if data.get("next_actions") is not None else None),
        "prompt_texts": [data["prompt_texts"][i] for i in idx.tolist()],
        "task_names": [data["task_names"][i] for i in idx.tolist()],
    }


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def compute_value_losses_from_preds(
    pred1: torch.Tensor,
    pred2: torch.Tensor,
    next_pred: torch.Tensor,
    batch: dict[str, torch.Tensor],
    gamma: float,
    args: Any,
) -> dict[str, torch.Tensor]:
    """Twin VF losses from (pred1, pred2) on obs and detached next_pred = min(Q1(next), Q2(next))."""
    pred_min = torch.minimum(pred1, pred2)
    td_target = batch["rewards"] + gamma * (1 - batch["dones"]) * next_pred
    success_mask = (batch["success_mask"] > 0.5).float()
    prog_target = float(args.alpha_prog) * batch["progress"]
    progress_mask = success_mask if bool(int(getattr(args, "progress_on_success_only", 0))) else torch.ones_like(success_mask)
    label_target = batch["mc_returns"] + prog_target * progress_mask
    success_weights = torch.full_like(pred_min, float(args.success_weight))
    fail_weights = torch.full_like(pred_min, float(args.fail_weight))
    sample_weights = torch.where(success_mask > 0.5, success_weights, fail_weights)
    def twin_weighted_mse(target: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            _weighted_mean((pred1 - target).pow(2), sample_weights)
            + _weighted_mean((pred2 - target).pow(2), sample_weights)
        )

    success_only_weights = sample_weights * success_mask
    progress_weights = sample_weights * progress_mask

    def twin_success_mse(target: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            _weighted_mean((pred1 - target).pow(2), success_only_weights)
            + _weighted_mean((pred2 - target).pow(2), success_only_weights)
        )

    def twin_progress_mse(target: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            _weighted_mean((pred1 - target).pow(2), progress_weights)
            + _weighted_mean((pred2 - target).pow(2), progress_weights)
        )

    return {
        "pred": pred_min,
        "pred1": pred1,
        "pred2": pred2,
        "td_target": td_target,
        "label_target": label_target,
        "prog_target": prog_target,
        "loss_td": twin_weighted_mse(td_target),
        "loss_label": twin_weighted_mse(label_target),
        "loss_mc": twin_weighted_mse(batch["mc_returns"]),
        "loss_prog": twin_progress_mse(prog_target),
        "loss_cons": 0.5 * (
            _weighted_mean(torch.relu(pred1 - batch["mc_returns"]), sample_weights)
            + _weighted_mean(torch.relu(pred2 - batch["mc_returns"]), sample_weights)
        ),
        "loss_cal": 0.5 * (
            _weighted_mean(torch.relu(batch["mc_returns"] - pred1), sample_weights)
            + _weighted_mean(torch.relu(batch["mc_returns"] - pred2), sample_weights)
        ),
        "head_disagreement": _weighted_mean((pred1 - pred2).abs(), sample_weights),
    }


def compute_value_losses(model: RobotValueModel, batch: dict[str, torch.Tensor], gamma: float, beta: float, args) -> dict[str, torch.Tensor]:
    pred1, pred2 = model.predict_value_pair_from_siglip_feats(batch["obs_feats"], prompt_texts=batch["prompt_texts"], actions=batch.get("actions"))
    with torch.no_grad():
        next_pred1, next_pred2 = model.predict_value_pair_from_siglip_feats(batch["next_obs_feats"], prompt_texts=batch["prompt_texts"], actions=batch.get("next_actions"))
        next_pred = torch.minimum(next_pred1, next_pred2)
    sub = {
        "rewards": batch["rewards"],
        "dones": batch["dones"],
        "mc_returns": batch["mc_returns"],
        "progress": batch["progress"],
        "success_mask": batch["success_mask"],
        "task_names": batch["task_names"],
    }
    return compute_value_losses_from_preds(pred1, pred2, next_pred, sub, gamma, args)



@torch.no_grad()
def evaluate_dataset(data: dict[str, Any], model: RobotValueModel, args) -> dict[str, float]:
    model.eval()
    stats = {k: 0.0 for k in [
        "loss_td", "loss_label", "loss_mc", "loss_prog", "loss_cons", "loss_cal", "loss_total", "head_disagreement",
        "value_success", "value_fail", "target_success", "target_fail", "mc_success", "mc_fail", "gap_success", "gap_fail",
        "n_success", "n_fail", "cons_activation_rate", "cal_activation_rate",
    ]}
    total = 0
    for start in range(0, data["obs_feats"].shape[0], args.eval_batch_size):
        idx = torch.arange(start, min(start + args.eval_batch_size, data["obs_feats"].shape[0]))
        batch = tensor_batch(data, idx, args.device)
        B = batch["obs_feats"].shape[0]
        losses = compute_value_losses(model, batch, args.gamma, args.beta, args)
        total += B
        total_loss = (
            args.lambda_td * losses["loss_td"]
            + args.lambda_label * losses["loss_label"]
            + args.lambda_mc * losses["loss_mc"]
            + args.lambda_prog * losses["loss_prog"]
            + (args.lambda_cons * losses["loss_cons"] if args.use_conservative else 0.0)
            + (args.lambda_cal * losses["loss_cal"] if args.use_calibration else 0.0)
        )
        success_mask = batch["success_mask"] > 0.5
        fail_mask = ~success_mask
        pred = losses["pred"]
        if success_mask.any():
            stats["value_success"] += pred[success_mask].sum().item()
            stats["target_success"] += losses["label_target"][success_mask].sum().item()
            stats["mc_success"] += batch["mc_returns"][success_mask].sum().item()
            stats["gap_success"] += (pred[success_mask] - batch["mc_returns"][success_mask]).sum().item()
            stats["n_success"] += float(success_mask.sum().item())
        if fail_mask.any():
            stats["value_fail"] += pred[fail_mask].sum().item()
            stats["target_fail"] += losses["label_target"][fail_mask].sum().item()
            stats["mc_fail"] += batch["mc_returns"][fail_mask].sum().item()
            stats["gap_fail"] += (pred[fail_mask] - batch["mc_returns"][fail_mask]).sum().item()
            stats["n_fail"] += float(fail_mask.sum().item())
        stats["loss_td"] += losses["loss_td"].item() * B
        stats["loss_label"] += losses["loss_label"].item() * B
        stats["loss_mc"] += losses["loss_mc"].item() * B
        stats["loss_prog"] += losses["loss_prog"].item() * B
        stats["loss_cons"] += losses["loss_cons"].item() * B
        stats["loss_cal"] += losses["loss_cal"].item() * B
        stats["loss_total"] += float(total_loss) * B
        stats["head_disagreement"] += losses["head_disagreement"].item() * B
        stats["cons_activation_rate"] += (losses["pred"] > batch["mc_returns"]).float().mean().item() * B
        stats["cal_activation_rate"] += (batch["mc_returns"] > losses["pred"]).float().mean().item() * B

    def div(a: float, b: float) -> float:
        return a / b if b > 0 else 0.0

    return {
        "loss_td": div(stats["loss_td"], total),
        "loss_label": div(stats["loss_label"], total),
        "loss_mc": div(stats["loss_mc"], total),
        "loss_prog": div(stats["loss_prog"], total),
        "loss_cons": div(stats["loss_cons"], total),
        "loss_cal": div(stats["loss_cal"], total),
        "loss_total": div(stats["loss_total"], total),
        "head_disagreement": div(stats["head_disagreement"], total),
        "value_success": div(stats["value_success"], stats["n_success"]),
        "value_fail": div(stats["value_fail"], stats["n_fail"]),
        "target_success": div(stats["target_success"], stats["n_success"]),
        "target_fail": div(stats["target_fail"], stats["n_fail"]),
        "mc_success": div(stats["mc_success"], stats["n_success"]),
        "mc_fail": div(stats["mc_fail"], stats["n_fail"]),
        "gap_success": div(stats["gap_success"], stats["n_success"]),
        "gap_fail": div(stats["gap_fail"], stats["n_fail"]),
        "cons_activation_rate": div(stats["cons_activation_rate"], total),
        "cal_activation_rate": div(stats["cal_activation_rate"], total),
    }


@torch.no_grad()
def plot_trajectory_diagnostics(data: dict[str, Any], model: RobotValueModel, args, output_path: str) -> None:
    success_meta = next((m for m in data["episode_meta"] if m["success"]), None)
    fail_meta = next((m for m in data["episode_meta"] if not m["success"]), None)
    if success_meta is None and fail_meta is None:
        return
    model.eval()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, meta, title in zip(axes, [success_meta, fail_meta], ["Success Trajectory", "Failure Trajectory"], strict=True):
        ax.set_title(title)
        ax.set_xlabel("Timestep")
        if meta is None:
            continue
        sl = slice(meta["start"], meta["end"])
        obs_feats = data["obs_feats"][sl].to(args.device)
        prompt_texts = data["prompt_texts"][sl]
        actions = data.get("actions")
        action_feats = actions[sl].to(args.device) if actions is not None else None
        values = model.predict_value_from_siglip_feats(obs_feats, prompt_texts=prompt_texts, actions=action_feats).cpu().numpy()
        mc = data["mc_returns"][sl].cpu().numpy()
        prog_target = (args.alpha_prog * data["progress"][sl]).cpu().numpy()
        label_target = mc + (prog_target if (meta["success"] or not bool(args.progress_on_success_only)) else 0.0)
        x = np.arange(len(values))
        ax.plot(x, values, label="V(o,l)")
        if args.lambda_mc > 0:
            ax.plot(x, mc, label="MC return")
        ax.plot(x, prog_target, label="alpha_prog * p_t", linestyle=":")
        if args.lambda_mc > 0 and (meta["success"] or not bool(args.progress_on_success_only)):
            ax.plot(x, label_target, label="G_t + alpha_prog * p_t", linestyle="--")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_train_batches(train_data: dict[str, Any], args) -> list[torch.Tensor]:
    n_total = train_data["obs_feats"].shape[0]
    if not args.balance_success_fail:
        perm = torch.randperm(n_total)
        return [perm[start:start + args.batch_size] for start in range(0, n_total, args.batch_size)]

    success_idx = torch.nonzero(train_data["success_mask"] > 0.5, as_tuple=False).flatten()
    fail_idx = torch.nonzero(train_data["success_mask"] <= 0.5, as_tuple=False).flatten()
    if success_idx.numel() == 0 or fail_idx.numel() == 0:
        perm = torch.randperm(n_total)
        return [perm[start:start + args.batch_size] for start in range(0, n_total, args.batch_size)]

    num_batches = math.ceil(n_total / args.batch_size)
    fail_per_batch = max(1, int(round(args.batch_size * args.fail_sample_ratio)))
    fail_per_batch = min(args.batch_size - 1, fail_per_batch)
    success_per_batch = args.batch_size - fail_per_batch
    batches: list[torch.Tensor] = []
    for _ in range(num_batches):
        success_pick = success_idx[torch.randint(len(success_idx), (success_per_batch,))]
        fail_pick = fail_idx[torch.randint(len(fail_idx), (fail_per_batch,))]
        batch = torch.cat([success_pick, fail_pick], dim=0)
        batch = batch[torch.randperm(batch.numel())]
        batches.append(batch)
    return batches

def train_model(train_data: dict[str, Any], val_data: dict[str, Any], model: RobotValueModel, optimizer, vision_tower, image_processor, args) -> tuple[float, dict[str, Any]]:
    best_val_loss = float(getattr(args, "resume_best_val_loss", float("inf")))
    best_state: dict[str, Any] = dict(getattr(args, "resume_best_state", {}) or {})
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    if os.path.exists(metrics_path) and not bool(getattr(args, "resume_append_metrics", False)):
        os.remove(metrics_path)

    vqa_loader = build_vqa_loader(args, image_processor, model.tokenizer)
    vqa_iter = iter(vqa_loader) if vqa_loader is not None else None
    last_val_metrics = dict(getattr(args, "resume_best_state", {}).get("metrics", {}) or {})

    warm_batches = make_train_batches(train_data, args)
    if warm_batches:
        warm_idx = warm_batches[0]
        warm_batch = tensor_batch(train_data, warm_idx, args.device)
        with torch.no_grad():
            _ = compute_value_losses(model, warm_batch, args.gamma, args.beta, args)
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        log(f"[warmup] train batch warmup ok (B={warm_batch['obs_feats'].shape[0]})")

    start_epoch = int(getattr(args, "resume_start_epoch", 1))
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        batches = make_train_batches(train_data, args)
        running = {k: 0.0 for k in ["loss_td", "loss_label", "loss_mc", "loss_prog", "loss_cons", "loss_cal", "loss_total", "loss_vqa", "head_disagreement"]}
        seen = 0
        warmup_only = bool(args.warmup_progress_only) and epoch <= args.progress_warmup_epochs

        for idx in batches:
            batch = tensor_batch(train_data, idx, args.device)
            B = batch["obs_feats"].shape[0]
            seen += B
            losses = compute_value_losses(model, batch, args.gamma, args.beta, args)
            total_loss = (
                args.lambda_label * losses["loss_label"]
                + args.lambda_mc * losses["loss_mc"]
                + args.lambda_prog * losses["loss_prog"]
            )
            if not warmup_only:
                total_loss = total_loss + args.lambda_td * losses["loss_td"]
                if args.use_conservative:
                    total_loss = total_loss + args.lambda_cons * losses["loss_cons"]
                if args.use_calibration:
                    total_loss = total_loss + args.lambda_cal * losses["loss_cal"]

            vqa_loss = torch.zeros((), device=args.device)
            if vqa_loader is not None:
                try:
                    vqa_batch = next(vqa_iter)
                except StopIteration:
                    vqa_iter = iter(vqa_loader)
                    vqa_batch = next(vqa_iter)
                pixel_values = vqa_batch["pixel_values"].to(args.device)
                input_ids = vqa_batch["input_ids"].to(args.device)
                attention_mask = vqa_batch["attention_mask"].to(args.device)
                labels = vqa_batch["labels"].to(args.device)
                with torch.no_grad():
                    vision_outputs = vision_tower(pixel_values=pixel_values)
                    siglip_feats = vision_outputs.last_hidden_state.mean(dim=1).float()
                vqa_loss = model.vqa_loss(siglip_feats, input_ids, attention_mask, labels)
                total_loss = total_loss + args.lambda_vqa * vqa_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            running["loss_td"] += losses["loss_td"].item() * B
            running["loss_label"] += losses["loss_label"].item() * B
            running["loss_mc"] += losses["loss_mc"].item() * B
            running["loss_prog"] += losses["loss_prog"].item() * B
            running["loss_cons"] += losses["loss_cons"].item() * B
            running["loss_cal"] += losses["loss_cal"].item() * B
            running["loss_total"] += total_loss.item() * B
            running["loss_vqa"] += vqa_loss.item() * B
            running["head_disagreement"] += losses["head_disagreement"].item() * B

        train_metrics = {k: v / max(seen, 1) for k, v in running.items()}
        should_eval = (args.eval_every_epochs > 0 and epoch % args.eval_every_epochs == 0) or (epoch == args.epochs)
        if should_eval:
            val_metrics = evaluate_dataset(val_data, model, args)
            last_val_metrics = val_metrics
        else:
            val_metrics = dict(last_val_metrics)
        metrics = {
            "epoch": epoch,
            "train/warmup_only": float(warmup_only),
            "train/fail_weight": float(args.fail_weight),
            "train/success_weight": float(args.success_weight),
            "train/fail_sample_ratio": float(args.fail_sample_ratio),
            "train/balance_success_fail": float(bool(args.balance_success_fail)),
            "train/did_eval": float(should_eval),
        }
        metrics.update({f"train/{k}": float(v) for k, v in train_metrics.items()})
        if should_eval:
            metrics.update({f"val/{k}": float(v) for k, v in val_metrics.items()})
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        log_wandb(metrics, step=epoch)
        if should_eval:
            log(
                "[epoch %3d] train_total=%.4f train_vqa=%.4f val_total=%.4f val_value_success=%.4f val_value_fail=%.4f val_gap_success=%.4f val_gap_fail=%.4f"
                % (
                    epoch,
                    train_metrics["loss_total"],
                    train_metrics["loss_vqa"],
                    val_metrics["loss_total"],
                    val_metrics["value_success"],
                    val_metrics["value_fail"],
                    val_metrics["gap_success"],
                    val_metrics["gap_fail"],
                )
            )
        else:
            log(
                "[epoch %3d] train_total=%.4f train_vqa=%.4f val=skipped"
                % (
                    epoch,
                    train_metrics["loss_total"],
                    train_metrics["loss_vqa"],
                )
            )
        if should_eval:
            if val_metrics["loss_total"] < best_val_loss:
                best_val_loss = val_metrics["loss_total"]
                best_state = {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "metrics": val_metrics,
                    "epoch": epoch,
                }
                log(f"  ★ new best val_total={best_val_loss:.4f}")

        if args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "best_val_loss": best_val_loss,
            }, checkpoint_path)
            log(f"  [checkpoint] saved -> {checkpoint_path}")


    if not best_state:
        best_state = {
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "metrics": {},
            "epoch": max(start_epoch - 1, 0),
        }
    model.load_state_dict(best_state["model"])
    plot_trajectory_diagnostics(val_data, model, args, os.path.join(args.output_dir, "trajectory_value_mc_progress.png"))
    return best_val_loss, best_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitched SigLIP + Gemma value-learning trainer with VQA retention")
    parser.add_argument("--buffer_dir", default="new_data/press_button/buffer_nttg")
    parser.add_argument("--vision_model", default=DEFAULT_VISION_MODEL)
    parser.add_argument("--language_model", default=DEFAULT_LANGUAGE_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output_dir", default="runs/vf_stitched")
    parser.add_argument("--cache_file", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--phase1_batch_size", type=int, default=256)
    parser.add_argument("--projector_micro_batch_size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--progress_warmup_epochs", type=int, default=5)
    parser.add_argument("--lambda_td", type=float, default=1.0)
    parser.add_argument("--lambda_label", type=float, default=1.0)
    parser.add_argument("--lambda_mc", type=float, default=1.0)
    parser.add_argument("--lambda_prog", type=float, default=1.0)
    parser.add_argument("--alpha_prog", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--terminal_success_reward", type=float, default=1.0)
    parser.add_argument("--terminal_failure_reward", type=float, default=0.0)
    parser.add_argument("--progress_on_success_only", type=int, default=1)
    parser.add_argument("--use_conservative", action="store_true")
    parser.add_argument("--lambda_cons", type=float, default=0.1)
    parser.add_argument("--use_calibration", action="store_true")
    parser.add_argument("--lambda_cal", type=float, default=0.1)
    parser.add_argument("--lambda_vqa", type=float, default=0.2)
    parser.add_argument("--success_weight", type=float, default=1.0)
    parser.add_argument("--fail_weight", type=float, default=2.0)
    parser.add_argument("--balance_success_fail", type=int, default=1)
    parser.add_argument("--fail_sample_ratio", type=float, default=0.5)
    parser.add_argument("--warmup_progress_only", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--eval_every_epochs", type=int, default=0)
    parser.add_argument("--save_every_epochs", type=int, default=0)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--allow_vqa_fallback", type=int, default=0)
    parser.add_argument("--vqa_dataset", default=DEFAULT_VQA_DATASET)
    parser.add_argument("--vqa_split", default="train")
    parser.add_argument("--vqa_manifest", default=None)
    parser.add_argument("--vqa_num_samples", type=int, default=20000)
    parser.add_argument("--vqa_batch_size", type=int, default=4)
    parser.add_argument("--vqa_max_length", type=int, default=192)
    parser.add_argument("--vqa_streaming", type=int, default=1)
    parser.add_argument("--vqa_trust_remote_code", type=int, default=0)
    parser.add_argument("--use_action_cond", type=int, default=0, help="1 enables the action-conditioned VF (default 0).")
    parser.add_argument("--use_delta_action", type=int, default=0, help="1 appends delta_action=a_t-a_{t-1} to the action features (requires use_action_cond=1).")
    parser.add_argument("--action_dim", type=int, default=0, help=">0 truncates/zero-pads action vectors to this dimension; 0 uses the data dimension.")
    parser.add_argument("--action_hidden_dim", type=int, default=256, help="Hidden dimension of the action encoder.")
    parser.add_argument("--use_external_wrist_keys", type=int, default=0, help="1 reads the external/wrist observation keys; 0 uses side_policy_256/wrist_1 (default).")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="vf_stitched")
    parser.add_argument("--wandb_run_name", default=None)
    args = parser.parse_args()

    if int(args.use_delta_action) and not int(args.use_action_cond):
        raise ValueError("use_delta_action=1 requires use_action_cond=1")
    if int(args.use_action_cond) and int(args.action_hidden_dim) <= 0:
        raise ValueError("action_hidden_dim must be >0 when use_action_cond=1")

    set_seed(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    wandb_run = init_wandb(args)
    log(f"[train_vf_stitched] device={args.device}")
    if wandb_run is not None:
        log(f"[wandb] project={args.wandb_project} run={wandb_run.name}")
    log(f"  vision_model={args.vision_model}")
    log(f"  language_model={args.language_model}")
    log(f"  prompt={args.prompt}")
    log(f"  vqa_dataset={args.vqa_manifest or args.vqa_dataset}")
    log(
        f"  action_cond={int(args.use_action_cond)} "
        f"delta_action={int(args.use_delta_action)} action_dim={int(args.action_dim)} "
        f"action_hidden_dim={int(args.action_hidden_dim)}"
    )
    log(f"  use_external_wrist_keys={int(args.use_external_wrist_keys)}")

    pkls = sorted(glob.glob(os.path.join(args.buffer_dir, "transitions_*_nttg.pkl")))
    if not pkls:
        pkls = sorted(glob.glob(os.path.join(args.buffer_dir, "transitions_*.pkl")))
    assert pkls, f"no pkl files found: {args.buffer_dir}"

    first_transition = None
    for _p in pkls:
        with open(_p, "rb") as _f:
            _obj = pickle.load(_f)
        if isinstance(_obj, list) and _obj and isinstance(_obj[0], dict):
            first_transition = _obj[0]
            break
    if first_transition is None:
        raise RuntimeError(f"buffer data is empty or malformed: {args.buffer_dir}")
    has_data_prompt = bool(first_transition.get("task_prompt"))
    has_manual_prompt = bool(str(args.prompt).strip())
    if (not has_data_prompt) and (not has_manual_prompt):
        raise ValueError(
            "missing prompt: the data has no task_prompt and --prompt was not given. "
            "Add task_prompt to the data or set PROMPT explicitly at launch."
        )
    train_pkls, val_pkls, test_pkls, split_stats = stratified_split_pkls(pkls, args.val_ratio, args.test_ratio, args.seed)
    n_val = len(val_pkls)
    n_test = len(test_pkls)
    log(f"[data] train={len(train_pkls)} episodes val={len(val_pkls)} episodes test={len(test_pkls)} episodes (task_stratified_random)")
    if split_stats:
        for task, st in sorted(split_stats.items()):
            log(f"  [split] {task}: total={st['total']} train={st['train']} val={st['val']} test={st['test']}")

    log(f"[model] loading SigLIP from {args.vision_model}")
    raw_vision = AutoModel.from_pretrained(args.vision_model, torch_dtype=torch.bfloat16)
    vision_tower = raw_vision.vision_model if hasattr(raw_vision, "vision_model") else raw_vision
    vision_tower = vision_tower.to(args.device)
    for p in vision_tower.parameters():
        p.requires_grad = False
    vision_tower.eval()
    image_processor = AutoImageProcessor.from_pretrained(args.vision_model)
    cfg = getattr(vision_tower, "config", None)
    actual_proj_in = cfg.hidden_size if cfg is not None and hasattr(cfg, "hidden_size") else PROJ_IN

    task_name = infer_task_name(args.buffer_dir)
    cache_file = args.cache_file or os.path.join(args.output_dir, f"{task_name}_siglip_cache_v{CACHE_VERSION}.pt")
    train_data = None
    val_data = None
    test_data = None
    if os.path.exists(cache_file):
        log(f"[Phase 1] loading cache {cache_file}")
        cached = torch.load(cache_file, map_location="cpu", weights_only=True)
        valid = (
            cached.get("cache_version") == CACHE_VERSION
            and abs(float(cached.get("gamma", -1.0)) - args.gamma) < 1e-9
            and int(cached.get("proj_in", -1)) == actual_proj_in
            and bool(cached.get("use_action_cond", False)) == bool(int(args.use_action_cond))
            and bool(cached.get("use_delta_action", False)) == bool(int(args.use_delta_action))
            and int(cached.get("action_dim", 0)) == int(args.action_dim)
            and bool(cached.get("use_external_wrist_keys", False)) == bool(int(args.use_external_wrist_keys))
        )
        if valid:
            try:
                ep_index = _build_episode_index(cached)
                train_data = _rebuild_subset_from_cache(ep_index, train_pkls)
                val_data = _rebuild_subset_from_cache(ep_index, val_pkls) if n_val > 0 else train_data
                test_data = _rebuild_subset_from_cache(ep_index, test_pkls) if n_test > 0 else val_data
                log("[Phase 1] cache hit: rebuilt train/val/test from existing cache with current task-stratified split")
            except Exception as exc:
                log(f"[Phase 1] cache reuse failed, recaching from raw buffer: {exc}")
                train_data = None
                val_data = None
                test_data = None
        else:
            os.remove(cache_file)

    need_cache = train_data is None or (n_val > 0 and val_data is None) or (n_test > 0 and test_data is None)
    if need_cache:
        log("[Phase 1] caching frozen SigLIP pooled features ...")
        train_data = cache_transition_features(train_pkls, vision_tower, image_processor, args.device, args.gamma, actual_proj_in, args.phase1_batch_size, args.prompt, args)
        if n_val > 0:
            val_data = cache_transition_features(val_pkls, vision_tower, image_processor, args.device, args.gamma, actual_proj_in, args.phase1_batch_size, args.prompt, args)
        else:
            val_data = train_data
        if n_test > 0:
            test_data = cache_transition_features(test_pkls, vision_tower, image_processor, args.device, args.gamma, actual_proj_in, args.phase1_batch_size, args.prompt, args)
        else:
            test_data = val_data
        cache_parent = os.path.dirname(os.path.abspath(cache_file))
        if cache_parent:
            os.makedirs(cache_parent, exist_ok=True)
        torch.save({
            "cache_version": CACHE_VERSION,
            "gamma": args.gamma,
            "proj_in": actual_proj_in,
            "use_action_cond": bool(int(args.use_action_cond)),
            "use_delta_action": bool(int(args.use_delta_action)),
            "action_dim": int(args.action_dim),
            "use_external_wrist_keys": bool(int(args.use_external_wrist_keys)),
            "train_data": train_data,
            "val_data": val_data,
            "test_data": test_data,
        }, cache_file)
        log(f"[Phase 1] cache saved -> {cache_file}")
    elif n_val == 0:
        val_data = train_data
        if test_data is None:
            test_data = val_data

    action_feature_dim = 0
    if bool(int(args.use_action_cond)):
        if train_data.get("actions") is None:
            raise ValueError("use_action_cond=1 but cached train_data has no actions")
        action_feature_dim = int(train_data["actions"].shape[-1])

    model = RobotValueModel(
        args.language_model,
        args.prompt,
        vision_dim=actual_proj_in,
        hidden_dim=HIDDEN_DIM,
        projector_micro_batch_size=args.projector_micro_batch_size,
        use_action_cond=bool(int(args.use_action_cond)),
        action_feature_dim=action_feature_dim,
        action_hidden_dim=int(args.action_hidden_dim),
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.resume_start_epoch = 1
    args.resume_best_val_loss = float("inf")
    args.resume_best_state = {}
    args.resume_append_metrics = False

    if args.resume_checkpoint:
        resume_path = args.resume_checkpoint
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(args.output_dir, resume_path)
        log(f"[resume] loading checkpoint {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        args.resume_start_epoch = int(ckpt["epoch"]) + 1
        args.resume_best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        args.resume_best_state = {
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "metrics": ckpt.get("val_metrics", {}),
            "epoch": int(ckpt["epoch"]),
        }
        args.resume_append_metrics = True
        log(f"[resume] start_epoch={args.resume_start_epoch} best_val_loss={args.resume_best_val_loss:.4f}")

    if args.device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        torch.cuda.init()
        with torch.inference_mode():
            warm_feats = torch.zeros(1, actual_proj_in, device=args.device, dtype=torch.float32)
            _ = model.predict_value_from_siglip_feats(warm_feats, prompt_texts=[args.prompt], actions=(torch.zeros(1, action_feature_dim, device=args.device, dtype=torch.float32) if action_feature_dim > 0 else None))
        torch.cuda.synchronize()
        log("[warmup] cuda projector/lm warmup ok")

    best_val_loss, best_state = train_model(train_data, val_data, model, optimizer, vision_tower, image_processor, args)

    test_metrics = {}
    if test_data is not None:
        test_metrics = evaluate_dataset(test_data, model, args)
        log(f"[test] total={test_metrics.get('loss_total', 0.0):.4f} value_success={test_metrics.get('value_success', 0.0):.4f} value_fail={test_metrics.get('value_fail', 0.0):.4f}")

    torch.save(model.projector.state_dict(), os.path.join(args.output_dir, "projector_best.pt"))
    torch.save(model.value_head1.state_dict(), os.path.join(args.output_dir, "value_head1_best.pt"))
    torch.save(model.value_head2.state_dict(), os.path.join(args.output_dir, "value_head2_best.pt"))
    torch.save(model.value_head1.state_dict(), os.path.join(args.output_dir, "value_head_best.pt"))
    torch.save(model.state_dict(), os.path.join(args.output_dir, "stitched_value_model_best.pt"))

    config = {
        "training_mode": "stitched_siglip_gemma_value_vqa",
        "vision_model": args.vision_model,
        "language_model": args.language_model,
        "prompt": args.prompt,
        "task_name": task_name,
        "cache_file": cache_file,
        "proj_in": actual_proj_in,
        "proj_out": HIDDEN_DIM,
        "projector_micro_batch_size": args.projector_micro_batch_size,
        "gamma": args.gamma,
        "progress_warmup_epochs": args.progress_warmup_epochs,
        "lambda_td": args.lambda_td,
        "lambda_label": args.lambda_label,
        "lambda_mc": args.lambda_mc,
        "lambda_prog": args.lambda_prog,
        "alpha_prog": args.alpha_prog,
        "beta": args.beta,
        "terminal_success_reward": args.terminal_success_reward,
        "terminal_failure_reward": args.terminal_failure_reward,
        "progress_on_success_only": bool(args.progress_on_success_only),
        "use_conservative": args.use_conservative,
        "lambda_cons": args.lambda_cons,
        "use_calibration": args.use_calibration,
        "lambda_cal": args.lambda_cal,
        "lambda_vqa": args.lambda_vqa,
        "success_weight": args.success_weight,
        "fail_weight": args.fail_weight,
        "balance_success_fail": bool(args.balance_success_fail),
        "fail_sample_ratio": args.fail_sample_ratio,
        "warmup_progress_only": bool(args.warmup_progress_only),
        "early_stop_patience": args.early_stop_patience,
        "save_every_epochs": args.save_every_epochs,
        "eval_every_epochs": args.eval_every_epochs,
        "test_ratio": args.test_ratio,
        "vqa_streaming": bool(args.vqa_streaming),
        "vqa_trust_remote_code": bool(args.vqa_trust_remote_code),
        "vqa_dataset": args.vqa_manifest or args.vqa_dataset,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_state.get("metrics", {}),
        "test_metrics": test_metrics,
        "split_stats": split_stats,
        "indicator_scoring": "min(V1(o,l,a), V2(o,l,a)) via SigLIP+Gemma latent" if bool(int(args.use_action_cond)) else "min(V1(o,l), V2(o,l)) via SigLIP+Gemma latent",
        "twin_value_heads": True,
        "use_action_cond": bool(int(args.use_action_cond)),
        "use_delta_action": bool(int(args.use_delta_action)),
        "action_dim": int(args.action_dim),
        "action_feature_dim": int(action_feature_dim),
        "action_hidden_dim": int(args.action_hidden_dim),
        "use_external_wrist_keys": bool(int(args.use_external_wrist_keys)),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    plot_path = os.path.join(args.output_dir, "trajectory_value_mc_progress.png")
    finish_wandb({"best_val_loss": best_val_loss, **{f"best/{k}": v for k, v in best_state.get("metrics", {}).items()}}, image_path=plot_path)

    log(f"[done] best_val_loss={best_val_loss:.4f}")
    log(f"[done] projector_best.pt / value_head1_best.pt / value_head2_best.pt / value_head_best.pt / stitched_value_model_best.pt -> {args.output_dir}/")


if __name__ == "__main__":
    main()
