"""VQ1 unified architecture pieces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


VQ_TOKEN = "<VQ_QUERY>"




def add_vq_query_token(tokenizer, model) -> int:
    """Register `<VQ_QUERY>` once; resize embeddings if newly added. Returns id."""
    existing = tokenizer.convert_tokens_to_ids(VQ_TOKEN)
    unk = getattr(tokenizer, "unk_token_id", None)
    if existing is None or existing == unk:
        n_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [VQ_TOKEN]}
        )
        if n_added > 0:
            base = getattr(model, "base_model", None)
            target = base.model if base is not None and hasattr(base, "model") else model
            target.resize_token_embeddings(len(tokenizer))
    return int(tokenizer.convert_tokens_to_ids(VQ_TOKEN))


def extract_h_t(
    outputs,
    input_ids: torch.Tensor,
    vq_token_id: int,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Hidden state at the last occurrence of <VQ_QUERY> in each row."""
    hs = outputs.hidden_states[-1]                              # [B, T, H]
    B, T, _ = hs.shape
    matches = input_ids == int(vq_token_id)                     # [B, T]
    has = matches.any(dim=1)
    if not bool(has.all()):
        missing = torch.where(~has)[0].detach().cpu().tolist()
        raise RuntimeError(f"<VQ_QUERY> is missing from batch rows {missing}")
    pos = (matches.float().cumsum(dim=1) * matches.float()).argmax(dim=1)
    idx = torch.arange(B, device=hs.device)
    return hs[idx, pos].contiguous()                            # [B, H]




class FutureHead(nn.Module):
    """h_t -> z_future_pred."""

    def __init__(self, hidden_size: int, latent_dim: int, mode: str, n_tokens: int = 0):
        super().__init__()
        mode = str(mode).lower()
        assert mode in ("pooled", "dense"), f"future_latent_mode={mode}"
        self.mode = mode
        self.n_tokens = int(n_tokens) if mode == "dense" else 0
        self.latent_dim = int(latent_dim)
        out_dim = self.latent_dim if mode == "pooled" else self.latent_dim * self.n_tokens
        self.proj = nn.Sequential(
            nn.Linear(int(hidden_size), 4 * int(hidden_size)),
            nn.GELU(),
            nn.Linear(4 * int(hidden_size), out_dim),
        )
        self._init()

    def _init(self):
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        z = self.proj(h_t)
        if self.mode == "dense":
            return z.view(z.size(0), self.n_tokens, self.latent_dim)
        return z


class AttnPool(nn.Module):
    """Single-query attention pooling over a [B, N, D] sequence."""

    def __init__(self, d_in: int, n_heads: int = 4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_in) * 0.02)
        self.attn = nn.MultiheadAttention(d_in, n_heads, batch_first=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            return z
        q = self.q.expand(z.size(0), -1, -1)
        out, _ = self.attn(q, z, z)
        return out.squeeze(1)


class TwinQHead(nn.Module):
    """[B, D] or [B, N, D] -> (Q1, Q2). Both heads regress to proxy_V(t+1)."""

    def __init__(self, in_dim: int, accepts_dense: bool = False):
        super().__init__()
        self.accepts_dense = bool(accepts_dense)
        self.pool = AttnPool(in_dim) if accepts_dense else None
        self.q1 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, 1))
        self.q2 = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, 1))
        for head in (self.q1, self.q2):
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if feat.dim() == 3:
            assert self.accepts_dense and self.pool is not None
            feat = self.pool(feat)
        return self.q1(feat).squeeze(-1), self.q2(feat).squeeze(-1)


class InterventionHead(nn.Module):
    """[B, D] (+ optional Q_min scalar) -> logit."""

    def __init__(self, in_dim: int, use_q_cond: bool, accepts_dense: bool = False):
        super().__init__()
        self.use_q_cond = bool(use_q_cond)
        self.accepts_dense = bool(accepts_dense)
        self.pool = AttnPool(in_dim) if accepts_dense else None
        d = in_dim + (1 if self.use_q_cond else 0)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * in_dim),
            nn.GELU(),
            nn.Linear(4 * in_dim, 1),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor, q_min: Optional[torch.Tensor] = None) -> torch.Tensor:
        if feat.dim() == 3:
            assert self.accepts_dense and self.pool is not None
            feat = self.pool(feat)
        if self.use_q_cond:
            assert q_min is not None, "intervention_head needs Q_min when use_q_cond=True"
            feat = torch.cat([feat, q_min.detach().to(dtype=feat.dtype).unsqueeze(-1)], dim=-1)
        return self.mlp(feat).squeeze(-1)


class ValueHistoryEncoder(nn.Module):
    """MLP encoder for value history [V_{t-K:t}, ΔV_{t-K+1:t}] -> e_hist [B, embed_dim]."""

    def __init__(self, K: int, embed_dim: int = 32):
        super().__init__()
        self.K = int(K)
        self.embed_dim = int(embed_dim)
        input_dim = 2 * self.K + 1
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hist_input: torch.Tensor) -> torch.Tensor:
        """hist_input: [B, 2K+1] -> [B, embed_dim]"""
        return self.mlp(hist_input.float())


class VCurrHead(nn.Module):
    """h_t -> V_curr_pred scalar [B]. Regresses proxy V_t (current_value mode aux loss)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        return self.mlp(h_t.float()).squeeze(-1)  # [B]


class RiskHead(nn.Module):
    """[z_feature, context] -> risk_pred scalar [B] (continuous TVR regression)."""

    def __init__(self, z_dim: int, context_dim: int, accepts_dense: bool = False):
        super().__init__()
        self.accepts_dense = bool(accepts_dense)
        self.pool = AttnPool(z_dim) if accepts_dense else None
        in_dim = int(z_dim) + int(context_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 4 * z_dim),
            nn.GELU(),
            nn.Linear(4 * z_dim, 1),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_feat: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """z_feat: [B, D] or [B, N, D]; context: [B, C] -> risk_pred [B]"""
        if z_feat.dim() == 3:
            assert self.accepts_dense and self.pool is not None
            z_feat = self.pool(z_feat)
        feat = torch.cat([z_feat.float(), context.float()], dim=-1)
        return self.mlp(feat).squeeze(-1)




class _OnlineVJEPA2Encoder(nn.Module):
    """Frozen V-JEPA2 encoder used to produce future-latent targets."""

    def __init__(self, path: str, device, dtype=torch.bfloat16, crop_size: int = 256):
        super().__init__()
        from transformers import AutoModel

        self.device = device
        self.dtype = dtype
        self.crop_size = int(crop_size)
        mdl = AutoModel.from_pretrained(path, torch_dtype=dtype, trust_remote_code=False)
        self.encoder = mdl
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval().to(device)
        cfg = getattr(self.encoder, "config", None)
        self.hidden_dim = int(getattr(cfg, "hidden_size", 1024) or 1024)
        self.tubelet_size = int(getattr(cfg, "tubelet_size", 2) or 2)

    def _prepare_clip(self, view_uint8_hwc: np.ndarray) -> torch.Tensor:
        """[H,W,3] uint8 -> [1, T=tubelet, 3, crop, crop] bf16 in [0,1]."""
        arr = np.asarray(view_uint8_hwc)
        if arr.ndim == 4:
            arr = arr[0]                                        # accept (1,H,W,3)
        x = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
        x = x.permute(2, 0, 1).contiguous()                     # [3, H, W]
        if x.shape[-1] != self.crop_size or x.shape[-2] != self.crop_size:
            x = F.interpolate(x.unsqueeze(0), size=(self.crop_size, self.crop_size),
                              mode="bilinear", align_corners=False).squeeze(0)
        x_t = x.unsqueeze(0).expand(self.tubelet_size, -1, -1, -1).contiguous()
        return x_t.unsqueeze(0).to(self.device, dtype=self.dtype)

    @torch.no_grad()
    def encode_views(self, views_per_sample: list[list[np.ndarray]], mode: str) -> Optional[torch.Tensor]:
        """views_per_sample : len B; each element is list of uint8 [H,W,3] arrays."""
        if not views_per_sample:
            return None
        clips = []
        per_sample_counts = []
        for views in views_per_sample:
            views = list(views or [])
            per_sample_counts.append(len(views))
            for v in views:
                if v is None:
                    continue
                clips.append(self._prepare_clip(v))
        if not clips:
            return None
        batch = torch.cat(clips, dim=0)                         # [Nflat, T, 3, H, W]
        out = self.encoder(pixel_values_videos=batch)
        last = getattr(out, "last_hidden_state", None)
        if last is None:
            last = out[0] if isinstance(out, (tuple, list)) else out
        D = int(last.shape[-1])
        targets = []
        cursor = 0
        for c in per_sample_counts:
            if c <= 0:
                targets.append(torch.zeros(1, D, device=last.device, dtype=last.dtype))
                continue
            row = last[cursor:cursor + c]                       # [c, N, D]
            cursor += c
            row = row.reshape(-1, D)                            # concat views along token axis
            targets.append(row)
        if mode == "pooled":
            return torch.stack([t.mean(dim=0) for t in targets], dim=0)             # [B, D]
        n_max = max(t.shape[0] for t in targets)
        out_t = torch.zeros(len(targets), n_max, D, device=last.device, dtype=last.dtype)
        for i, t in enumerate(targets):
            out_t[i, : t.shape[0]] = t
        return out_t                                            # [B, N, D]


class _PrecomputedJEPALatent:
    """Loads latent_<stem>.npz per buffer episode and returns z_{t+1}."""

    _POOLED_MEAN_KEYS  = ("z_tp1_mean", "z_tp1_target_mean")
    _POOLED_TOKEN_KEYS = ("z_tp1_tokens", "z_tp1", "z_tp1_target", "z_tp1_target_tokens")
    _DENSE_KEYS        = ("z_tp1_tokens", "z_tp1", "z_tp1_target", "z_tp1_target_tokens")

    def __init__(
        self,
        latent_dir: str,
        manifest_path: Optional[str] = None,
        mode: str = "pooled",
    ):
        import json as _json

        self.latent_dir = str(latent_dir)
        self._mode = str(mode).lower()
        self._cache: dict[str, Optional[dict]] = {}
        self._mapping: dict[str, str] = {}
        self._cache_count: int = 0
        self._cache_bytes: int = 0
        if manifest_path:
            from pathlib import Path

            mp = Path(manifest_path)
            if mp.is_file():
                payload = _json.loads(mp.read_text(encoding="utf-8"))
                eps = payload.get("episodes", []) if isinstance(payload, dict) else []
                for r in eps:
                    if isinstance(r, dict) and r.get("episode_file") and r.get("latent_file"):
                        self._mapping[str(r["episode_file"])] = str(r["latent_file"])
        print(
            f"[precomputed_jepa] init  mode={self._mode}"
            f"  latent_dir={self.latent_dir}"
            f"  manifest_entries={len(self._mapping)}",
            flush=True,
        )

    def _resolve(self, episode_stem: str) -> Optional[str]:
        import os as _os

        ep_file_pkl = f"{episode_stem}.pkl"
        if ep_file_pkl in self._mapping:
            return _os.path.join(self.latent_dir, self._mapping[ep_file_pkl])
        cand = _os.path.join(self.latent_dir, f"latent_{episode_stem}.npz")
        if _os.path.isfile(cand):
            return cand
        return None

    def _select_keys(self, available: list) -> list:
        """Return only the array keys we need to read for self._mode."""
        if self._mode == "pooled":
            mean_keys = [k for k in self._POOLED_MEAN_KEYS if k in available]
            if mean_keys:
                return mean_keys
            for k in self._POOLED_TOKEN_KEYS:
                if k in available:
                    return [k]
            return []
        for k in self._DENSE_KEYS:
            if k in available:
                return [k]
        return []

    def _load(self, episode_stem: str) -> Optional[dict]:
        if episode_stem in self._cache:
            return self._cache[episode_stem]
        path = self._resolve(episode_stem)
        if path is None:
            self._cache[episode_stem] = None
            return None
        with np.load(path) as npz:
            keys = self._select_keys(list(npz.files))
            payload = {k: np.asarray(npz[k]) for k in keys} if keys else {}
        if not payload:
            self._cache[episode_stem] = None
            return None
        ep_bytes = sum(a.nbytes for a in payload.values())
        n_before = self._cache_count
        self._cache[episode_stem] = payload
        self._cache_bytes += ep_bytes
        self._cache_count += 1
        if n_before == 0:
            print(
                f"[precomputed_jepa] first_load  stem={episode_stem}"
                f"  keys={list(payload)}  {ep_bytes / 1e6:.2f} MB/episode",
                flush=True,
            )
        elif self._cache_count % 100 == 0:
            print(
                f"[precomputed_jepa] cache  episodes={self._cache_count}"
                f"  total={self._cache_bytes / 1e6:.0f} MB"
                f"  avg={self._cache_bytes / self._cache_count / 1e6:.2f} MB/ep",
                flush=True,
            )
        return payload

    def cache_info(self) -> dict:
        return {
            "episodes_cached": self._cache_count,
            "episodes_missing": sum(1 for v in self._cache.values() if v is None),
            "cache_mb": round(self._cache_bytes / 1e6, 1),
        }

    def get_next(self, episode_stem: str, step_in_ep: int, mode: str,
                 device, dtype=torch.float32) -> Optional[torch.Tensor]:
        payload = self._load(episode_stem)
        if not payload:
            return None
        if mode == "pooled":
            for k in self._POOLED_MEAN_KEYS:
                if k in payload:
                    arr = payload[k]
                    if step_in_ep >= int(arr.shape[0]):
                        return None
                    v = torch.from_numpy(arr[int(step_in_ep)]).to(device=device, dtype=dtype)
                    return v.unsqueeze(0)                                            # [1, D]
            for k in self._POOLED_TOKEN_KEYS:
                if k in payload:
                    arr = payload[k]
                    if step_in_ep >= int(arr.shape[0]):
                        return None
                    v = torch.from_numpy(arr[int(step_in_ep)])
                    if v.ndim == 2:
                        v = v.mean(dim=0)
                    return v.to(device=device, dtype=dtype).unsqueeze(0)
            return None
        for k in self._DENSE_KEYS:
            if k in payload:
                arr = payload[k]
                if step_in_ep >= int(arr.shape[0]):
                    return None
                v = torch.from_numpy(arr[int(step_in_ep)])
                if v.ndim == 1:
                    v = v.unsqueeze(0)
                return v.to(device=device, dtype=dtype).unsqueeze(0)
        return None


class FutureTargetEncoder(nn.Module):
    """Unified frozen future-latent target wrapper supporting an online V-JEPA2 encoder or precomputed JEPA latents."""

    def __init__(
        self,
        source: str,
        device,
        dtype=torch.bfloat16,
        vjepa_path: Optional[str] = None,
        crop_size: int = 256,
        latent_dir: Optional[str] = None,
        latent_manifest: Optional[str] = None,
        latent_mode: str = "pooled",
    ):
        super().__init__()
        self.source = str(source).lower()
        self.device = device
        self.dtype = dtype
        self.online = None
        self.precomputed = None
        self.hidden_dim = 0
        self.n_tokens = 0

        if self.source == "online_vjepa2":
            if not vjepa_path:
                print("[future_target] online_vjepa2 requires --vjepa_path; encoder disabled.")
                return
            try:
                self.online = _OnlineVJEPA2Encoder(vjepa_path, device=device, dtype=dtype, crop_size=crop_size)
                self.hidden_dim = int(self.online.hidden_dim)
            except Exception as exc:
                print(f"[future_target] failed to load V-JEPA2 from {vjepa_path}: {exc}")
                self.online = None
        elif self.source == "precomputed_jepa":
            if not latent_dir:
                print("[future_target] precomputed_jepa requires --jepa_latent_dir; encoder disabled.")
                return
            self.precomputed = _PrecomputedJEPALatent(latent_dir, latent_manifest, mode=latent_mode)
        else:
            print(f"[future_target] unknown source={source}; encoder disabled.")

    @property
    def available(self) -> bool:
        return (self.online is not None) or (self.precomputed is not None)

    @torch.no_grad()
    def compute_target(
        self,
        mode: str,
        views_per_sample: Optional[list[list[np.ndarray]]] = None,
        episode_stem: Optional[str] = None,
        step_in_ep: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if self.online is not None and views_per_sample is not None:
            out = self.online.encode_views(views_per_sample, mode=mode)
            if out is not None:
                if self.hidden_dim == 0:
                    self.hidden_dim = int(out.shape[-1])
                if mode == "dense" and self.n_tokens == 0:
                    self.n_tokens = int(out.shape[1])
            return out
        if self.precomputed is not None and episode_stem is not None and step_in_ep is not None:
            out = self.precomputed.get_next(episode_stem, int(step_in_ep), mode=mode,
                                            device=self.device, dtype=torch.float32)
            if out is not None:
                if self.hidden_dim == 0:
                    self.hidden_dim = int(out.shape[-1])
                if mode == "dense" and self.n_tokens == 0:
                    self.n_tokens = int(out.shape[1])
            return out
        return None




def normalized_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L2-normalize along the last dim, then MSE. Makes pooled & dense modes"""
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    return ((p - t) ** 2).sum(dim=-1).mean()


def cosine_sim_mean(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    return (p * t).sum(dim=-1).mean()


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary focal loss with sigmoid logits."""
    logits = logits.float()
    targets = targets.float()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    a_t = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
    loss = a_t * (1.0 - p_t).clamp(min=1e-7).pow(float(gamma)) * ce
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    return loss.mean()


def threshold_sweep_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> dict:
    """Sweep classification thresholds and return best-F1 metrics + AUPRC."""
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probs.size == 0 or labels.size == 0:
        return {
            "best_f1": 0.0, "best_threshold": 0.5,
            "precision_at_best": 0.0, "recall_at_best": 0.0,
            "trigger_at_best": 0.0, "auprc": 0.0, "n_pos": 0, "n_total": 0,
        }
    best = {"best_f1": -1.0, "best_threshold": 0.5,
            "precision_at_best": 0.0, "recall_at_best": 0.0,
            "trigger_at_best": 0.0}
    for t in thresholds:
        pred = (probs > float(t)).astype(np.float64)
        tp = float(((pred == 1) & (labels == 1)).sum())
        fp = float(((pred == 1) & (labels == 0)).sum())
        fn = float(((pred == 0) & (labels == 1)).sum())
        p = tp / max(1.0, tp + fp)
        r = tp / max(1.0, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r) if (p + r) > 0 else 0.0
        if f1 > best["best_f1"]:
            best.update({
                "best_f1": float(f1),
                "best_threshold": float(t),
                "precision_at_best": float(p),
                "recall_at_best": float(r),
                "trigger_at_best": float(pred.mean()),
            })
    pr = []
    rc = []
    for t in np.concatenate([[0.0], thresholds, [1.0]]):
        pred = (probs > float(t)).astype(np.float64)
        tp = float(((pred == 1) & (labels == 1)).sum())
        fp = float(((pred == 1) & (labels == 0)).sum())
        fn = float(((pred == 0) & (labels == 1)).sum())
        p = tp / max(1.0, tp + fp)
        r = tp / max(1.0, tp + fn)
        pr.append(p); rc.append(r)
    rc = np.asarray(rc); pr = np.asarray(pr)
    order = np.argsort(rc)
    rc_s = rc[order]; pr_s = pr[order]
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    auprc = float(_trapz(pr_s, rc_s)) if (_trapz is not None and rc_s.size >= 2) else 0.0
    best["auprc"] = max(0.0, auprc)
    best["n_pos"] = int(labels.sum())
    best["n_total"] = int(labels.size)
    if best["best_f1"] < 0:
        best["best_f1"] = 0.0
    return best




@dataclass
class VariantSpec:
    use_future_head: bool       # build / use future_head
    use_future_loss: bool       # backprop L_future
    use_q_head: bool            # build / use twin_Q_head
    use_q_loss: bool            # backprop L_Q
    use_int_head: bool          # build / use intervention_head
    use_int_loss: bool          # backprop L_intervention
    q_input: str                # "z" or "h"
    int_input_z: str            # "z" or "h"
    int_use_q_cond: bool        # whether intervention concatenates Q_min


VARIANT_SPEC: dict[str, VariantSpec] = {
    "full": VariantSpec(True, True, True, True, True, True, "z", "z", True),
    "no_future": VariantSpec(False, False, True, True, True, True, "h", "h", True),
    "no_value_cond": VariantSpec(True, True, True, True, True, True, "z", "z", False),
    "no_future_value_cond": VariantSpec(False, False, True, True, True, True, "h", "h", False),
    "no_intervention": VariantSpec(True, True, True, True, False, False, "z", "z", False),
    "no_future_loss": VariantSpec(True, False, True, True, True, True, "z", "z", True),
}


def get_variant(name: str) -> VariantSpec:
    key = str(name).strip().lower()
    if key not in VARIANT_SPEC:
        raise SystemExit(
            f"unknown variant={name}. choices: {sorted(VARIANT_SPEC.keys())}"
        )
    return VARIANT_SPEC[key]
