"""FAST+ decoder head for VQ2."""

from typing import TYPE_CHECKING

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    import numpy as np


class _CausalSelfAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int = 256):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = attn.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class _DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, max_len: int = 256):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _CausalSelfAttn(d_model, n_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult, bias=False),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class FastDecoder(nn.Module):
    """Parameters"""

    def __init__(
        self,
        backbone_hidden: int,
        action_dim: int,
        d_dec: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        max_traj_len: int = 128,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.d_dec = int(d_dec)
        self.max_traj_len = int(max_traj_len)

        self.ctx_proj = nn.Sequential(
            nn.Linear(backbone_hidden, d_dec, bias=True),
            nn.Tanh(),
        )

        self.act_in = nn.Linear(action_dim, d_dec, bias=True)

        self.pos_emb = nn.Embedding(max_traj_len + 1, d_dec)  # +1 for context position 0

        self.blocks = nn.ModuleList([
            _DecoderBlock(d_dec, n_heads, max_len=max_traj_len + 1)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_dec)

        self.act_out = nn.Linear(d_dec, action_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode_context(self, backbone_hidden: torch.Tensor) -> torch.Tensor:
        """backbone_hidden: [B, H] — pooled last-token hidden from Qwen prompt-only forward."""
        ctx = self.ctx_proj(backbone_hidden.float())  # [B, d_dec]
        pos0 = self.pos_emb(
            torch.zeros(ctx.shape[0], 1, dtype=torch.long, device=ctx.device)
        )  # [B, 1, d_dec]
        return ctx.unsqueeze(1) + pos0  # [B, 1, d_dec]

    def decode(
        self,
        context: torch.Tensor,
        traj_gt: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced decoding for training."""
        B, T, _ = traj_gt.shape
        device = context.device

        act_emb = self.act_in(traj_gt.float())

        step_pos = torch.arange(1, T + 1, dtype=torch.long, device=device).unsqueeze(0)
        act_emb = act_emb + self.pos_emb(step_pos)  # [B, T, d_dec]

        seq = torch.cat([context, act_emb[:, :-1, :]], dim=1)  # [B, T, d_dec]  (drop last step input)

        for blk in self.blocks:
            seq = blk(seq)
        seq = self.ln_out(seq)  # [B, T, d_dec]

        return self.act_out(seq)  # [B, T, action_dim]

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        n_steps: int,
    ) -> torch.Tensor:
        """Autoregressive generation at inference."""
        B = context.shape[0]
        device = context.device
        T = min(int(n_steps), self.max_traj_len)

        tokens = context  # [B, 1, d_dec]
        preds: list[torch.Tensor] = []

        for t in range(T):
            seq = tokens
            for blk in self.blocks:
                seq = blk(seq)
            seq = self.ln_out(seq)
            last_pred = self.act_out(seq[:, -1, :])  # [B, action_dim]
            preds.append(last_pred)

            next_emb = self.act_in(last_pred)
            pos_t = self.pos_emb(
                torch.full((B, 1), t + 1, dtype=torch.long, device=device)
            )
            next_tok = (next_emb + pos_t[:, 0, :]).unsqueeze(1)  # [B, 1, d_dec]
            tokens = torch.cat([tokens, next_tok], dim=1)

        return torch.stack(preds, dim=1)  # [B, T, action_dim]



def fast_decoder_losses(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    tail_k: int = 3,
    margin: float = 0.0,
    end_weight: float = 0.15,
    tail_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Compute FAST decoder losses directly in action space."""
    B, T, D = pred.shape
    losses: dict[str, torch.Tensor] = {}

    l1 = F.l1_loss(pred, gt, reduction="mean")
    losses["l1"] = l1

    if end_weight > 0.0:
        l_end = F.smooth_l1_loss(pred[:, -1, :], gt[:, -1, :], reduction="mean", beta=0.2)
        losses["end"] = l_end
    else:
        losses["end"] = pred.new_zeros(())

    if tail_weight > 0.0 and T > 1:
        k = min(int(tail_k), T - 1)
        tail_pred = pred[:, -k-1:, :]  # [B, k+1, D]
        goal = gt[:, -1:, :].expand_as(tail_pred)
        d = torch.linalg.norm(tail_pred - goal, ord=2, dim=-1)  # [B, k+1]
        l_tail = torch.relu(d[:, 1:] - d[:, :-1] + float(margin)).mean()
        losses["tail"] = l_tail
    else:
        losses["tail"] = pred.new_zeros(())

    return losses


def fast_decoder_total_loss(
    losses: dict[str, torch.Tensor],
    *,
    l1_weight: float = 1.0,
    end_weight: float = 0.15,
    tail_weight: float = 0.05,
) -> torch.Tensor:
    return (
        l1_weight * losses["l1"]
        + end_weight * losses["end"]
        + tail_weight * losses["tail"]
    )


def build_fast_decoder_gt(
    sample: dict,
    a_min: "np.ndarray",
    a_max: "np.ndarray",
    n_bins: int,
    device: "torch.device",
    *,
    s_mean: "np.ndarray | None" = None,
    s_std: "np.ndarray | None" = None,
    clip_k: float = 4.0,
) -> "torch.Tensor | None":
    """Return the continuous Round2 target trajectory."""
    import numpy as np
    from hil_vlm_train import _parse_target_trajectory_from_sample, _dequantize_bins_action, _dequantize_bins_state_norm

    space = str(sample.get("round2_target_space", "action")).strip().lower()

    if space == "action":
        action_traj = sample.get("round2_action_traj")
        if action_traj is not None:
            arr = np.asarray(action_traj, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != 7:
                raise ValueError(
                    f"round2_action_traj must be nonempty [T,7], got {arr.shape}"
                )
            if not np.isfinite(arr).all():
                raise ValueError("round2_action_traj contains non-finite values")
            return torch.as_tensor(arr, device=device)

    if space == "state11_pose":
        pose_traj = sample.get("round2_pose_traj")
        if pose_traj is None:
            return None
        arr = np.asarray(pose_traj, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 7 or arr.shape[0] == 0:
            return None
        return torch.as_tensor(arr, device=device)

    gt_bins = _parse_target_trajectory_from_sample(sample)  # [T, D] int64 or None
    if gt_bins is None or int(gt_bins.shape[0]) <= 0:
        return None
    T, D = gt_bins.shape

    if space == "state":
        gt_cont = _dequantize_bins_state_norm(
            gt_bins.reshape(-1).astype(np.float64), clip_k, n_bins
        ).reshape(T, D).astype(np.float32)
    else:
        lo = np.asarray(a_min[:D], dtype=np.float32)
        hi = np.asarray(a_max[:D], dtype=np.float32)
        gt_cont = _dequantize_bins_action(gt_bins.reshape(-1), lo, hi, n_bins).reshape(T, D).astype(np.float32)

    return torch.as_tensor(gt_cont, device=device)
