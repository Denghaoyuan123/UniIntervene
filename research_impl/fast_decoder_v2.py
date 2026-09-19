"""FAST tokenization and causal action decoding for VQ2."""

from __future__ import annotations

import math
import hashlib
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FASTBPETokenizer:
    """Thin adapter around the official Physical Intelligence FAST processor."""

    def __init__(self, processor_path: str, action_dim: int, horizon: int):
        self.processor_path = str(processor_path)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.processor = None
        self.bpe_tokenizer = None
        self.scale = None
        self.min_token = None
        from pathlib import Path
        import json
        local = Path(self.processor_path)
        if local.is_dir() and (local / "tokenizer.json").is_file():
            from transformers import PreTrainedTokenizerFast
            self.bpe_tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(local / "tokenizer.json"),
                clean_up_tokenization_spaces=False,
            )
            config = json.loads((local / "processor_config.json").read_text())
            self.scale = float(config["scale"])
            self.min_token = int(config["min_token"])
            expected_vocab = int(config["vocab_size"])
            if len(self.bpe_tokenizer) != expected_vocab:
                raise RuntimeError(
                    f"FAST vocab mismatch: tokenizer={len(self.bpe_tokenizer)} config={expected_vocab}"
                )
            self.asset_sha256 = {
                name: hashlib.sha256((local / name).read_bytes()).hexdigest()
                for name in ("tokenizer.json", "processor_config.json")
            }
        else:
            raise FileNotFoundError(
                "FAST processor must be a local directory containing tokenizer.json "
                "and processor_config.json"
            )
        self.action_vocab_size = int(len(self.bpe_tokenizer))
        self.pad_id = self.action_vocab_size
        self.eos_id = self.action_vocab_size + 1
        self.model_vocab_size = self.action_vocab_size + 2

    def _fixed_horizon(self, trajectory: np.ndarray) -> np.ndarray:
        x = np.asarray(trajectory, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.action_dim or x.shape[0] == 0:
            raise ValueError(
                f"expected nonempty [T,{self.action_dim}] trajectory, got {x.shape}"
            )
        out = np.empty((self.horizon, self.action_dim), dtype=np.float32)
        n = min(self.horizon, x.shape[0])
        out[:n] = x[:n]
        out[n:] = x[n - 1]
        return out

    def encode(self, trajectory: np.ndarray) -> np.ndarray:
        fixed = self._fixed_horizon(trajectory)
        if self.processor is not None:
            encoded = self.processor(fixed[None, ...])
            ids = encoded[0] if isinstance(encoded, (list, tuple)) else encoded
        else:
            from scipy.fft import dct
            coefficients = np.around(dct(fixed, axis=0, norm="ortho") * self.scale)
            text = "".join(
                map(chr, np.maximum(coefficients.flatten() - self.min_token, 0).astype(int))
            )
            ids = self.bpe_tokenizer(text)["input_ids"]
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise RuntimeError("FAST processor returned an empty token sequence")
        if ids.min() < 0 or ids.max() >= self.action_vocab_size:
            raise RuntimeError(
                f"FAST token id outside [0,{self.action_vocab_size}): "
                f"min={ids.min()} max={ids.max()}"
            )
        return np.concatenate([ids, np.asarray([self.eos_id], dtype=np.int64)])

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        ids = np.asarray(tokens, dtype=np.int64).reshape(-1)
        ids = ids[(ids != self.pad_id) & (ids != self.eos_id)]
        if ids.size == 0:
            raise ValueError("no FAST action tokens to decode")
        if self.processor is not None:
            actions = self.processor.decode(
                [ids.tolist()], time_horizon=self.horizon, action_dim=self.action_dim
            )[0]
        else:
            from scipy.fft import idct
            text = self.bpe_tokenizer.decode(ids.tolist())
            coefficients = np.asarray(list(map(ord, text)), dtype=np.float32) + self.min_token
            expected = self.horizon * self.action_dim
            if coefficients.size != expected:
                raise ValueError(
                    f"decoded FAST coefficients={coefficients.size}, expected={expected}"
                )
            coefficients = coefficients.reshape(self.horizon, self.action_dim)
            actions = idct(coefficients / self.scale, axis=0, norm="ortho")
        return np.asarray(actions, dtype=np.float32)

    def state_dict(self) -> dict:
        from pathlib import Path
        return {
            "kind": "physical-intelligence/fast",
            "processor_name": Path(self.processor_path).name,
            "asset_sha256": dict(self.asset_sha256),
            "action_dim": self.action_dim,
            "horizon": self.horizon,
            "action_vocab_size": self.action_vocab_size,
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
        }

    @classmethod
    def from_state_dict(
        cls, state: dict, *, processor_path: str | None = None
    ) -> "FASTBPETokenizer":
        resolved_path = processor_path or state.get("processor_path")
        if not resolved_path:
            raise ValueError("processor_path is required to restore the FAST tokenizer")
        obj = cls(
            processor_path=resolved_path,
            action_dim=int(state["action_dim"]),
            horizon=int(state["horizon"]),
        )
        if obj.action_vocab_size != int(state["action_vocab_size"]):
            raise RuntimeError("FAST processor vocabulary changed since checkpoint creation")
        expected_hashes = state.get("asset_sha256")
        if expected_hashes and dict(expected_hashes) != obj.asset_sha256:
            raise RuntimeError("FAST tokenizer assets do not match the checkpoint")
        return obj



def _dct_matrix(T: int) -> np.ndarray:
    """Return orthonormal DCT-II matrix M of shape [T, T] such that"""
    n = np.arange(T)[None, :]   # [1, T]   sample index
    k = np.arange(T)[:, None]   # [T, 1]   freq index
    M = np.cos(np.pi * (2 * n + 1) * k / (2 * T))  # [T, T]
    M[0, :] *= 1.0 / math.sqrt(T)
    M[1:, :] *= math.sqrt(2.0 / T)
    return M.astype(np.float32)



class FastTokenizer:
    """DCT-II + per-coefficient scalar quantization."""

    def __init__(self, action_dim: int, max_T: int, n_bins: int = 256,
                 clip_quantile: float = 0.995):
        self.action_dim = int(action_dim)
        self.max_T = int(max_T)
        self.n_bins = int(n_bins)
        self.clip_quantile = float(clip_quantile)
        self.lo: Optional[np.ndarray] = None   # [max_T, D]
        self.hi: Optional[np.ndarray] = None   # [max_T, D]
        self._M: np.ndarray = _dct_matrix(self.max_T)  # [T, T]


    def fit(self, trajectories: Iterable[np.ndarray]):
        coefs = []
        for traj in trajectories:
            arr = np.asarray(traj, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] == 0:
                continue
            padded = self._pad_or_truncate(arr)
            coefs.append(self._M @ padded)  # [T, D]
        if not coefs:
            raise ValueError("FastTokenizer.fit() got no usable trajectories")
        stacked = np.stack(coefs, axis=0)   # [N, T, D]
        q = self.clip_quantile
        self.lo = np.quantile(stacked, 1.0 - q, axis=0).astype(np.float32)
        self.hi = np.quantile(stacked, q,        axis=0).astype(np.float32)
        span = self.hi - self.lo
        self.hi = np.where(span < 1e-6, self.lo + 1.0, self.hi)

    def _pad_or_truncate(self, traj: np.ndarray) -> np.ndarray:
        T_in, D = traj.shape
        T = self.max_T
        if T_in == T:
            return traj
        out = np.zeros((T, D), dtype=np.float32)
        n = min(T_in, T)
        out[:n] = traj[:n]
        if T_in < T:
            out[n:] = traj[n - 1:n]  # pad with last action (constant hold)
        return out


    def encode(self, traj: np.ndarray) -> np.ndarray:
        assert self.lo is not None, "FastTokenizer not fitted"
        arr = np.asarray(traj, dtype=np.float32)
        T_in, D = arr.shape
        assert D == self.action_dim, f"action_dim mismatch: {D} vs {self.action_dim}"
        padded = self._pad_or_truncate(arr)
        coefs = self._M @ padded                                  # [T, D]
        norm = (coefs - self.lo) / (self.hi - self.lo + 1e-8)
        norm = np.clip(norm, 0.0, 1.0)
        bins = np.minimum((norm * self.n_bins).astype(np.int64),
                          self.n_bins - 1)                        # [T, D]
        return bins

    def decode(self, tokens: np.ndarray, T_actual: Optional[int] = None) -> np.ndarray:
        assert self.lo is not None, "FastTokenizer not fitted"
        arr = np.asarray(tokens, dtype=np.int64)
        T_in, D = arr.shape
        assert D == self.action_dim
        T = self.max_T
        bins = np.zeros((T, D), dtype=np.int64)
        n = min(T_in, T)
        bins[:n] = arr[:n]
        norm = (bins.astype(np.float32) + 0.5) / float(self.n_bins)
        coefs = norm * (self.hi - self.lo) + self.lo              # [T, D]
        traj = self._M.T @ coefs                                  # iDCT
        if T_actual is not None:
            return traj[:int(T_actual)]
        return traj[:n] if n > 0 else traj


    def state_dict(self) -> dict:
        return {
            "action_dim": self.action_dim,
            "max_T": self.max_T,
            "n_bins": self.n_bins,
            "clip_quantile": self.clip_quantile,
            "lo": self.lo.tolist() if self.lo is not None else None,
            "hi": self.hi.tolist() if self.hi is not None else None,
        }

    @classmethod
    def from_state_dict(cls, sd: dict) -> "FastTokenizer":
        tk = cls(action_dim=int(sd["action_dim"]),
                 max_T=int(sd["max_T"]),
                 n_bins=int(sd["n_bins"]),
                 clip_quantile=float(sd.get("clip_quantile", 0.995)))
        if sd.get("lo") is not None:
            tk.lo = np.asarray(sd["lo"], dtype=np.float32)
            tk.hi = np.asarray(sd["hi"], dtype=np.float32)
        return tk



class _CausalSelfAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int):
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


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, max_len: int):
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


class FASTBPEActionDecoder(nn.Module):
    """Causal action-token head for variable-length official FAST BPE tokens."""

    def __init__(
        self,
        backbone_hidden: int,
        vocab_size: int,
        pad_id: int,
        eos_id: int,
        max_tokens: int = 128,
        d_dec: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.eos_id = int(eos_id)
        self.max_tokens = int(max_tokens)
        self.d_dec = int(d_dec)
        self.ctx_proj = nn.Sequential(nn.Linear(backbone_hidden, d_dec), nn.Tanh())
        self.tok_emb = nn.Embedding(self.vocab_size, d_dec, padding_idx=self.pad_id)
        self.pos_emb = nn.Embedding(self.max_tokens + 1, d_dec)
        self.blocks = nn.ModuleList(
            [_Block(d_dec, n_heads, ffn_mult, self.max_tokens + 1) for _ in range(n_layers)]
        )
        self.ln_out = nn.LayerNorm(d_dec)
        self.head = nn.Linear(d_dec, self.vocab_size)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_context(self, backbone_hidden: torch.Tensor) -> torch.Tensor:
        context = self.ctx_proj(backbone_hidden.float())
        pos = self.pos_emb(torch.zeros(context.shape[0], 1, dtype=torch.long, device=context.device))
        return context.unsqueeze(1) + pos

    def _embed(self, tokens: torch.Tensor, start_pos: int = 1) -> torch.Tensor:
        batch, length = tokens.shape
        pos = torch.arange(start_pos, start_pos + length, device=tokens.device)
        pos = pos.unsqueeze(0).expand(batch, -1)
        return self.tok_emb(tokens) + self.pos_emb(pos)

    def decode(self, context: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Teacher forcing; targets is padded [B,N], including EOS."""
        batch, length = targets.shape
        if length > self.max_tokens:
            raise ValueError(f"FAST target length {length} exceeds max_tokens={self.max_tokens}")
        prior = self._embed(targets[:, :-1], 1) if length > 1 else context[:, :0]
        sequence = torch.cat([context, prior], dim=1)
        for block in self.blocks:
            sequence = block(sequence)
        return self.head(self.ln_out(sequence))

    @torch.no_grad()
    def generate(self, context: torch.Tensor) -> list[np.ndarray]:
        batch = context.shape[0]
        sequence = context
        outputs: list[torch.Tensor] = []
        finished = torch.zeros(batch, dtype=torch.bool, device=context.device)
        for index in range(self.max_tokens):
            hidden = sequence
            for block in self.blocks:
                hidden = block(hidden)
            token = self.head(self.ln_out(hidden[:, -1])).argmax(-1)
            token = torch.where(finished, torch.full_like(token, self.pad_id), token)
            outputs.append(token)
            finished |= token.eq(self.eos_id)
            if bool(finished.all()):
                break
            sequence = torch.cat([sequence, self._embed(token[:, None], index + 1)], dim=1)
        stacked = torch.stack(outputs, dim=1).cpu().numpy()
        result = []
        for row in stacked:
            eos = np.where(row == self.eos_id)[0]
            result.append(row[: int(eos[0]) if eos.size else len(row)])
        return result


def fast_bpe_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    label_smoothing: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Per-token classification loss on BPE action tokens and EOS."""
    vocab = logits.shape[-1]
    flat_targets = targets.reshape(-1)
    ce = F.cross_entropy(
        logits.reshape(-1, vocab), flat_targets,
        ignore_index=int(pad_id), label_smoothing=float(label_smoothing),
    )
    with torch.no_grad():
        mask = targets.ne(int(pad_id))
        correct = logits.argmax(-1).eq(targets) & mask
        accuracy = correct.sum().float() / mask.sum().clamp_min(1)
        token_count = mask.sum()
    return {"ce": ce, "acc": accuracy, "token_count": token_count}



class FastDecoderV2(nn.Module):
    """Token-based AR decoder."""

    def __init__(
        self,
        backbone_hidden: int,
        action_dim: int,
        n_bins: int,
        max_T: int,
        d_dec: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.n_bins = int(n_bins)
        self.max_T = int(max_T)
        self.d_dec = int(d_dec)
        self.max_tokens = self.max_T * self.action_dim

        max_seq = self.max_tokens + 1  # +1 for context token

        self.ctx_proj = nn.Sequential(
            nn.Linear(backbone_hidden, d_dec, bias=True),
            nn.Tanh(),
        )
        self.tok_emb = nn.Embedding(self.n_bins, d_dec)
        self.pos_emb = nn.Embedding(max_seq, d_dec)
        self.dim_emb = nn.Embedding(self.action_dim, d_dec)

        self.blocks = nn.ModuleList([
            _Block(d_dec, n_heads, ffn_mult, max_len=max_seq)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_dec)
        self.head = nn.Linear(d_dec, self.n_bins, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def encode_context(self, backbone_hidden: torch.Tensor) -> torch.Tensor:
        """[B, H] → [B, 1, d_dec] (with position-0 embedding added)."""
        ctx = self.ctx_proj(backbone_hidden.float())                  # [B, d_dec]
        pos0 = self.pos_emb(torch.zeros(ctx.shape[0], 1,
                                        dtype=torch.long,
                                        device=ctx.device))           # [B, 1, d_dec]
        return ctx.unsqueeze(1) + pos0                                # [B, 1, d_dec]


    def _embed_token_seq(self, tokens: torch.Tensor, start_pos: int = 1) -> torch.Tensor:
        """tokens: [B, N] long → [B, N, d_dec] with bin + position + dim embedding."""
        B, N = tokens.shape
        device = tokens.device
        emb = self.tok_emb(tokens)                                    # [B, N, d_dec]
        pos = self.pos_emb(torch.arange(start_pos, start_pos + N,
                                        device=device).unsqueeze(0).expand(B, -1))
        dim_idx = torch.arange(N, device=device) % self.action_dim
        dim = self.dim_emb(dim_idx).unsqueeze(0).expand(B, -1, -1)
        return emb + pos + dim


    def decode(self, context: torch.Tensor, gt_tokens: torch.Tensor) -> torch.Tensor:
        """context : [B, 1, d_dec] from encode_context()"""
        B, T, D = gt_tokens.shape
        assert D == self.action_dim
        N = T * D
        flat = gt_tokens.reshape(B, N)
        tok_emb = self._embed_token_seq(flat[:, :N - 1], start_pos=1) if N > 1 else \
            torch.zeros(B, 0, self.d_dec, device=context.device, dtype=context.dtype)
        seq = torch.cat([context, tok_emb], dim=1)                     # [B, N, d_dec]
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.ln_out(seq)
        return self.head(seq)                                          # [B, N, n_bins]


    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        T: int,
        *,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """context : [B, 1, d_dec]"""
        B = context.shape[0]
        device = context.device
        D = self.action_dim
        N = T * D
        assert N <= self.max_tokens

        seq = context                                                  # [B, 1, d_dec]
        out_tokens: list[torch.Tensor] = []

        for i in range(N):
            x = seq
            for blk in self.blocks:
                x = blk(x)
            x = self.ln_out(x)
            logits = self.head(x[:, -1, :])                            # [B, n_bins]
            if temperature > 0.0:
                probs = torch.softmax(logits / float(temperature), dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tok = logits.argmax(dim=-1)
            out_tokens.append(next_tok)
            if i == N - 1:
                break
            emb = self._embed_token_seq(next_tok.unsqueeze(1), start_pos=i + 1)
            seq = torch.cat([seq, emb], dim=1)

        flat = torch.stack(out_tokens, dim=1)                          # [B, N]
        return flat.reshape(B, T, D)



def fast_decoder_v2_ce_loss(
    logits: torch.Tensor,
    gt_tokens: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    per_dim_weight: Optional[torch.Tensor] = None,
    per_step_weight: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """logits : [B, N, n_bins] from FastDecoderV2.decode()"""
    B, T, D = gt_tokens.shape
    N = T * D
    n_bins = logits.size(-1)
    gt_flat = gt_tokens.reshape(B, N)
    logits_flat = logits.reshape(B, N, n_bins)

    ce_per_tok = F.cross_entropy(
        logits_flat.reshape(-1, n_bins),
        gt_flat.reshape(-1),
        label_smoothing=label_smoothing,
        reduction="none",
    ).reshape(B, T, D)  # [B, T, D]

    weight = torch.ones_like(ce_per_tok)
    if per_dim_weight is not None:
        weight = weight * per_dim_weight.to(weight.device).view(1, 1, D)
    if per_step_weight is not None:
        weight = weight * per_step_weight.to(weight.device).view(1, T, 1)

    ce = (ce_per_tok * weight).sum() / weight.sum().clamp_min(1e-6)

    with torch.no_grad():
        pred = logits_flat.argmax(dim=-1).reshape(B, T, D)
        acc = (pred == gt_tokens).float().mean()
        per_step_ce = ce_per_tok.mean(dim=(0, 2))    # [T]
        per_dim_ce = ce_per_tok.mean(dim=(0, 1))     # [D]

    return {
        "ce": ce,
        "acc": acc.detach(),
        "per_step_ce": per_step_ce.detach(),
        "per_dim_ce": per_dim_ce.detach(),
    }



def build_fast_decoder_v2_gt_tokens(
    sample: dict,
    tokenizer: FastTokenizer,
    a_min: "np.ndarray",
    a_max: "np.ndarray",
    n_bins_act: int,
    device: "torch.device",
    *,
    s_mean: "np.ndarray | None" = None,
    s_std: "np.ndarray | None" = None,
    clip_k: float = 4.0,
) -> "tuple[torch.Tensor, int] | None":
    """Parse GT trajectory from sample and tokenize via DCT+quantize."""
    from fast_decoder import build_fast_decoder_gt

    gt_cont = build_fast_decoder_gt(
        sample, a_min, a_max, n_bins_act, device,
        s_mean=s_mean, s_std=s_std, clip_k=clip_k,
    )
    if gt_cont is None or int(gt_cont.shape[0]) <= 0:
        return None
    arr = gt_cont.detach().cpu().numpy()  # [T, D]
    T_actual = int(arr.shape[0])
    bins = tokenizer.encode(arr)  # [max_T, D] returns full padded
    bins = bins[:T_actual]
    return torch.as_tensor(bins, dtype=torch.long, device=device), T_actual


def build_fast_bpe_gt_tokens(
    sample: dict,
    tokenizer: FASTBPETokenizer,
    a_min: "np.ndarray",
    a_max: "np.ndarray",
    n_bins_act: int,
    device: "torch.device",
    *,
    s_mean: "np.ndarray | None" = None,
    s_std: "np.ndarray | None" = None,
    clip_k: float = 4.0,
) -> "torch.Tensor | None":
    """Build the official FAST BPE target sequence, including EOS."""
    from fast_decoder import build_fast_decoder_gt

    trajectory = build_fast_decoder_gt(
        sample, a_min, a_max, n_bins_act, device,
        s_mean=s_mean, s_std=s_std, clip_k=clip_k,
    )
    if trajectory is None or int(trajectory.shape[0]) == 0:
        return None
    array = trajectory.detach().cpu().numpy().astype(np.float32, copy=False)
    if array.shape[1] != tokenizer.action_dim:
        raise ValueError(
            f"FAST action dim mismatch: trajectory={array.shape[1]} tokenizer={tokenizer.action_dim}"
        )
    ids = tokenizer.encode(array)
    return torch.as_tensor(ids, dtype=torch.long, device=device)
