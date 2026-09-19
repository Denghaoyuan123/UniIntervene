from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from memory.query_builder import MemoryQueryBuilder, QueryBuilderConfig


def _tensor_to_list(x: Optional[torch.Tensor]) -> Optional[list[float]]:
    if x is None:
        return None
    return x.detach().cpu().float().tolist()


def _ensure_tensor(x: Optional[torch.Tensor | np.ndarray | Sequence[float]]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    return torch.as_tensor(x, dtype=torch.float32)


@dataclass
class MemoryItem:
    task_text: str
    progress_value: float
    failure_embedding: torch.Tensor
    target_embedding: torch.Tensor
    failure_index: int = -1
    target_index: int = -1
    episode_id: str = ""
    success_score: float = 0.0
    task_id: Optional[str] = None
    action_chunk: Optional[torch.Tensor] = None
    recovery_action_trajectory: Optional[torch.Tensor] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_serializable(self) -> dict[str, Any]:
        data = asdict(self)
        data["failure_embedding"] = _tensor_to_list(self.failure_embedding)
        data["target_embedding"] = _tensor_to_list(self.target_embedding)
        data["action_chunk"] = _tensor_to_list(self.action_chunk)
        data["recovery_action_trajectory"] = _tensor_to_list(self.recovery_action_trajectory)
        return data

    @classmethod
    def from_serializable(cls, data: dict[str, Any]) -> "MemoryItem":
        data = dict(data)

        alias_task_name = data.pop("task_name", None)
        if alias_task_name is not None and not str(data.get("task_text", "")).strip():
            data["task_text"] = str(alias_task_name)

        if data.get("progress_value", None) is None:
            pv = None
            meta = data.get("metadata")
            if isinstance(meta, dict):
                for k in ("progress_value", "vf_progress_value", "vf_ref_value"):
                    if meta.get(k) is not None:
                        pv = meta.get(k)
                        break
            if pv is None:
                for k in ("progress_pos", "vf_anchor"):
                    if data.get(k) is not None:
                        pv = data.get(k)
                        break
            try:
                data["progress_value"] = float(pv) if pv is not None else 0.0
            except Exception:
                data["progress_value"] = 0.0

        if data.get("failure_embedding") is None:
            for k in ("z_failure", "failure_emb_1d", "failure_emb", "failure_feature", "failure_latent"):
                if data.get(k) is not None:
                    data["failure_embedding"] = data.get(k)
                    break

        if data.get("target_embedding") is None:
            for k in ("z_target", "target_emb_1d", "target_emb", "recovery_target_embedding"):
                if data.get(k) is not None:
                    data["target_embedding"] = data.get(k)
                    break
            if data.get("target_embedding") is None and data.get("failure_embedding") is not None:
                data["target_embedding"] = data.get("failure_embedding")

        if data.get("recovery_action_trajectory") is None:
            for k in ("z_recovery_segment", "recovery_segment"):
                if data.get(k) is not None:
                    data["recovery_action_trajectory"] = data.get(k)
                    break

        data["failure_embedding"] = _ensure_tensor(data.get("failure_embedding"))
        data["target_embedding"] = _ensure_tensor(data.get("target_embedding"))
        data.pop("task_embedding", None)
        data.pop("trend_embedding", None)
        data["action_chunk"] = _ensure_tensor(data.get("action_chunk"))
        if "recovery_action_trajectory" not in data:
            data["recovery_action_trajectory"] = None
        data["recovery_action_trajectory"] = _ensure_tensor(data.get("recovery_action_trajectory"))

        known = {f.name for f in fields(cls)}
        unknown = {k: data.pop(k) for k in list(data.keys()) if k not in known}
        if unknown:
            meta = data.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
            for k, v in unknown.items():
                meta.setdefault(k, v)
            data["metadata"] = meta

        if data.get("failure_embedding") is None:
            raise ValueError("MemoryItem missing failure_embedding after schema mapping")
        if data.get("target_embedding") is None:
            raise ValueError("MemoryItem missing target_embedding after schema mapping")

        return cls(**data)


def memory_item_recovery_steps(item: MemoryItem) -> int:
    """Return the number of action steps stored in a memory item."""
    t = item.recovery_action_trajectory
    if t is not None and t.numel() > 0:
        if t.dim() == 1:
            return 1
        return int(t.shape[0])
    c = item.action_chunk
    if c is not None and c.numel() > 0:
        if c.dim() == 1:
            return 1
        return int(c.shape[0])
    return 0


@dataclass
class RetrievalResult:
    indices: torch.Tensor
    scores: torch.Tensor
    target_embeddings: torch.Tensor
    items: list[list[MemoryItem]]


class MemoryBank:
    """Simple in-memory top-k cosine retrieval over mined recovery targets."""

    def __init__(
        self,
        items: Optional[list[MemoryItem]] = None,
        query_builder: Optional[MemoryQueryBuilder] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.items: list[MemoryItem] = items or []
        self.query_builder = query_builder or MemoryQueryBuilder(QueryBuilderConfig())
        self.device = torch.device(device)
        self.key_tensor: Optional[torch.Tensor] = None
        self.target_tensor: Optional[torch.Tensor] = None

        if self.items:
            self.finalize()

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)
        self.key_tensor = None
        self.target_tensor = None

    def extend(self, items: list[MemoryItem]) -> None:
        self.items.extend(items)
        self.key_tensor = None
        self.target_tensor = None

    def finalize(self) -> None:
        if not self.items:
            self.key_tensor = None
            self.target_tensor = None
            return

        keys = []
        targets = []
        for item in self.items:
            key = self.query_builder.build_query(item.failure_embedding)
            keys.append(key)
            targets.append(_ensure_tensor(item.target_embedding))

        self.key_tensor = torch.stack(keys, dim=0).to(self.device)
        self.target_tensor = torch.stack(targets, dim=0).to(self.device)

    def save(self, path: str | Path) -> None:
        self.finalize()
        payload: dict[str, Any] = {
            "schema_version": 2,
            "items": [item.to_serializable() for item in self.items],
            "query_builder_config": self.query_builder.config.to_dict(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "MemoryBank":
        payload = torch.load(path, map_location=map_location, weights_only=True)
        _ = int(payload.get("schema_version", 1))
        items = [MemoryItem.from_serializable(x) for x in payload["items"]]
        qb_cfg = payload.get("query_builder_config")
        if qb_cfg is None:
            qb_cfg = payload.get("query_config")
        if isinstance(qb_cfg, dict):
            config = QueryBuilderConfig(**qb_cfg)
        else:
            config = QueryBuilderConfig()
        bank = cls(
            items=items,
            query_builder=MemoryQueryBuilder(config),
            device=str(map_location),
        )
        bank.finalize()
        return bank

    def _retrieve_single(
        self,
        query: torch.Tensor,
        topk: int,
        task_filter: Optional[str] = None,
        exclude_episode_id: Optional[str] = None,
        exclude_failure_index: Optional[int] = None,
        *,
        min_recovery_steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, list[MemoryItem]]:
        if self.key_tensor is None or self.target_tensor is None:
            self.finalize()

        if self.key_tensor is None or len(self.items) == 0:
            raise RuntimeError("MemoryBank is empty; add items before retrieval.")

        valid_indices = list(range(len(self.items)))
        if task_filter is not None:
            valid_indices = [
                i
                for i, item in enumerate(self.items)
                if item.task_text == task_filter or item.task_id == task_filter
            ]
            if not valid_indices:
                raise RuntimeError(
                    "Memory retrieval: no candidates match task_filter "
                    f"{task_filter!r}; strict mode does not fallback to all tasks."
                )

        if exclude_episode_id is not None:
            ex_ep = str(exclude_episode_id)
            valid_indices = [
                i
                for i in valid_indices
                if str(self.items[i].episode_id) != ex_ep
            ]
        if not valid_indices:
            raise RuntimeError(
                "Memory retrieval: no candidates left after excluding episode "
                f"{exclude_episode_id!r}; add more train episodes to the bank."
            )

        min_rs = int(min_recovery_steps)
        if min_rs > 1:
            valid_indices = [
                i
                for i in valid_indices
                if memory_item_recovery_steps(self.items[i]) >= min_rs
            ]
        if not valid_indices:
            raise RuntimeError(
                "Memory retrieval: no candidates left after min_recovery_steps="
                f"{min_rs} (raise memory_top_k or set min_recovery_steps=1)."
            )

        valid_idx_tensor = torch.as_tensor(valid_indices, dtype=torch.long, device=self.device)
        key_subset = self.key_tensor.index_select(0, valid_idx_tensor)

        scores = torch.matmul(key_subset, query.to(self.device))
        k = min(topk, key_subset.size(0))
        top_scores, local_indices = torch.topk(scores, k=k, dim=0)
        global_indices = valid_idx_tensor[local_indices]
        items = [self.items[i] for i in global_indices.tolist()]
        return global_indices, top_scores, items

    def retrieve(
        self,
        query: torch.Tensor,
        topk: int = 1,
        task_filter: Optional[str] = None,
        exclude_episode_id: Optional[str] = None,
        exclude_failure_index: Optional[int] = None,
        *,
        min_recovery_steps: int = 1,
    ) -> RetrievalResult:
        if query.dim() != 1:
            raise ValueError(f"query must have shape [D], got {tuple(query.shape)}")
        indices, scores, items = self._retrieve_single(
            query,
            topk=topk,
            task_filter=task_filter,
            exclude_episode_id=exclude_episode_id,
            exclude_failure_index=exclude_failure_index,
            min_recovery_steps=min_recovery_steps,
        )
        target_embeddings = self.target_tensor.index_select(0, indices)
        return RetrievalResult(
            indices=indices.unsqueeze(0),
            scores=scores.unsqueeze(0),
            target_embeddings=target_embeddings.unsqueeze(0),
            items=[items],
        )

    def batch_retrieve(
        self,
        queries: torch.Tensor,
        topk: int = 1,
        task_filter: Optional[Sequence[Optional[str]] | str] = None,
    ) -> RetrievalResult:
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)
        if queries.dim() != 2:
            raise ValueError(
                f"queries must have shape [B, D] or [D], got {tuple(queries.shape)}"
            )

        if isinstance(task_filter, str) or task_filter is None:
            task_filters = [task_filter] * queries.size(0)
        else:
            task_filters = list(task_filter)
            if len(task_filters) != queries.size(0):
                raise ValueError(
                    f"task_filter length mismatch: expected {queries.size(0)}, got {len(task_filters)}"
                )

        all_indices = []
        all_scores = []
        all_targets = []
        all_items: list[list[MemoryItem]] = []
        for query, task_name in zip(queries, task_filters):
            indices, scores, items = self._retrieve_single(
                query=query,
                topk=topk,
                task_filter=task_name,
                exclude_episode_id=None,
                exclude_failure_index=None,
                min_recovery_steps=1,
            )
            all_indices.append(indices)
            all_scores.append(scores)
            all_targets.append(self.target_tensor.index_select(0, indices))
            all_items.append(items)

        return RetrievalResult(
            indices=torch.stack(all_indices, dim=0),
            scores=torch.stack(all_scores, dim=0),
            target_embeddings=torch.stack(all_targets, dim=0),
            items=all_items,
        )
