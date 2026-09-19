from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch


TensorLike = torch.Tensor | np.ndarray


def _to_tensor(x: Optional[TensorLike], dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


@dataclass
class QueryBuilderConfig:
    l2_normalize: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryQueryBuilder:
    """Build normalized retrieval embeddings for observation-instruction contexts."""

    def __init__(self, config: QueryBuilderConfig | None = None) -> None:
        self.config = config or QueryBuilderConfig()

    def _prepare_feature(
        self,
        feature: Optional[TensorLike],
        batch_size: int | None = None,
    ) -> Optional[torch.Tensor]:
        feature = _to_tensor(feature)
        if feature is None:
            return None
        if feature.dim() == 1:
            feature = feature.unsqueeze(0)
        if batch_size is not None and feature.size(0) != batch_size:
            raise ValueError(
                f"Feature batch size mismatch: expected {batch_size}, got {feature.size(0)}"
            )
        return feature

    def build_batch_query(
        self,
        failure_embedding: TensorLike,
    ) -> torch.Tensor:
        failure_embedding = self._prepare_feature(failure_embedding)
        assert failure_embedding is not None
        query = failure_embedding
        if self.config.l2_normalize:
            query = l2_normalize(query)
        return query

    def build_query(
        self,
        failure_embedding: TensorLike,
    ) -> torch.Tensor:
        return self.build_batch_query(failure_embedding=failure_embedding)[0]
