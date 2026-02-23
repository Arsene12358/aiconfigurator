# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aiconfigurator.sdk.v2.estimator import Estimator
from aiconfigurator.sdk.v2.operation_v2 import OperationV2

logger = logging.getLogger(__name__)


class EmbeddingV2(OperationV2):
    """Embedding table lookup (memory-bound).

    Lookup key: ``(d2d_bytes,)`` — modelled as a memory operation.
    """

    def __init__(
        self,
        backend_context: Any,
        row_size: int,
        column_size: int,
        estimator: Estimator | None = None,
        scale_factor: float = 1.0,
    ) -> None:
        self._row_size = row_size
        self._column_size = column_size
        self._weights = row_size * column_size * 2  # fp16
        super().__init__(backend_context, estimator, scale_factor)

    def load_data(self) -> tuple[dict[tuple, dict[str, float]], np.ndarray | None]:
        return {}, None

    def get_lookup_key(self, **kwargs) -> tuple:
        x = kwargs["x"]
        d2d_bytes = x * self._column_size * 2
        return (d2d_bytes,)

    def get_weights(self, **kwargs) -> float:
        return self._weights * self.scale_factor
