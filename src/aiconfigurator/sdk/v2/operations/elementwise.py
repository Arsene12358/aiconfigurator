# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aiconfigurator.sdk.v2.estimator import Estimator
from aiconfigurator.sdk.v2.operation_v2 import OperationV2

logger = logging.getLogger(__name__)


class ElementWiseV2(OperationV2):
    """Element-wise (memory-bound) operation.

    Lookup key: ``(total_bytes,)`` — modelled as a memory operation.
    """

    def __init__(
        self,
        backend_context: Any,
        dim_in: int,
        dim_out: int,
        estimator: Estimator | None = None,
        scale_factor: float = 1.0,
        scale_num_tokens: int = 1,
    ) -> None:
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._scale_num_tokens = scale_num_tokens
        super().__init__(backend_context, estimator, scale_factor)

    def load_data(self) -> tuple[dict[tuple, dict[str, float]], np.ndarray | None]:
        return {}, None

    def get_lookup_key(self, **kwargs) -> tuple:
        x = kwargs["x"] // self._scale_num_tokens
        read_bytes = x * self._dim_in * 2   # fp16
        write_bytes = x * self._dim_out * 2
        return (read_bytes + write_bytes,)
