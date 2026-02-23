# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.v2.estimator import Estimator
from aiconfigurator.sdk.v2.operation_v2 import OperationV2

logger = logging.getLogger(__name__)


class P2PV2(OperationV2):
    """Point-to-point communication operation.

    Lookup key: ``(p2p_bytes,)``.
    """

    def __init__(
        self,
        backend_context: Any,
        h: int,
        pp_size: int,
        estimator: Estimator | None = None,
        scale_factor: float = 1.0,
    ) -> None:
        self._h = h
        self._pp_size = pp_size
        super().__init__(backend_context, estimator, scale_factor)

    def load_data(self) -> tuple[dict[tuple, dict[str, float]], np.ndarray | None]:
        return {}, None

    def get_lookup_key(self, **kwargs) -> tuple:
        size = kwargs["x"] * self._h
        p2p_bytes = size * 2
        return (p2p_bytes,)

    def query(self, **kwargs) -> PerformanceResult:
        if self._pp_size == 1:
            return PerformanceResult(0.0, 0.0)
        return super().query(**kwargs)
