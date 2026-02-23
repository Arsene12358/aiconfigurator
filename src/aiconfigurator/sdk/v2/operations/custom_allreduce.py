# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aiconfigurator.sdk import common
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.v2.estimator import Estimator
from aiconfigurator.sdk.v2.operation_v2 import OperationV2

logger = logging.getLogger(__name__)


class CustomAllReduceV2(OperationV2):
    """Custom AllReduce operation.

    Lookup key: ``(quant_mode, tp_size, message_size)``.
    """

    def __init__(
        self,
        backend_context: Any,
        h: int,
        tp_size: int,
        estimator: Estimator | None = None,
        scale_factor: float = 1.0,
    ) -> None:
        self._h = h
        self._tp_size = tp_size
        super().__init__(backend_context, estimator, scale_factor)

    def load_data(self) -> tuple[dict[tuple, dict[str, float]], np.ndarray | None]:
        return {}, None

    def get_lookup_key(self, **kwargs) -> tuple:
        size = kwargs["x"] * self._h
        return (common.CommQuantMode.half.name, self._tp_size, size)

    def query(self, **kwargs) -> PerformanceResult:
        if self._tp_size == 1:
            return PerformanceResult(0.0, 0.0)
        return super().query(**kwargs)
