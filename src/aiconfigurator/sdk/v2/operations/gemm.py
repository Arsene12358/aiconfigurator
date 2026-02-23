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


class GemmV2(OperationV2):
    """GEMM operation – self-contained data loading and querying.

    Stores its own ``(quant_mode, M, N, K) → {latency, energy}`` lookup and
    an optional ``Estimator`` for interpolation.

    Args:
        backend_context: Backend context (carries ``data_dir`` etc.).
        n: GEMM N dimension (weight rows).
        k: GEMM K dimension (weight cols).
        quant_mode: Quantization mode for this GEMM.
        estimator: Optional fallback estimator.
        scale_factor: Latency / energy multiplier.
        scale_num_tokens: Divisor applied to the input token count.
    """

    def __init__(
        self,
        backend_context: Any,
        n: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        estimator: Estimator | None = None,
        scale_factor: float = 1.0,
        scale_num_tokens: int = 1,
        low_precision_input: bool = False,
    ) -> None:
        self._n = n
        self._k = k
        self._quant_mode = quant_mode
        self._scale_num_tokens = scale_num_tokens
        self._low_precision_input = low_precision_input
        self._weights = n * k * quant_mode.value.memory
        super().__init__(backend_context, estimator, scale_factor)

    def load_data(self) -> tuple[dict[tuple, dict[str, float]], np.ndarray | None]:
        # Data loading is delegated to the V1 PerfDatabase during the
        # migration period.  Concrete loaders will be added as the flat
        # file-loading logic is migrated.
        return {}, None

    def get_lookup_key(self, **kwargs) -> tuple:
        m = kwargs["M"] // self._scale_num_tokens
        return (self._quant_mode.name, m, self._n, self._k)

    def query(self, **kwargs) -> PerformanceResult:
        key = self.get_lookup_key(**kwargs)
        if key in self.exact_match_dict:
            entry = self.exact_match_dict[key]
            return PerformanceResult(
                latency=entry["latency"] * self.scale_factor,
                energy=entry.get("energy", 0.0) * self.scale_factor,
            )
        if self.estimator is not None:
            latency, energy = self.estimator.predict(
                M=kwargs["M"] // self._scale_num_tokens,
                N=self._n,
                K=self._k,
            )
            return PerformanceResult(
                latency=latency * self.scale_factor,
                energy=energy * self.scale_factor,
            )
        raise ValueError(f"No match for GEMM key {key} and no Estimator configured.")

    def get_weights(self, **kwargs) -> float:
        return self._weights * self.scale_factor
