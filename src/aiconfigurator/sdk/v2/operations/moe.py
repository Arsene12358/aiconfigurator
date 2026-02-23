# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aiconfigurator.sdk import common
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.v2.estimator import Estimator
from aiconfigurator.sdk.v2.operation_v2 import CompositeOperationV2, OperationV2

logger = logging.getLogger(__name__)


class MoEV2(OperationV2):
    """Mixture-of-Experts compute operation.

    Lookup key: ``(num_tokens, hidden_size, inter_size, topk, num_experts,
    moe_tp_size, moe_ep_size, quant_mode, workload_distribution, is_context)``.
    """

    def __init__(
        self,
        backend_context: Any,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        attention_dp_size: int,
        estimator: Estimator | None = None,
        scale_factor: float = 1.0,
        is_context: bool = True,
        is_gated: bool = True,
    ) -> None:
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._quant_mode = quant_mode
        self._workload_distribution = workload_distribution
        self._attention_dp_size = attention_dp_size
        self._is_context = is_context
        self._is_gated = is_gated

        num_gemms = 3 if is_gated else 2
        self._weights = (
            hidden_size
            * inter_size
            * num_experts
            * quant_mode.value.memory
            * num_gemms
            // moe_ep_size
            // moe_tp_size
        )
        super().__init__(backend_context, estimator, scale_factor)

    def load_data(self) -> tuple[dict[tuple, dict[str, float]], np.ndarray | None]:
        return {}, None

    def get_lookup_key(self, **kwargs) -> tuple:
        x = kwargs["x"] * self._attention_dp_size
        quant_mode = kwargs.get("quant_mode", self._quant_mode)
        return (
            x,
            self._hidden_size,
            self._inter_size,
            self._topk,
            self._num_experts,
            self._moe_tp_size,
            self._moe_ep_size,
            quant_mode.name if hasattr(quant_mode, "name") else str(quant_mode),
            self._workload_distribution,
            self._is_context,
        )

    def get_weights(self, **kwargs) -> float:
        return self._weights * self.scale_factor


class MoEDispatchV2(CompositeOperationV2):
    """MoE dispatch communication – a composite of NCCL / custom-allreduce sub-ops.

    The dispatch decomposes into backend-specific communication primitives
    (allgather, reduce-scatter, alltoall, custom allreduce) and aggregates
    their latencies.
    """

    def __init__(
        self,
        db_reference: Any,
        hidden_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        attention_dp_size: int,
        pre_dispatch: bool,
        scale_factor: float = 1.0,
    ) -> None:
        self._hidden_size = hidden_size
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._pre_dispatch = pre_dispatch
        self._attention_tp_size = moe_tp_size * moe_ep_size // attention_dp_size
        self.num_gpus = moe_ep_size * moe_tp_size
        super().__init__(db_reference, sub_operations=["nccl", "custom_allreduce"], scale_factor=scale_factor)

    def query(self, **kwargs) -> PerformanceResult:
        return PerformanceResult(0.0, energy=0.0)
