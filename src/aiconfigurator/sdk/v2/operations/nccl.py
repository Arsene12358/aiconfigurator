# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aiconfigurator.sdk import common
from aiconfigurator.sdk.v2.estimator import Estimator
from aiconfigurator.sdk.v2.operation_v2 import OperationV2

logger = logging.getLogger(__name__)


class NcclV2(OperationV2):
    """NCCL collective communication operation.

    Lookup key: ``(comm_quant_mode, nccl_op, num_gpus, message_size)``.
    """

    def __init__(
        self,
        backend_context: Any,
        nccl_op: str,
        num_elements_per_token: int,
        num_gpus: int,
        comm_quant_mode: common.CommQuantMode,
        estimator: Estimator | None = None,
        scale_factor: float = 1.0,
    ) -> None:
        self._nccl_op = nccl_op
        self._num_elements_per_token = num_elements_per_token
        self._num_gpus = num_gpus
        self._comm_quant_mode = comm_quant_mode
        super().__init__(backend_context, estimator, scale_factor)

    def load_data(self) -> tuple[dict[tuple, dict[str, float]], np.ndarray | None]:
        return {}, None

    def get_lookup_key(self, **kwargs) -> tuple:
        message_size = kwargs["x"] * self._num_elements_per_token
        return (self._comm_quant_mode.name, self._nccl_op, self._num_gpus, message_size)
