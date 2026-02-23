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


class ContextMLAV2(OperationV2):
    """Context MLA (Multi-head Latent Attention) operation.

    Lookup key: ``(batch_size, seq_len, prefix, num_heads, kvcache_quant, fmha_quant)``.
    """

    def __init__(
        self,
        backend_context: Any,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        estimator: Estimator | None = None,
        scale_factor: float = 1.0,
    ) -> None:
        self._num_heads = num_heads
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        super().__init__(backend_context, estimator, scale_factor)

    def load_data(self) -> tuple[dict[tuple, dict[str, float]], np.ndarray | None]:
        return {}, None

    def get_lookup_key(self, **kwargs) -> tuple:
        return (
            kwargs["batch_size"],
            kwargs["s"],
            kwargs.get("prefix", 0),
            self._num_heads,
            self._kvcache_quant_mode.name,
            self._fmha_quant_mode.name,
        )


class GenerationMLAV2(OperationV2):
    """Generation MLA operation.

    Lookup key: ``(batch_size, seq_len, num_heads, kv_cache_dtype)``.
    """

    def __init__(
        self,
        backend_context: Any,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        estimator: Estimator | None = None,
        scale_factor: float = 1.0,
    ) -> None:
        self._num_heads = num_heads
        self._kv_cache_dtype = kv_cache_dtype
        super().__init__(backend_context, estimator, scale_factor)

    def load_data(self) -> tuple[dict[tuple, dict[str, float]], np.ndarray | None]:
        return {}, None

    def get_lookup_key(self, **kwargs) -> tuple:
        return (
            kwargs["batch_size"],
            kwargs["s"],
            self._num_heads,
            self._kv_cache_dtype.name,
        )


class MLABmmV2(OperationV2):
    """MLA batched matrix multiply operation.

    Lookup key: ``(batch_size, num_heads, quant_mode, if_pre)``.
    """

    def __init__(
        self,
        backend_context: Any,
        num_heads: int,
        quant_mode: common.GEMMQuantMode,
        if_pre: bool = True,
        estimator: Estimator | None = None,
        scale_factor: float = 1.0,
    ) -> None:
        self._num_heads = num_heads
        self._quant_mode = quant_mode
        self._if_pre = if_pre
        super().__init__(backend_context, estimator, scale_factor)

    def load_data(self) -> tuple[dict[tuple, dict[str, float]], np.ndarray | None]:
        return {}, None

    def get_lookup_key(self, **kwargs) -> tuple:
        return (
            kwargs["batch_size"],
            self._num_heads,
            self._quant_mode.name,
            self._if_pre,
        )
