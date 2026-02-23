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


class ContextAttentionV2(OperationV2):
    """Context (prefill) attention operation.

    Lookup key: ``(batch_size, seq_len, prefix, n_heads, n_kv_heads,
    kvcache_quant, fmha_quant, window_size, head_size)``.
    """

    def __init__(
        self,
        backend_context: Any,
        n: int,
        n_kv: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        estimator: Estimator | None = None,
        scale_factor: float = 1.0,
        window_size: int = 0,
        head_size: int = 128,
    ) -> None:
        self._n = n
        self._n_kv = n_kv
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._window_size = window_size
        self._head_size = head_size
        super().__init__(backend_context, estimator, scale_factor)

    def load_data(self) -> tuple[dict[tuple, dict[str, float]], np.ndarray | None]:
        return {}, None

    def get_lookup_key(self, **kwargs) -> tuple:
        return (
            kwargs["batch_size"],
            kwargs["s"],
            kwargs.get("prefix", 0),
            self._n,
            self._n_kv,
            self._kvcache_quant_mode.name,
            self._fmha_quant_mode.name,
            self._window_size,
            self._head_size,
        )


class GenerationAttentionV2(OperationV2):
    """Generation (decode) attention operation.

    Lookup key: ``(batch_size, seq_len, n_heads, n_kv_heads,
    kv_cache_dtype, window_size, head_size)``.
    """

    def __init__(
        self,
        backend_context: Any,
        n: int,
        n_kv: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        estimator: Estimator | None = None,
        scale_factor: float = 1.0,
        window_size: int = 0,
        head_size: int = 128,
    ) -> None:
        self._n = n
        self._n_kv = n_kv
        self._kv_cache_dtype = kv_cache_dtype
        self._window_size = window_size
        self._head_size = head_size
        super().__init__(backend_context, estimator, scale_factor)

    def load_data(self) -> tuple[dict[tuple, dict[str, float]], np.ndarray | None]:
        return {}, None

    def get_lookup_key(self, **kwargs) -> tuple:
        return (
            kwargs["batch_size"],
            kwargs["s"],
            self._n,
            self._n_kv,
            self._kv_cache_dtype.name,
            self._window_size,
            self._head_size,
        )
