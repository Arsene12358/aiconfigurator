# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DatabaseV2 – the Orchestrator / Shim Layer.

``DatabaseV2`` is the single entry-point for latency queries from model code.
It manages:

* **Global state** – ``mode`` (SILICON, HYBRID, SOL) and ``backend``.
* **Operation registry** – a mapping of operation names → ``OperationV2``
  instances.
* **Query routing** – delegates ``query(op_name, **kwargs)`` to the correct
  ``OperationV2``.

Example (migration from V1)::

    # Before (V1 – tightly coupled)
    fc1_time = self.db.get_gemm_latency(tokens, intermediate_dim, hidden_dim)

    # After (V2 – unified API)
    fc1_time = self.db.query("gemm", M=tokens, N=intermediate_dim, K=hidden_dim)
"""

from __future__ import annotations

import logging
from typing import Any

from aiconfigurator.sdk import common
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.v2.operation_v2 import OperationV2

logger = logging.getLogger(__name__)


class DatabaseV2:
    """High-level orchestrator for V2 operation queries.

    Args:
        backend: Active inference backend name (e.g. ``"trtllm"``, ``"vllm"``).
        mode: Database mode controlling data-source selection.
        system_spec: Optional system specification dict (GPU details etc.).
    """

    def __init__(
        self,
        backend: str,
        mode: common.DatabaseMode = common.DatabaseMode.SILICON,
        system_spec: dict[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.mode = mode
        self.system_spec = system_spec or {}
        self._operations: dict[str, OperationV2] = {}

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def register_operation(self, name: str, op: OperationV2) -> None:
        """Register an operation instance under the given name.

        Overwrites any previously registered operation with the same name.
        """
        if name in self._operations:
            logger.info("Overwriting existing operation '%s'", name)
        self._operations[name] = op

    def unregister_operation(self, name: str) -> OperationV2 | None:
        """Remove and return a registered operation, or ``None`` if absent."""
        return self._operations.pop(name, None)

    def has_operation(self, name: str) -> bool:
        """Check whether an operation is registered."""
        return name in self._operations

    def get_operation(self, name: str) -> OperationV2 | None:
        """Retrieve a registered operation by name (or ``None``)."""
        return self._operations.get(name)

    @property
    def registered_operations(self) -> list[str]:
        """Return the names of all currently registered operations."""
        return list(self._operations)

    # ------------------------------------------------------------------
    # Query routing
    # ------------------------------------------------------------------

    def query(self, op_name: str, **kwargs) -> PerformanceResult:
        """Unified entry-point for all model latency queries.

        Routes the request to the ``OperationV2`` instance registered under
        *op_name* and forwards all keyword arguments.

        Args:
            op_name: Name of the operation (e.g. ``"gemm"``, ``"attention"``).
            **kwargs: Operation-specific parameters (``M``, ``N``, ``K``, …).

        Returns:
            ``PerformanceResult`` carrying latency (ms) and energy (W·ms).

        Raises:
            NotImplementedError: If *op_name* is not registered.
        """
        if op_name not in self._operations:
            raise NotImplementedError(
                f"Operation '{op_name}' is not registered. "
                f"Available: {self.registered_operations}"
            )
        return self._operations[op_name].query(**kwargs)

    # ------------------------------------------------------------------
    # Bulk helpers
    # ------------------------------------------------------------------

    def register_operations(self, ops: dict[str, OperationV2]) -> None:
        """Register multiple operations at once from a ``{name: op}`` mapping."""
        for name, op in ops.items():
            self.register_operation(name, op)

    def clear(self) -> None:
        """Remove all registered operations."""
        self._operations.clear()
