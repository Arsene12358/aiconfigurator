# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OperationV2 – the core abstraction for the operation-oriented architecture.

Each operation (GEMM, Attention, MoE, …) is a self-contained class that:
  * knows which data files to load  (``load_data``)
  * translates query kwargs into a flat tuple key  (``get_lookup_key``)
  * performs O(1) exact-match queries with optional Estimator fallback  (``query``)

CompositeOperationV2 uses the Composite pattern to aggregate latencies
across multiple sub-operations (e.g. dispatch = allgather + reduce-scatter).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.v2.database_v2 import DatabaseV2
    from aiconfigurator.sdk.v2.estimator import Estimator

logger = logging.getLogger(__name__)


class OperationV2(ABC):
    """Abstract base class for all V2 operations.

    Subclasses must implement :meth:`load_data` and :meth:`get_lookup_key`.
    The default :meth:`query` first tries an O(1) exact-match lookup in the
    flat tuple dictionary, then falls back to the pluggable ``Estimator``.

    Args:
        backend_context: Backend-specific context (e.g. backend name string
            or a rich context object carrying system specs).
        estimator: Optional estimation strategy used when exact data is missing.
        scale_factor: Multiplicative factor applied to the returned latency/energy.
    """

    def __init__(
        self,
        backend_context: Any,
        estimator: Estimator | None = None,
        scale_factor: float = 1.0,
    ) -> None:
        self.backend_context = backend_context
        self.estimator = estimator
        self.scale_factor = scale_factor

        self.exact_match_dict: dict[tuple, dict[str, float]]
        self.flat_matrix: np.ndarray | None
        self.exact_match_dict, self.flat_matrix = self.load_data()

        if self.estimator is not None and self.flat_matrix is not None:
            self.estimator.fit(self.flat_matrix)

    # ------------------------------------------------------------------
    # Abstract interface – subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def load_data(self) -> tuple[dict[tuple, dict[str, float]], np.ndarray | None]:
        """Load performance data from backend-specific files.

        Returns:
            A 2-tuple of:
            * **exact_match_dict** – ``{(param1, …): {"latency": …, "energy": …}}``
            * **flat_matrix** – NumPy 2-D array with columns
              ``[param1, …, latency, energy]``, or ``None`` if no data is
              available for interpolation.
        """

    @abstractmethod
    def get_lookup_key(self, **kwargs) -> tuple:
        """Translate caller kwargs into a normalised tuple key.

        The returned tuple must match the keys stored in
        ``self.exact_match_dict``.
        """

    # ------------------------------------------------------------------
    # Query – may be overridden by subclasses for custom logic
    # ------------------------------------------------------------------

    def query(self, **kwargs) -> PerformanceResult:
        """Query the operation for latency (+ energy).

        1. Exact-match lookup via :meth:`get_lookup_key`.
        2. Estimator fallback if available.
        3. ``ValueError`` otherwise.
        """
        key = self.get_lookup_key(**kwargs)

        if key in self.exact_match_dict:
            entry = self.exact_match_dict[key]
            return PerformanceResult(
                latency=entry["latency"] * self.scale_factor,
                energy=entry.get("energy", 0.0) * self.scale_factor,
            )

        if self.estimator is not None:
            latency, energy = self.estimator.predict(**kwargs)
            return PerformanceResult(
                latency=latency * self.scale_factor,
                energy=energy * self.scale_factor,
            )

        raise ValueError(f"No exact match for key {key} and no Estimator configured.")

    # ------------------------------------------------------------------
    # Weights – default to 0; subclasses override when applicable
    # ------------------------------------------------------------------

    def get_weights(self, **kwargs) -> float:
        """Return the weight-memory footprint (bytes) for this operation."""
        return 0.0


class CompositeOperationV2(OperationV2):
    """An operation composed of multiple sub-operations.

    Uses the Composite pattern: the composite holds a reference back to
    ``DatabaseV2`` so it can issue sub-queries, then aggregates the results.

    Args:
        db_reference: The ``DatabaseV2`` orchestrator.
        sub_operations: List of operation names to query in sequence.
        scale_factor: Multiplicative factor applied to the aggregated result.
    """

    def __init__(
        self,
        db_reference: DatabaseV2,
        sub_operations: list[str],
        scale_factor: float = 1.0,
    ) -> None:
        self._db = db_reference
        self.sub_operations = sub_operations
        # Bypass parent __init__ data loading since composites don't own data
        self.backend_context = db_reference.backend
        self.estimator = None
        self.scale_factor = scale_factor
        self.exact_match_dict = {}
        self.flat_matrix = None

    # Data loading is a no-op for composite operations
    def load_data(self) -> tuple[dict[tuple, dict[str, float]], np.ndarray | None]:
        return {}, None

    def get_lookup_key(self, **kwargs) -> tuple:
        return ()

    @abstractmethod
    def query(self, **kwargs) -> PerformanceResult:
        """Decompose kwargs and call ``self._db.query()`` for each sub-operation."""
