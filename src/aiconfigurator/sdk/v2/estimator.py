# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Modular Estimation (Strategy Pattern) for OperationV2.

Estimators provide fallback latency prediction when exact-match data is
unavailable.  Each strategy is pluggable: ``OperationV2`` delegates to
its assigned ``Estimator`` whenever ``query()`` misses the flat lookup dict.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Estimator(ABC):
    """Abstract base for all estimation / interpolation strategies."""

    @abstractmethod
    def fit(self, flat_matrix: np.ndarray) -> None:
        """Prepare the estimation model from the flat data matrix.

        Args:
            flat_matrix: 2-D NumPy array where each row is
                ``[param_1, param_2, ..., param_N, latency, energy]``.
                The last two columns are always the target values.
        """

    @abstractmethod
    def predict(self, **kwargs) -> tuple[float, float]:
        """Return ``(estimated_latency, estimated_energy)`` for the given parameters.

        Raises:
            ValueError: If the estimator has not been fitted or input is
                out of the supported range.
        """


class LinearInterpolator(Estimator):
    """N-dimensional linear interpolation backed by :mod:`scipy`.

    Uses ``scipy.interpolate.LinearNDInterpolator`` on the stored flat matrix.
    Falls back to nearest-neighbour extrapolation for points outside the convex
    hull of the training data.
    """

    def __init__(self) -> None:
        self._latency_interp: Any = None
        self._energy_interp: Any = None
        self._latency_nn: Any = None
        self._energy_nn: Any = None
        self._fitted = False

    def fit(self, flat_matrix: np.ndarray) -> None:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        if flat_matrix is None or len(flat_matrix) == 0:
            logger.warning("LinearInterpolator.fit called with empty data – skipping")
            return

        points = flat_matrix[:, :-2]
        latencies = flat_matrix[:, -2]
        energies = flat_matrix[:, -1]

        self._latency_interp = LinearNDInterpolator(points, latencies)
        self._energy_interp = LinearNDInterpolator(points, energies)
        self._latency_nn = NearestNDInterpolator(points, latencies)
        self._energy_nn = NearestNDInterpolator(points, energies)
        self._fitted = True

    def predict(self, **kwargs) -> tuple[float, float]:
        if not self._fitted:
            raise ValueError("LinearInterpolator has not been fitted yet.")

        point = np.array([list(kwargs.values())])
        latency = self._latency_interp(point).item()
        energy = self._energy_interp(point).item()

        if np.isnan(latency):
            latency = float(self._latency_nn(point).item())
            energy = float(self._energy_nn(point).item())
            logger.debug("LinearInterpolator: fell back to nearest-neighbour for %s", kwargs)

        return float(latency), float(energy)


class NearestNeighbor(Estimator):
    """Simple nearest-neighbour lookup in the parameter space."""

    def __init__(self) -> None:
        self._tree: Any = None
        self._latencies: np.ndarray | None = None
        self._energies: np.ndarray | None = None
        self._fitted = False

    def fit(self, flat_matrix: np.ndarray) -> None:
        from scipy.spatial import cKDTree

        if flat_matrix is None or len(flat_matrix) == 0:
            logger.warning("NearestNeighbor.fit called with empty data – skipping")
            return

        points = flat_matrix[:, :-2]
        self._latencies = flat_matrix[:, -2]
        self._energies = flat_matrix[:, -1]
        self._tree = cKDTree(points)
        self._fitted = True

    def predict(self, **kwargs) -> tuple[float, float]:
        if not self._fitted:
            raise ValueError("NearestNeighbor has not been fitted yet.")

        point = np.array([list(kwargs.values())])
        _, idx = self._tree.query(point)
        idx = idx.item()
        return float(self._latencies[idx]), float(self._energies[idx])
