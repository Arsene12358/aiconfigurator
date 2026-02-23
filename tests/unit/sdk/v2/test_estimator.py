# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from aiconfigurator.sdk.v2.estimator import Estimator, LinearInterpolator, NearestNeighbor


def _sample_matrix() -> np.ndarray:
    """2-D parameter space with columns [p1, p2, latency, energy]."""
    rows = []
    for p1 in (1, 2, 4, 8):
        for p2 in (10, 20, 40, 80):
            latency = 0.1 * p1 + 0.01 * p2  # linear in both params
            energy = latency * 350.0
            rows.append([p1, p2, latency, energy])
    return np.array(rows, dtype=np.float64)


class TestEstimatorABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Estimator()


class TestLinearInterpolator:
    def test_exact_match(self):
        interp = LinearInterpolator()
        mat = _sample_matrix()
        interp.fit(mat)
        lat, eng = interp.predict(p1=2.0, p2=20.0)
        assert lat == pytest.approx(0.1 * 2 + 0.01 * 20, rel=1e-4)
        assert eng == pytest.approx((0.1 * 2 + 0.01 * 20) * 350.0, rel=1e-4)

    def test_interpolated_point(self):
        interp = LinearInterpolator()
        interp.fit(_sample_matrix())
        lat, eng = interp.predict(p1=3.0, p2=30.0)
        expected_lat = 0.1 * 3 + 0.01 * 30
        assert lat == pytest.approx(expected_lat, rel=0.2)
        assert eng > 0

    def test_extrapolation_falls_back_to_nn(self):
        interp = LinearInterpolator()
        interp.fit(_sample_matrix())
        lat, eng = interp.predict(p1=100.0, p2=100.0)
        assert lat > 0
        assert eng > 0

    def test_predict_before_fit_raises(self):
        interp = LinearInterpolator()
        with pytest.raises(ValueError, match="not been fitted"):
            interp.predict(p1=1.0, p2=10.0)

    def test_fit_with_empty_data_skips(self):
        interp = LinearInterpolator()
        interp.fit(np.array([]))
        assert not interp._fitted


class TestNearestNeighbor:
    def test_exact_match(self):
        nn = NearestNeighbor()
        nn.fit(_sample_matrix())
        lat, eng = nn.predict(p1=4.0, p2=40.0)
        assert lat == pytest.approx(0.1 * 4 + 0.01 * 40)
        assert eng == pytest.approx((0.1 * 4 + 0.01 * 40) * 350.0)

    def test_nearest_point(self):
        nn = NearestNeighbor()
        nn.fit(_sample_matrix())
        lat, eng = nn.predict(p1=3.9, p2=39.9)
        expected_lat = 0.1 * 4 + 0.01 * 40
        assert lat == pytest.approx(expected_lat)

    def test_predict_before_fit_raises(self):
        nn = NearestNeighbor()
        with pytest.raises(ValueError, match="not been fitted"):
            nn.predict(p1=1.0, p2=10.0)

    def test_fit_with_empty_data_skips(self):
        nn = NearestNeighbor()
        nn.fit(np.array([]))
        assert not nn._fitted
