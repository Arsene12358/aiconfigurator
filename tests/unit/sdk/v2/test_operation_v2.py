# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.v2.database_v2 import DatabaseV2
from aiconfigurator.sdk.v2.estimator import NearestNeighbor
from aiconfigurator.sdk.v2.operation_v2 import CompositeOperationV2, OperationV2


class _StubOperation(OperationV2):
    """Minimal concrete subclass for testing the ABC."""

    def __init__(self, data=None, estimator=None, scale_factor=1.0):
        self._data = data or {}
        super().__init__(backend_context="test", estimator=estimator, scale_factor=scale_factor)

    def load_data(self):
        return self._data, None

    def get_lookup_key(self, **kwargs):
        return tuple(sorted(kwargs.values()))


class _StubComposite(CompositeOperationV2):
    """Minimal composite that sums two sub-operation results."""

    def query(self, **kwargs) -> PerformanceResult:
        total = PerformanceResult(0.0, 0.0)
        for sub in self.sub_operations:
            result = self._db.query(sub, **kwargs)
            total = total + result
        return PerformanceResult(
            float(total) * self.scale_factor,
            energy=total.energy * self.scale_factor,
        )


class TestOperationV2:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            OperationV2(backend_context="test")

    def test_exact_match_query(self):
        data = {
            (1, 2): {"latency": 10.0, "energy": 50.0},
            (3, 4): {"latency": 20.0, "energy": 100.0},
        }
        op = _StubOperation(data=data)
        result = op.query(a=1, b=2)
        assert isinstance(result, PerformanceResult)
        assert float(result) == 10.0
        assert result.energy == 50.0

    def test_scale_factor(self):
        data = {(1, 2): {"latency": 10.0, "energy": 50.0}}
        op = _StubOperation(data=data, scale_factor=2.0)
        result = op.query(a=1, b=2)
        assert float(result) == 20.0
        assert result.energy == 100.0

    def test_missing_key_no_estimator_raises(self):
        op = _StubOperation(data={})
        with pytest.raises(ValueError, match="No exact match"):
            op.query(a=99, b=99)

    def test_estimator_fallback(self):
        mat = np.array([
            [1.0, 2.0, 10.0, 50.0],
            [3.0, 4.0, 20.0, 100.0],
        ])
        nn = NearestNeighbor()

        class _WithMatrix(_StubOperation):
            def load_data(self):
                return {}, mat

        op = _WithMatrix(estimator=nn)
        result = op.query(a=1, b=2)
        assert float(result) > 0

    def test_get_weights_default(self):
        op = _StubOperation()
        assert op.get_weights() == 0.0


class TestCompositeOperationV2:
    def test_composite_aggregation(self):
        db = DatabaseV2(backend="test")
        op_a = _StubOperation(data={(1, 2): {"latency": 5.0, "energy": 25.0}})
        op_b = _StubOperation(data={(1, 2): {"latency": 3.0, "energy": 15.0}})
        db.register_operation("a", op_a)
        db.register_operation("b", op_b)

        composite = _StubComposite(db_reference=db, sub_operations=["a", "b"])
        result = composite.query(a=1, b=2)
        assert float(result) == pytest.approx(8.0)
        assert result.energy == pytest.approx(40.0)

    def test_composite_scale_factor(self):
        db = DatabaseV2(backend="test")
        op_a = _StubOperation(data={(1, 2): {"latency": 10.0, "energy": 50.0}})
        db.register_operation("a", op_a)

        composite = _StubComposite(db_reference=db, sub_operations=["a"], scale_factor=3.0)
        result = composite.query(a=1, b=2)
        assert float(result) == pytest.approx(30.0)
        assert result.energy == pytest.approx(150.0)

    def test_composite_load_data_noop(self):
        db = DatabaseV2(backend="test")
        composite = _StubComposite(db_reference=db, sub_operations=[])
        d, m = composite.load_data()
        assert d == {}
        assert m is None
