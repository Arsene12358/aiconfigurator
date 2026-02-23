# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.v2.database_v2 import DatabaseV2
from aiconfigurator.sdk.v2.operation_v2 import OperationV2


class _DummyOp(OperationV2):
    """Minimal op that returns a fixed result."""

    def __init__(self, latency=1.0, energy=5.0):
        self._lat = latency
        self._eng = energy
        super().__init__(backend_context="dummy")

    def load_data(self):
        return {(42,): {"latency": self._lat, "energy": self._eng}}, None

    def get_lookup_key(self, **kwargs):
        return (kwargs.get("x", 42),)


class TestDatabaseV2:
    def test_register_and_query(self):
        db = DatabaseV2(backend="trtllm")
        op = _DummyOp(latency=7.0, energy=35.0)
        db.register_operation("my_op", op)

        result = db.query("my_op", x=42)
        assert isinstance(result, PerformanceResult)
        assert float(result) == 7.0
        assert result.energy == 35.0

    def test_query_unknown_op_raises(self):
        db = DatabaseV2(backend="trtllm")
        with pytest.raises(NotImplementedError, match="not registered"):
            db.query("nonexistent")

    def test_has_operation(self):
        db = DatabaseV2(backend="trtllm")
        assert not db.has_operation("op")
        db.register_operation("op", _DummyOp())
        assert db.has_operation("op")

    def test_unregister_operation(self):
        db = DatabaseV2(backend="trtllm")
        op = _DummyOp()
        db.register_operation("op", op)
        removed = db.unregister_operation("op")
        assert removed is op
        assert not db.has_operation("op")

    def test_unregister_nonexistent_returns_none(self):
        db = DatabaseV2(backend="trtllm")
        assert db.unregister_operation("nope") is None

    def test_get_operation(self):
        db = DatabaseV2(backend="trtllm")
        op = _DummyOp()
        db.register_operation("op", op)
        assert db.get_operation("op") is op
        assert db.get_operation("nope") is None

    def test_registered_operations(self):
        db = DatabaseV2(backend="trtllm")
        db.register_operation("a", _DummyOp())
        db.register_operation("b", _DummyOp())
        assert set(db.registered_operations) == {"a", "b"}

    def test_register_operations_bulk(self):
        db = DatabaseV2(backend="trtllm")
        ops = {"x": _DummyOp(), "y": _DummyOp()}
        db.register_operations(ops)
        assert db.has_operation("x")
        assert db.has_operation("y")

    def test_clear(self):
        db = DatabaseV2(backend="trtllm")
        db.register_operation("a", _DummyOp())
        db.clear()
        assert db.registered_operations == []

    def test_overwrite_existing_operation(self):
        db = DatabaseV2(backend="trtllm")
        db.register_operation("op", _DummyOp(latency=1.0))
        db.register_operation("op", _DummyOp(latency=9.0))
        result = db.query("op", x=42)
        assert float(result) == 9.0

    def test_mode_and_backend(self):
        db = DatabaseV2(backend="vllm", mode=common.DatabaseMode.HYBRID)
        assert db.backend == "vllm"
        assert db.mode == common.DatabaseMode.HYBRID

    def test_system_spec(self):
        spec = {"gpu": {"mem_bw": 100.0}}
        db = DatabaseV2(backend="trtllm", system_spec=spec)
        assert db.system_spec == spec

    def test_default_system_spec_is_empty_dict(self):
        db = DatabaseV2(backend="trtllm")
        assert db.system_spec == {}


class TestDatabaseV2WithConcreteOps:
    """Integration-style tests that wire up concrete V2 operations."""

    def test_gemm_v2_registration_and_query(self):
        from aiconfigurator.sdk.v2.operations.gemm import GemmV2

        db = DatabaseV2(backend="trtllm")
        gemm = GemmV2(
            backend_context="trtllm",
            n=1024,
            k=512,
            quant_mode=common.GEMMQuantMode.float16,
        )
        # Manually inject some data for testing
        gemm.exact_match_dict = {
            ("float16", 32, 1024, 512): {"latency": 0.5, "energy": 10.0}
        }
        db.register_operation("gemm", gemm)
        result = db.query("gemm", M=32)
        assert float(result) == pytest.approx(0.5)
        assert result.energy == pytest.approx(10.0)

    def test_custom_allreduce_v2_tp1_shortcircuit(self):
        from aiconfigurator.sdk.v2.operations.custom_allreduce import CustomAllReduceV2

        db = DatabaseV2(backend="trtllm")
        op = CustomAllReduceV2(backend_context="trtllm", h=4096, tp_size=1)
        db.register_operation("custom_allreduce", op)
        result = db.query("custom_allreduce", x=128)
        assert float(result) == 0.0

    def test_p2p_v2_pp1_shortcircuit(self):
        from aiconfigurator.sdk.v2.operations.p2p import P2PV2

        db = DatabaseV2(backend="trtllm")
        op = P2PV2(backend_context="trtllm", h=4096, pp_size=1)
        db.register_operation("p2p", op)
        result = db.query("p2p", x=128)
        assert float(result) == 0.0

    def test_gemm_v2_weights(self):
        from aiconfigurator.sdk.v2.operations.gemm import GemmV2

        gemm = GemmV2(
            backend_context="trtllm",
            n=1024,
            k=512,
            quant_mode=common.GEMMQuantMode.float16,
            scale_factor=2.0,
        )
        expected = 1024 * 512 * 2 * 2.0  # n * k * memory_per_element * scale
        assert gemm.get_weights() == expected
