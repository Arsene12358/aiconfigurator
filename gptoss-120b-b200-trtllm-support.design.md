# GPT-OSS 120B on B200 GPU (trtllm) — SILICON Mode Support Design Doc
# NOTE: This file should be moved to C:/Users/KaiserL/docs/designs/ and NOT committed

**Author:** AI Configurator Team  
**Status:** Draft | **Version:** 1.0 | **Last Updated:** 2026-02-28

---

## 1. Executive Summary

The `openai/gpt-oss-120b` model uses a native `w4a16_mxfp4` MoE quantization format (4-bit MXFP4 weights, 16-bit activations). When running `aiconfigurator cli default` with `--system b200_sxm --backend trtllm --database-mode SILICON`, the tool fails at validation because the B200 trtllm perf database does not contain MoE performance data collected under the `w4a16_mxfp4` label — it only has `float16`, `fp8`, and `nvfp4`.

However, `w4a16_mxfp4` has compute factor 1 (same as `float16`), meaning its MoE computation performance is identical to `float16`. The only difference is memory footprint (0.5 bytes/param vs 2 bytes/param), which is already handled by the `QuantMapping` definition. Therefore, `w4a16_mxfp4` can safely reuse `float16` MoE performance tables, following the exact same precedent as `fp8_static` reusing `fp8` GEMM performance tables.

**Key Goals:**
- Enable GPT-OSS 120B to run through CLI default SILICON mode on B200 SXM with trtllm backend
- Follow the established `fp8_static` to `fp8` remapping pattern for consistency
- Keep changes minimal and non-breaking

**Deliverables:**
- Code changes to `perf_database.py` (validation + query normalization)
- Updated `support_matrix.csv` entries for gpt-oss-120b on b200_sxm/trtllm

---

## 2. Architecture Overview

### 2.1 Components Affected

| Path | Type | Description |
|------|------|-------------|
| src/aiconfigurator/sdk/perf_database.py | MODIFY | Add MoE quant mode normalization and update supported_quant_mode |
| src/aiconfigurator/systems/support_matrix.csv | MODIFY | Update gpt-oss-120b entries for b200_sxm/trtllm from FAIL to PASS |

### 2.2 Data Flow

```
CLI default mode
    |
    v
TaskConfig.__init__()
    |
    +-- _apply_model_quant_defaults()
    |       +-- gpt-oss-120b config -> quant_algo="mxfp4"
    |               +-- moe_quant_mode = MoEQuantMode.w4a16_mxfp4
    |
    +-- validate()
    |       +-- _supported_or_raise("moe", "w4a16_mxfp4")
    |               +-- checks database.supported_quant_mode["moe"]
    |               +-- FIX: add w4a16_mxfp4 when float16 is present
    |
    v
InferenceSession.run_agg()
    |
    +-- MoE.query() -> database.query_moe(quant_mode=w4a16_mxfp4)
            +-- FIX: normalize w4a16_mxfp4 -> float16 for table lookup
```

---

## 3. Detailed Design

### 3.1 Root Cause

The GPT-OSS 120B model config (`openai--gpt-oss-120b_config.json`) specifies:
```json
"quantization_config": { "quant_method": "mxfp4" }
```

This is parsed by `_apply_model_quant_defaults()` in `models.py`:
```python
elif quant_algo == "mxfp4":
    overrides["gemm_quant_mode"] = common.GEMMQuantMode.float16
    overrides["moe_quant_mode"] = common.MoEQuantMode.w4a16_mxfp4
```

The `MoEQuantMode.w4a16_mxfp4` enum: `QuantMapping(0.5, 1, "w4a16_mxfp4")` — 0.5 bytes/param memory, compute factor 1 (same as float16).

During `TaskConfig.validate()`, `_supported_or_raise("moe", "w4a16_mxfp4")` fails because the B200 trtllm MoE perf data only contains `nvfp4`, `float16`, and `fp8`.

### 3.2 Existing Precedent: fp8_static -> fp8

The codebase has this exact pattern for GEMM:

1. **Validation**: `fp8_static` is added to supported GEMM modes when `fp8` is present
2. **Normalization**: `_normalize_gemm_quant_mode_for_table` maps `fp8_static` -> `fp8`
3. **Usage**: `query_gemm` calls the normalizer before table lookup

### 3.3 Proposed Changes

#### 3.3.1 Add `_normalize_moe_quant_mode_for_table` static method

```python
@staticmethod
def _normalize_moe_quant_mode_for_table(quant_mode: common.MoEQuantMode) -> common.MoEQuantMode:
    """w4a16_mxfp4 is a weight-only 4-bit mode with fp16 activations (compute=1),
    so it reuses float16 MoE perf tables for computation estimation."""
    if quant_mode == common.MoEQuantMode.w4a16_mxfp4:
        return common.MoEQuantMode.float16
    return quant_mode
```

#### 3.3.2 Update supported_quant_mode for trtllm and sglang MoE

After existing `fp8_static` block, add `w4a16_mxfp4` to MoE modes when `float16` is present.

#### 3.3.3 Normalize quant mode in query_moe

Use `table_quant_mode = self._normalize_moe_quant_mode_for_table(quant_mode)` before perf table lookups. Keep original `quant_mode` for SOL/empirical calculations.

---

## 4. Alternatives Considered

1. **Collect actual w4a16_mxfp4 MoE perf data** — Not chosen; compute factor identical to float16.
2. **Force HYBRID mode** — Not chosen; SILICON is default/recommended.

---

## 5. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| src/aiconfigurator/sdk/perf_database.py | MODIFY | Add normalization and supported mode |
| src/aiconfigurator/systems/support_matrix.csv | MODIFY | Update FAIL to PASS entries |

---

## 6. Testing

- Run support matrix test for gpt-oss-120b on b200_sxm/trtllm
- Run CLI default mode with SILICON database mode
- Verify no regression on other models/systems/backends

---

## Appendix: GPT-OSS 120B Model Parameters

| Parameter | Value |
|-----------|-------|
| Architecture | GptOssForCausalLM |
| Hidden Size | 2880 |
| Num Layers | 36 |
| Num Experts | 128 |
| Experts per Token | 4 |
| Quant Method | mxfp4 |

## Appendix: MoEQuantMode Reference

| Mode | Memory (bytes/param) | Compute Factor | Description |
|------|---------------------|----------------|-------------|
| float16 | 2 | 1 | w16a16 baseline |
| fp8 | 1 | 2 | w8fp8 |
| nvfp4 | 9/16 | 4 | nvfp4 on Blackwell |
| w4a16_mxfp4 | 0.5 | 1 | Native GPT-OSS format |
