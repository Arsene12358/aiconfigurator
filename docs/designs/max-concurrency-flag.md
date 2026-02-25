# `--max-concurrency` Flag for Default CLI Mode — Design Doc

**Author:** AI Agent  
**Reviewers:** —  
**Status:** Draft  
**Version:** 1.0  
**Last Updated:** 2026-02-26

---

## 1. Executive Summary

Users running `aiconfigurator cli default` currently cannot limit the maximum concurrency level explored during the configuration search. The tool evaluates all possible batch sizes and parallelism combinations, producing configurations with potentially very high concurrency (hundreds of concurrent requests). In deployment scenarios where the expected concurrency is bounded (e.g., a small number of users or API clients), many of the high-concurrency configurations are irrelevant and can obscure the best low-concurrency option.

This design introduces a `--max-concurrency` flag for the `default` CLI mode that filters the search results to only include configurations with a concurrency level at or below the specified limit. The filtering is applied post-sweep, before the picking/ranking stage, so it constrains which configurations are eligible for selection without altering the search space itself.

**Key Goals:**
- Allow users to cap the concurrency in the `default` CLI mode results
- Filter out configurations exceeding the limit before ranking
- Expose the parameter in both the CLI and the Python API (`cli_default`)

**Deliverables:**
- `--max-concurrency` CLI argument in `_add_default_mode_arguments`
- Concurrency filter applied in `_execute_task_configs` on `pareto_df`
- `max_concurrency` parameter added to `cli_default()` Python API
- Unit tests for argument parsing and filtering logic

---

## 2. Architecture Overview

### 2.1 Components Modified

| Path | Type | Description |
|------|------|-------------|
| `src/aiconfigurator/cli/main.py` | MODIFY | Add `--max-concurrency` CLI arg, pass through `main()` → `_execute_task_configs()`, apply filter |
| `src/aiconfigurator/cli/api.py` | MODIFY | Add `max_concurrency` kwarg to `cli_default()` and `_execute_and_wrap_result()` |
| `tests/unit/cli/test_argument_parsing.py` | MODIFY | Add test for new argument parsing |

### 2.2 Data Flow Diagram

```
CLI args (--max-concurrency N)
        │
        ▼
main(args)
        │
        ▼
_execute_task_configs(task_configs, ..., max_concurrency=N)
        │
        ├── for each experiment:
        │       TaskRunner.run() → pareto_df
        │       ┌─────────────────────────────┐
        │       │ FILTER: concurrency <= N    │  ← NEW
        │       └─────────────────────────────┘
        │       process_experiment_result(filtered_df, ...)
        │
        ▼
log_final_summary / save_results
```

---

## 3. Detailed Design

### 3.1 CLI Argument

**Location:** `src/aiconfigurator/cli/main.py` — `_add_default_mode_arguments()`

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `--max-concurrency` | No | `None` (no limit) | Maximum concurrency level for configurations. Filters out configs where the `concurrency` column exceeds this value. |

```python
parser.add_argument(
    "--max-concurrency",
    type=int,
    default=None,
    help="Maximum concurrency level. Filters out configurations whose concurrency exceeds this value.",
)
```

### 3.2 Filter in `_execute_task_configs`

**File:** `src/aiconfigurator/cli/main.py`

The filter is applied to each experiment's `pareto_df` immediately after `TaskRunner.run()` returns and before `process_experiment_result()` is called:

```python
if max_concurrency is not None and pareto_df is not None and not pareto_df.empty:
    pre_filter_count = len(pareto_df)
    pareto_df = pareto_df[pareto_df["concurrency"] <= max_concurrency]
    logger.info(
        "max_concurrency=%d: filtered %d → %d configurations for %s",
        max_concurrency, pre_filter_count, len(pareto_df), exp_name,
    )
```

This works for both `agg` and `disagg` modes because both `ColumnsAgg` and `ColumnsDisagg` include a `concurrency` column.

### 3.3 Modifications to `main()`

**File:** `src/aiconfigurator/cli/main.py` — `main()`

Pass `max_concurrency` from `args` to `_execute_task_configs`:

```python
_, best_configs, pareto_fronts, _, _ = _execute_task_configs(
    task_configs,
    args.mode,
    top_n=args.top_n,
    max_concurrency=getattr(args, "max_concurrency", None),
)
```

### 3.4 Python API Update

**File:** `src/aiconfigurator/cli/api.py` — `cli_default()`

Add `max_concurrency: int | None = None` parameter and pass it through:

```python
def cli_default(
    ...,
    max_concurrency: int | None = None,
) -> CLIResult:
```

### 3.5 Error Handling

| Condition | Response | Recovery |
|-----------|----------|----------|
| `--max-concurrency` filters out ALL configs for an experiment | Warning logged, experiment treated as empty (same as existing "no results" path) | User should increase `--max-concurrency` or relax other SLA constraints |
| `--max-concurrency` <= 0 | argparse rejects via `type=int` + validation | User provides a positive integer |

---

## 4. Alternatives Considered

### Alternative 1: Apply filter inside TaskRunner.run()

**Description:** Filter at the lowest level where pareto_df is produced.

**Pros:**
- Filter is applied once, close to data production

**Cons:**
- `TaskRunner` is a general SDK component; adding CLI-specific filtering there breaks separation of concerns
- Would require threading `max_concurrency` through `TaskConfig` and `TaskRunner`

**Why not chosen:** `TaskRunner` is a reusable SDK component. CLI-level constraints should be applied at the CLI layer.

### Alternative 2: Apply filter in picking functions (`pick_default`/`pick_load_match`)

**Description:** Add `max_concurrency` parameter to the picking module.

**Pros:**
- Clean: picking already filters by SLA constraints

**Cons:**
- Picking functions are standalone and used from external contexts; adding CLI-specific parameters pollutes their API
- Pareto frontier computation happens before picking, so unfiltered high-concurrency points would still appear in Pareto analysis

**Why not chosen:** Filtering before Pareto frontier computation produces cleaner results. The `_execute_task_configs` layer is the right boundary for CLI-level constraints.

---

## 5. Usage Examples

### 5.1 CLI Usage

```bash
# Limit concurrency to at most 64
aiconfigurator cli default \
    --model Qwen/Qwen3-32B-FP8 \
    --total-gpus 32 --system h200_sxm \
    --ttft 300 --tpot 10 \
    --max-concurrency 64

# No concurrency limit (default behavior, unchanged)
aiconfigurator cli default \
    --model Qwen/Qwen3-32B-FP8 \
    --total-gpus 32 --system h200_sxm
```

### 5.2 Python API Usage

```python
from aiconfigurator.cli import cli_default

result = cli_default(
    model_path="Qwen/Qwen3-32B-FP8",
    total_gpus=32,
    system="h200_sxm",
    ttft=300,
    tpot=10,
    max_concurrency=64,
)
print(result.best_configs["disagg"].head())
```

---

## 6. Integration Points

- **Report / save**: No changes needed. The `log_final_summary` and `save_results` functions operate on already-filtered DataFrames, so they automatically reflect the concurrency constraint.
- **Webapp**: Not affected. The webapp uses its own Gradio-based flow and does not call `_execute_task_configs`. The `max_concurrency` filter can be added to the webapp independently in the future.

---

## 7. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `src/aiconfigurator/cli/main.py` | MODIFY | Add `--max-concurrency` arg, pass to `_execute_task_configs`, apply filter on pareto_df |
| `src/aiconfigurator/cli/api.py` | MODIFY | Add `max_concurrency` param to `cli_default()` and `_execute_and_wrap_result()` |
| `tests/unit/cli/test_argument_parsing.py` | MODIFY | Add test for `--max-concurrency` parsing and defaults |

---

## 8. Testing Strategy

### 8.1 Unit Tests

**File:** `tests/unit/cli/test_argument_parsing.py`

```python
def test_max_concurrency_default_is_none(self, cli_parser):
    args = cli_parser.parse_args(
        ["default", "--model-path", "Qwen/Qwen3-32B",
         "--total-gpus", "8", "--system", "h200_sxm"]
    )
    assert args.max_concurrency is None

def test_max_concurrency_set(self, cli_parser):
    args = cli_parser.parse_args(
        ["default", "--model-path", "Qwen/Qwen3-32B",
         "--total-gpus", "8", "--system", "h200_sxm",
         "--max-concurrency", "64"]
    )
    assert args.max_concurrency == 64
```

### 8.2 Test Coverage Requirements

- [x] CLI argument parsing: default is `None`, setting to `64` works
- [x] Type validation: value is `int`
- [x] Integration with existing default values test

---

## 9. Deployment & Rollout

### 9.1 Migration Steps

1. Merge PR
2. No migration needed — new optional flag with `None` default preserves existing behavior

### 9.2 Rollback Procedure

1. Revert the PR — single-commit change, no data migration

### 9.3 Feature Flags

Not applicable. The feature is opt-in via CLI flag; omitting it preserves existing behavior.

---

## 10. Monitoring & Observability

Not applicable. This is a CLI tool, not a service. Existing logging (`logger.info`) provides observability for the filter step.

---

## 11. Benefits

| Benefit | Description |
|---------|-------------|
| Deployment relevance | Users see only configurations that match their expected concurrency, reducing noise |
| Faster decision-making | Fewer irrelevant high-concurrency configs in the output table |
| No performance cost | Filter is a simple DataFrame mask on an already-computed result |

---

## 12. Future Enhancements

1. **`--min-concurrency`:** A lower-bound filter for users who want to ensure a minimum concurrency (e.g., for stress testing).
2. **Webapp integration:** Expose `max_concurrency` as a slider in the Gradio webapp.
3. **YAML exp mode:** Allow `max_concurrency` as a per-experiment parameter in YAML configs.

---

## 13. Open Questions

- [x] Should the filter be applied to `exp` mode as well? **Decision:** Start with `default` mode only. Can be extended to `exp` mode later via YAML-level parameter.

---

## Appendix A: Column Schema Reference

The `concurrency` column exists in all three result schemas:

- `ColumnsStatic[4]` → `concurrency` (per-worker concurrency)
- `ColumnsAgg[4]` → `concurrency` (global_bs = bs × attention_dp × pp)
- `ColumnsDisagg[4]` → `concurrency` (decode concurrency × decode workers)
