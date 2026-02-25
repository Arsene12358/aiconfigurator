# Max Concurrency Flag for CLI Default Mode — Design Doc

**Author:** AI Agent  
**Reviewers:** —  
**Status:** Draft  
**Version:** 1.0  
**Last Updated:** 2026-02-26

---

## 1. Executive Summary

AIConfigurator's `cli default` mode sweeps over parallelism configurations and batch sizes to find the optimal serving configuration under user-specified SLA constraints (TTFT/TPOT). Today the maximum batch size (which directly corresponds to the concurrency level) used during the sweep is hardcoded at 512. Users who know their deployment will never serve more than, say, 64 concurrent requests have no way to tell AIC to skip irrelevant high-concurrency configurations, leading to wasted search time and potentially selecting configurations optimized for concurrency levels the user will never reach.

This design adds a `--max-concurrency` CLI flag to the `default` mode that caps the maximum concurrency level explored during the configuration search, for both aggregated (agg) and disaggregated (disagg) serving modes.

**Key Goals:**
- Allow users to limit the concurrency search space via `--max-concurrency N`
- Reduce sweep time by pruning batch size candidates beyond `N`
- Surface only configurations achievable at the user's actual deployment concurrency
- Maintain full backward compatibility when the flag is omitted

**Deliverables:**
- Modified CLI parser with `--max-concurrency` argument
- End-to-end plumbing from CLI to TaskConfig to TaskRunner to pareto sweep
- Updated Python API (`cli_default()`) with the new parameter
- Unit tests covering parsing, defaults, and validation

---

## 2. Architecture Overview

### 2.1 Components

| Path | Type | Description |
|------|------|-------------|
| `src/aiconfigurator/cli/main.py` | MODIFY | Add `--max-concurrency` arg, pass through build and sweep |
| `src/aiconfigurator/sdk/task.py` | MODIFY | Store `max_concurrency` in TaskConfig; use in TaskRunner |
| `src/aiconfigurator/sdk/pareto_analysis.py` | MODIFY | Accept `max_batch_size` parameter in `agg_pareto()` |
| `src/aiconfigurator/cli/api.py` | MODIFY | Add `max_concurrency` kwarg to `cli_default()` |
| `tests/unit/cli/test_argument_parsing.py` | MODIFY | Add tests for `--max-concurrency` parsing |

### 2.2 Data Flow

```
CLI --max-concurrency 64
  -> build_default_task_configs(max_concurrency=64)
    -> TaskConfig(max_concurrency=64)
      -> config.max_concurrency = 64
        -> TaskRunner.run_agg():    agg_pareto(max_batch_size=64)
        -> TaskRunner.run_disagg(): decode_max_num_tokens=min(512, 64)
```

---

## 3. Detailed Design

### 3.1 CLI Argument

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `--max-concurrency` | No | None (uses default of 512) | Max concurrency level (batch size) to explore during config search |

### 3.2 Modifications to `cli/main.py`

#### `_add_default_mode_arguments(parser)`
Add argument after `--prefix`:
```python
parser.add_argument(
    "--max-concurrency",
    type=int,
    default=None,
    help="Maximum concurrency level (batch size) to explore during the "
    "configuration search. Limits the search space to configurations "
    "with at most this many concurrent requests. Default: no limit (512).",
)
```

#### `build_default_task_configs()`
Add `max_concurrency: int | None = None` parameter, pass to TaskConfig via common_kwargs.

#### `main()`
Pass `args.max_concurrency` to `build_default_task_configs()`.

### 3.3 Modifications to `sdk/task.py`

#### `TaskConfig.__init__()`
Add `max_concurrency: int | None = None`, store as `self.config.max_concurrency`.

#### `TaskRunner.run_agg()`
Read `max_concurrency` from config, pass as `max_batch_size` to `agg_pareto()`.

#### `TaskRunner.run_disagg()`
Read `max_concurrency` from config, cap `decode_max_num_tokens`.

### 3.4 Modifications to `sdk/pareto_analysis.py`

#### `agg_pareto()`
Add `max_batch_size: int = 512` parameter, pass to `find_best_agg_result_under_constraints()`.

### 3.5 Modifications to `cli/api.py`

#### `cli_default()`
Add `max_concurrency: int | None = None` keyword argument, pass through.

### 3.6 Error Handling

| Error Condition | Response | Recovery |
|-----------------|----------|----------|
| `--max-concurrency` <= 0 | argparse validation error | User corrects value |

---

## 4. Alternatives Considered

### Alternative 1: --max-batch-size
More technically precise but "concurrency" matches user mental model better. Not chosen.

### Alternative 2: YAML config only
Exp mode already supports patches. Default mode needs a dedicated CLI flag. Not chosen.

---

## 5. Usage Examples

```bash
# Limit search to 64 concurrent requests
aiconfigurator cli default \
    --model Qwen/Qwen3-32B --total-gpus 8 --system h200_sxm \
    --ttft 600 --tpot 50 --max-concurrency 64

# Python API
from aiconfigurator.cli.api import cli_default
result = cli_default("Qwen/Qwen3-32B", 8, "h200_sxm", max_concurrency=64)
```

---

## 6. File Change Summary

| File | Action | Description |
|------|--------|-------------|
| src/aiconfigurator/cli/main.py | MODIFY | Add --max-concurrency, plumb through |
| src/aiconfigurator/sdk/task.py | MODIFY | Store and use max_concurrency |
| src/aiconfigurator/sdk/pareto_analysis.py | MODIFY | Parameterize max_batch_size |
| src/aiconfigurator/cli/api.py | MODIFY | Add max_concurrency to cli_default |
| tests/unit/cli/test_argument_parsing.py | MODIFY | Add tests |

---

## 7. Testing Strategy

- CLI parsing: default None, explicit int value, type validation
- TaskConfig: stores max_concurrency on config
- agg_pareto: respects max_batch_size parameter
- Backward compatibility: omitted flag produces identical behavior

---

## 8. Deployment

No migration needed. Purely additive, backward compatible. No feature flags required.

---

## 9. Concurrency to Batch Size Mapping

- **Agg mode**: batch_size = concurrent requests per replica. max_batch_size caps sweep.
- **Disagg mode**: decode_max_batch_size = max decode batch per worker. Capped by max_concurrency.
  prefill_max_batch_size (typically 1) is not affected.
