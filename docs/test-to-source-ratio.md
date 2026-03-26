# Test-to-Source Ratio Improvement

> Principle: Aim for at least 1 test file per 2 source files (ratio >= 0.4 scores 70+).

## Before

| Metric | Value |
|---|---|
| Source files | 12 |
| Test files | 2 (`test_notebook_parameters.py`, `test_notebook_execution.py`) |
| **Ratio** | **0.17** |

## After

| Metric | Value |
|---|---|
| Source files | 12 |
| Test files | 7 |
| **Ratio** | **0.58** |

## New Test Files

| File | Tests | What It Covers |
|---|---|---|
| `tests/test_subset_selection_config.py` | 16 | All 5 config dataclasses (`BasicConfig`, `EncoderConfig`, `TemplateConfig`, `SystemConfig`, `ProcessingConfig`), validation rules, epsilon warnings, default values |
| `tests/test_subset_selection_processor.py` | 18 | `DataProcessor` pure logic: `calculate_subset_size`, `get_subset_name`, `get_dataset_name`, `format_text` with all template types |
| `tests/test_cli.py` | 14 | CLI argument parsing (`parse_args`), subset size string parsing (int/float detection, mixed values, whitespace) |
| `tests/test_encoder_registry.py` | 5 | Encoder registry lookup, unknown encoder errors, error message quality |
| `tests/test_kfp_constants.py` | 4 | `PYTHON_BASE_IMAGE` and `DOCLING_BASE_IMAGE` env var defaults and overrides |

**Total new tests: 57**

## Existing Test Files (unchanged)

| File | Tests | What It Covers |
|---|---|---|
| `tests/test_notebook_parameters.py` | ~6 | Validates all notebooks have papermill `parameters` cell |
| `tests/test_notebook_execution.py` | ~6 | Executes notebooks end-to-end via papermill |

## Source Files and Coverage Map

| Source File | Lines | Test File(s) | Coverage |
|---|---|---|---|
| `scripts/subset_selection/subset_selection.py` | 930 | `test_subset_selection_config.py`, `test_subset_selection_processor.py` | Config validation, pure logic methods |
| `scripts/subset_selection/cli.py` | 153 | `test_cli.py` | Argument parsing, size string parsing |
| `scripts/subset_selection/encoders/__init__.py` | 22 | `test_encoder_registry.py` | Registry lookup, error handling |
| `scripts/subset_selection/encoders/arctic_encoder.py` | 207 | `test_encoder_registry.py` | Class registration (instantiation requires GPU) |
| `scripts/subset_selection/utils/subset_selection_utils.py` | 146 | — | Requires GPU/torch for meaningful tests |
| `kubeflow-pipelines/common/components.py` | 412 | — | KFP serialized components (integration-test scope) |
| `kubeflow-pipelines/common/constants.py` | 9 | `test_kfp_constants.py` | Env var defaults and overrides |
| `kubeflow-pipelines/docling-standard/standard_components.py` | 207 | — | KFP serialized (integration-test scope) |
| `kubeflow-pipelines/docling-standard/standard_convert_pipeline.py` | 121 | — | Pipeline wiring (CI compile test covers this) |
| `kubeflow-pipelines/docling-vlm/vlm_components.py` | 194 | — | KFP serialized (integration-test scope) |
| `kubeflow-pipelines/docling-vlm/vlm_convert_pipeline.py` | 110 | — | Pipeline wiring (CI compile test covers this) |
| `kubeflow-pipelines/docling-standard/local_run.py` | 62 | — | Docker orchestration (manual testing) |

## Design Decisions

1. **Lazy imports in test files**: Config and processor tests use function-level imports to avoid triggering `torch`/`datasets` at module load time, which would fail in environments without GPU dependencies.

2. **Mocked torch in processor tests**: `DataProcessor.__init__` calls `torch.device()` and `torch.manual_seed()`. We mock torch to test pure business logic without GPU requirements.

3. **No tests for KFP components**: KFP `@dsl.component` functions are serialized into containers — unit testing them requires the full container runtime. CI already validates these via pipeline compilation and local runner execution.

4. **No tests for `subset_selection_utils.py`**: The `compute_pairwise_dense` function requires torch tensors and GPU for meaningful performance testing. The `retry_on_exception` decorator is tightly coupled to the `DataProcessor` class. These are better tested as integration tests.

## Running the New Tests

```bash
# All new tests (no GPU required)
pytest tests/test_subset_selection_config.py tests/test_subset_selection_processor.py tests/test_cli.py tests/test_encoder_registry.py tests/test_kfp_constants.py -v

# Quick smoke test
pytest tests/test_cli.py tests/test_subset_selection_config.py -v
```

Note: `test_encoder_registry.py` and `test_kfp_constants.py` require the source packages to be importable (i.e., run from repo root or with appropriate `PYTHONPATH`).
