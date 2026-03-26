# Structured Logging Guide for data-processing

> Reference: [AI Bug Automation Readiness - Code Navigability](https://github.com/ugiordan/ai-bug-automation-readiness/blob/main/docs/TEAM_ACTION_GUIDE.md#6-code-navigability-5)

## Current State

The codebase has two distinct logging patterns:

| Area | Current Pattern | Problem |
|---|---|---|
| **KFP components** (`kubeflow-pipelines/`) | Bare `print()` with `flush=True` | Not structured, not parseable, no log levels |
| **Scripts** (`scripts/subset_selection/`) | `logging.basicConfig()` with format string | Better, but not structured (no JSON), duplicated `basicConfig` calls across modules |
| **CLI** (`scripts/subset_selection/cli.py`) | Bare `print()` | No log levels, mixed with operational output |
| **Tests** (`tests/`) | Bare `print()` | Acceptable for test output |

### KFP Component Constraint

KFP `@dsl.component` functions are serialized and executed in isolated containers. They **cannot import project-level modules** — all imports must happen inside the function body. This means:

- A shared logging setup module is not possible for KFP components
- Each component must configure its own logging inline
- `packages_to_install` can add `structlog` but adds container build time

## Recommended Approach

### 1. Scripts: Migrate to `structlog` (highest impact)

The `scripts/subset_selection/` directory is a standalone Python package with its own `requirements.txt`. This is the best candidate for structured logging.

**Add to `scripts/subset_selection/requirements.txt`:**

```
structlog>=24.0.0
```

**Create `scripts/subset_selection/log_config.py`:**

```python
import structlog
import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for structured JSON logging."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
```

**Replace in each script module (e.g., `subset_selection.py`):**

```python
# BEFORE
import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info(f"Processing dataset: {dataset_name}")
logger.error(f"Error processing files: {str(e)}")

# AFTER
import structlog
logger = structlog.get_logger()
logger.info("processing_dataset", dataset_name=dataset_name)
logger.error("processing_failed", error=str(e), dataset_name=dataset_name)
```

**Key changes:**
- Event names become machine-parseable keys (`"processing_dataset"` not `f"Processing dataset: {name}"`)
- Context is passed as keyword arguments, not interpolated into strings
- Errors include structured fields that log aggregators can filter on
- Remove all `logging.basicConfig()` calls — configure once in `cli.py`

**Update `scripts/subset_selection/cli.py`:**

```python
# BEFORE
print("=="*100)
print(f"Starting subset selection...")
print(f"  Input files: {args.input}")

# AFTER
from .log_config import configure_logging
configure_logging()
logger = structlog.get_logger()
logger.info("starting_subset_selection",
    input_files=args.input,
    subset_sizes=subset_sizes,
    output_dir=args.output_dir,
    num_folds=args.num_folds,
    epsilon=args.epsilon,
)
```

### Files to change for scripts migration

| File | What to change |
|---|---|
| `scripts/subset_selection/requirements.txt` | Add `structlog>=24.0.0` |
| `scripts/subset_selection/log_config.py` | **New file** — shared config |
| `scripts/subset_selection/cli.py:115-121` | Replace `print()` calls with `structlog` |
| `scripts/subset_selection/subset_selection.py:40-43` | Remove `logging.basicConfig`, use `structlog.get_logger()` |
| `scripts/subset_selection/utils/subset_selection_utils.py:14-17` | Remove `logging.basicConfig`, use `structlog.get_logger()` |
| `scripts/subset_selection/encoders/arctic_encoder.py:15` | Replace `logging.getLogger` with `structlog.get_logger()` |

### 2. KFP Components: Structured `print()` with JSON (pragmatic)

Given the KFP serialization constraint, the pragmatic approach is structured JSON via `print()` + `json.dumps()`. This keeps components self-contained while producing parseable output.

**Pattern for KFP components:**

```python
@dsl.component(base_image=PYTHON_BASE_IMAGE)
def import_pdfs(...):
    import json
    import sys
    from datetime import datetime, timezone

    def log(level: str, event: str, **kwargs):
        """Emit a structured JSON log line."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "component": "import_pdfs",
            "event": event,
            **kwargs,
        }
        print(json.dumps(entry), flush=True, file=sys.stderr if level == "ERROR" else sys.stdout)

    # BEFORE
    print(f"import-test-pdfs: downloading {url} -> {dest}", flush=True)

    # AFTER
    log("INFO", "downloading_file", url=url, dest=str(dest), source="http")
```

**Example output:**

```json
{"timestamp": "2026-03-26T14:30:00+00:00", "level": "INFO", "component": "import_pdfs", "event": "downloading_file", "url": "https://...", "dest": "/tmp/out/doc.pdf", "source": "http"}
{"timestamp": "2026-03-26T14:30:05+00:00", "level": "INFO", "component": "import_pdfs", "event": "download_complete", "files_count": 3}
{"timestamp": "2026-03-26T14:30:05+00:00", "level": "ERROR", "component": "import_pdfs", "event": "download_failed", "url": "https://...", "error": "HTTP 404"}
```

### Files to change for KFP components

| File | Lines with bare `print()` |
|---|---|
| `kubeflow-pipelines/common/components.py` | Lines 122, 131, 139, 295, 297, 319, 322, 333, 340, 353, 395, 402, 407, 412 |
| `kubeflow-pipelines/docling-standard/standard_components.py` | Lines 133, 196, 202, 207 |
| `kubeflow-pipelines/docling-vlm/vlm_components.py` | Lines 76, 183, 189, 194 |

### 3. Wrap Errors with Context

Currently errors are raised with basic messages. Add context about what operation failed:

```python
# BEFORE (common/components.py:37-39)
raise ValueError(
    "filenames must contain at least one filename (comma-separated)"
)

# AFTER
raise ValueError(
    f"import_pdfs failed: filenames must contain at least one filename "
    f"(comma-separated), got: {filenames!r}"
)

# BEFORE (subset_selection.py:502)
logger.error(f"Error processing files: {str(e)}")

# AFTER
logger.error("file_processing_failed",
    error=str(e),
    error_type=type(e).__name__,
    input_files=input_files,
)
```

**Pattern**: Always include the operation name, the error, and the inputs that caused it.

## Priority Order

1. **Scripts (`subset_selection/`)** — Highest impact. Standalone package, long-running GPU workloads where structured logs are critical for debugging. Migrate to `structlog`.
2. **KFP components** — Medium impact. Add inline `log()` helper with JSON output. Each component is isolated, so this is incremental.
3. **CLI output** — Low impact. Replace bare `print()` in `cli.py` with `structlog`.
4. **Tests** — Skip. Bare `print()` in test files is fine.

## Structured Log Output Comparison

```
# BEFORE (bare print)
import-test-pdfs: downloading https://example.com/doc.pdf -> /tmp/doc.pdf

# AFTER (structured JSON)
{"timestamp": "2026-03-26T14:30:00+00:00", "level": "INFO", "component": "import_pdfs", "event": "downloading_file", "url": "https://example.com/doc.pdf", "dest": "/tmp/doc.pdf"}
```

The structured format enables:
- Log aggregation tools (CloudWatch, Loki, Splunk) to parse and filter
- AI agents to programmatically search for specific error events
- Correlation across pipeline components by structured fields
- Severity-based alerting without regex
