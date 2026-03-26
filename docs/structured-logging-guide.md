# Structured Logging Guide for KFP Components

> Reference: [AI Bug Automation Readiness - Code Navigability](https://github.com/ugiordan/ai-bug-automation-readiness/blob/main/docs/TEAM_ACTION_GUIDE.md#6-code-navigability-5)

## Current State

KFP components currently use bare `print()` with `flush=True`:

| File | Lines with bare `print()` |
|---|---|
| `kubeflow-pipelines/common/components.py` | Lines 122, 131, 139, 295, 297, 319, 322, 333, 340, 353, 395, 402, 407, 412 |
| `kubeflow-pipelines/docling-standard/standard_components.py` | Lines 133, 196, 202, 207 |
| `kubeflow-pipelines/docling-vlm/vlm_components.py` | Lines 76, 183, 189, 194 |

**Problem**: Output is not structured, not parseable by log aggregators, and has no log levels.

### KFP Serialization Constraint

KFP `@dsl.component` functions are serialized into isolated containers. They **cannot import project-level modules** — all imports must happen inside the function body. This means:

- A shared logging setup module is not possible for KFP components
- Each component must configure its own logging inline
- `packages_to_install` can add `structlog` but adds container build time

## Recommended Approach: Structured `print()` with JSON

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

## Wrap Errors with Context

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
```

**Pattern**: Always include the component name, the error, and the inputs that caused it.

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
