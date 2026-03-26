# Test Fixtures and Sample Data

This directory contains representative sample data for each stage of the data-processing pipeline. These fixtures serve two purposes:

1. **Documenting data formats** — AI agents and contributors can inspect these files to understand the schema at each pipeline stage without running the full pipeline
2. **Enabling offline tests** — Unit tests can validate parsing, chunking, and transformation logic without network access or GPU resources

## Directory Structure

```
fixtures/
├── docling-pipeline/           Pipeline I/O formats at each stage
│   ├── input/                  Stage 1: Raw input
│   │   └── sample.pdf          Minimal single-page PDF for testing
│   ├── converted/              Stage 2: Docling conversion output
│   │   ├── sample.json         Docling Document JSON (full structured representation)
│   │   └── sample.md           Markdown rendering of the same document
│   └── chunked/                Stage 3: Chunker output
│       └── sample_chunks.jsonl One JSON object per line — semantic chunks for RAG
│
├── subset-selection/           Subset selection I/O formats
│   ├── input.jsonl             Conversation-format dataset (10 records)
│   ├── output_metadata.npz     NumPy archive with indices + gains arrays
│   └── output_subset.jsonl     Selected subset (3 records from input)
│
└── README.md                   This file
```

## Data Format Reference

### Docling Pipeline

| Stage | File | Format | Produced By |
|---|---|---|---|
| Input | `sample.pdf` | PDF | User-provided |
| Converted | `sample.json` | Docling Document JSON | `docling_convert_standard` / `docling_convert_vlm` |
| Converted | `sample.md` | Markdown | `docling_convert_standard` / `docling_convert_vlm` |
| Chunked | `sample_chunks.jsonl` | JSONL (one chunk per line) | `docling_chunk` |

### Subset Selection

| Stage | File | Format | Produced By |
|---|---|---|---|
| Input | `input.jsonl` | JSONL with `messages` array | User-provided |
| Metadata | `output_metadata.npz` | NumPy `.npz` with `indices` and `gains` | `subset_datasets()` |
| Output | `output_subset.jsonl` | JSONL (subset of input) | `subset_datasets()` |

## Using Fixtures in Tests

```python
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"

def test_chunk_schema():
    """Verify chunker output matches expected schema."""
    chunks_file = FIXTURES_DIR / "docling-pipeline" / "chunked" / "sample_chunks.jsonl"
    for line in chunks_file.read_text().splitlines():
        chunk = json.loads(line)
        assert "text" in chunk
        assert "source_document" in chunk
        assert "chunk_index" in chunk
        assert "chunking_config" in chunk

def test_subset_selection_input():
    """Verify subset selection input format."""
    input_file = FIXTURES_DIR / "subset-selection" / "input.jsonl"
    for line in input_file.read_text().splitlines():
        record = json.loads(line)
        assert "messages" in record
        assert isinstance(record["messages"], list)
```

## Updating Fixtures

When pipeline output formats change, regenerate fixtures:

1. **Docling pipeline**: Run `local_run.py` and copy representative output files
2. **Subset selection**: Run CLI with `--testing-mode` on a small dataset

Keep fixtures minimal — just enough to document the schema, not to stress-test.
