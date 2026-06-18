# Test Fixtures

Sample data for unit and integration tests. These files represent each stage of the KFP pipeline.

## Directory Structure

```
fixtures/
└── docling-pipeline/          # KFP pipeline stage samples
    ├── input/
    │   └── sample.pdf         # Minimal valid PDF for import_pdfs testing
    ├── converted/
    │   ├── sample.json        # Docling JSON output (from docling_convert_*)
    │   └── sample.md          # Markdown output
    └── chunked/
        └── sample_chunks.jsonl  # HybridChunker output (from docling_chunk)
```

## Usage

Tests reference these files via `Path(__file__).parent / "fixtures"`:

```python
FIXTURES = Path(__file__).parent.parent / "fixtures"
sample_pdf = FIXTURES / "docling-pipeline" / "input" / "sample.pdf"
```

## Guidelines

- Keep fixture files small (under 50KB each)
- Use real but minimal data that exercises the code path
- Never commit sensitive or proprietary data
