# Contributing to Data Processing

Thank you for your interest in contributing. This guide covers KFP pipeline development.

## Prerequisites

- Python 3.12
- Git
- [pre-commit](https://pre-commit.com/) (`pip install pre-commit`)
- Docker (only for local pipeline testing via `local_run.py`)

## Development Setup

1. **Fork and clone** the repository:

   ```bash
   git clone https://github.com/<your-fork>/data-processing.git
   cd data-processing
   ```

2. **Create a virtual environment**:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dev dependencies**:

   ```bash
   pip install -r requirements-dev.txt
   ```

   For pipeline work, also install pipeline-specific dependencies:

   ```bash
   pip install -r kubeflow-pipelines/docling-standard/requirements.txt
   # or
   pip install -r kubeflow-pipelines/docling-vlm/requirements.txt
   ```

4. **Install pre-commit hooks**:

   ```bash
   pre-commit install
   ```

## Building and Running Tests

```bash
make help              # Show all available targets
make lint              # Ruff lint + format check
make format-python     # Auto-format Python files
```

Run directly with pytest:

```bash
pytest tests/unit/ -v                    # Unit tests only
pytest tests/unit/test_kfp_constants.py  # Specific test file
```

## Submitting a Pull Request

1. **Open an issue first** describing what you want to change.

2. **Create a feature branch**:

   ```bash
   git checkout -b feature/your-change
   ```

3. **Make your changes** and verify locally:

   ```bash
   make lint              # Must pass
   ```

4. **For pipeline or component changes**, recompile the YAML and commit it:

   ```bash
   cd kubeflow-pipelines/docling-standard && python standard_convert_pipeline.py
   # or
   cd kubeflow-pipelines/docling-vlm && python vlm_convert_pipeline.py
   ```

   CI will fail if compiled YAML doesn't match the committed version.

5. **Push and open a PR**. CI runs automatically:

   | Check | What it validates |
   |---|---|
   | `validate-python.yml` | Ruff lint + format check |
   | `compile-kfp.yml` | Pipeline YAML matches compiled output |

6. **Get one approval** from `@opendatahub-io/odh-data-processing`. PRs are squash-merged via Mergify.

## KFP Component Guidelines

KFP `@dsl.component` functions are serialized into isolated containers:

- **Imports must be inside the function body**, not at the top of the file
- Shared modules (`common/components.py`) only work at compile time
- You'll see `# pylint: disable=import-outside-toplevel` — this is intentional

## Generated Files

Files ending in `_compiled.yaml` are auto-generated. Do not edit them directly — edit the source `.py` file and regenerate.

## Adding a New KFP Component

1. Add the component function to `kubeflow-pipelines/common/components.py` (if shared) or the pipeline-specific `*_components.py` file
2. Use `@dsl.component` with inline imports
3. Wire it into the pipeline definition (`*_convert_pipeline.py`)
4. Recompile: `python *_convert_pipeline.py`
5. Commit both the source `.py` and the compiled `_compiled.yaml`

## Getting Help

- Run `make help` for all available commands
- See `CLAUDE.md` for architecture overview and debugging
- See `docs/maintainers/release-strategy.md` for release process
- Open an issue for questions or problems
