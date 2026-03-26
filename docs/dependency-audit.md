# Dependency Complexity Audit

> Principle: Keep direct dependency count reasonable relative to codebase size.

## Codebase Size

| Metric | Count |
|---|---|
| Python source files | 15 |
| Total Python lines | ~2,966 |
| Jupyter notebooks | 6 |
| Requirement files | 5 |

## Dependency Inventory

### By Requirement File

| File | Scope | Direct Deps | Status |
|---|---|---|---|
| `kubeflow-pipelines/docling-standard/requirements.txt` | Pipeline compilation | 5 | OK |
| `kubeflow-pipelines/docling-vlm/requirements.txt` | Pipeline compilation | 4 | OK |
| `scripts/subset_selection/requirements.txt` | Subset selection CLI | 8 | OK |
| `requirements-dev.txt` | Dev/CI tooling | 7 | OK |
| `tests/requirements-gpu.txt` | GPU test runner | 3 | OK |

**Total unique direct dependencies: 21** (across 5 scoped requirement files)

### Inline Dependencies (KFP `packages_to_install`)

| Component | Packages | Purpose |
|---|---|---|
| `import_pdfs` | `boto3`, `requests` | S3 and HTTP downloads at runtime |

### Notebook Dependencies (self-installing via `%pip install`)

Each notebook installs its own dependencies inline. This is standard for notebook-based projects and doesn't add to the repository's dependency footprint.

## Assessment: Healthy

The dependency count is **well-proportioned** for a ~3,000-line codebase with 3 distinct subsystems:

| Subsystem | Deps | Lines of Code | Ratio |
|---|---|---|---|
| KFP Pipelines | 5 | ~1,150 | 1 dep per 230 lines |
| Subset Selection | 8 | ~1,450 | 1 dep per 180 lines |
| Dev Tooling | 7 | (tooling) | N/A |
| GPU Tests | 3 | (testing) | N/A |

**Positive patterns**:
- Dependencies are **scoped per subsystem** — each has its own `requirements.txt` instead of one monolithic file
- KFP pipeline deps are **exact-pinned** (`==`) for reproducible compilation
- No unused dependencies detected — every package maps to actual imports
- No "kitchen sink" frameworks — each dep serves a clear purpose

## Findings

### 1. Version Pinning Inconsistency (minor)

| Strategy | Where Used | Packages |
|---|---|---|
| Exact pin (`==`) | KFP pipelines | `docling`, `kfp`, `kfp-kubernetes`, `boto3`, `tesserocr` |
| Floor pin (`>=`) | Dev tools, subset selection | `torch`, `numpy`, `datasets`, `ruff`, `pytest`, etc. |
| Unpinned | Subset selection, GPU tests | `submodlib-py`, `torch` (in GPU tests), `torchvision`, `torchaudio` |

**Recommendation**: Pin `submodlib-py` to a minimum version. It's the most fragile dependency (must be built from source per the inline comment) and an unpinned version makes builds non-reproducible.

```diff
- submodlib-py
+ submodlib-py>=1.1.0
```

### 2. Duplicate Dependencies Across Pipeline Files (acceptable)

`docling-standard/requirements.txt` and `docling-vlm/requirements.txt` share 4 of 5 dependencies:

```
docling == 2.57.0       # both
kfp == 2.14.6           # both
kfp-kubernetes == 2.14.6 # both
boto3 == 1.40.52        # both
tesserocr == 2.9.1      # standard only
```

This duplication is **intentional and correct** — each pipeline is independently installable and compilable. A shared `requirements-base.txt` with `-r ../requirements-base.txt` could reduce duplication but adds indirection. The current approach is clearer for a 2-pipeline project.

### 3. `torch` Appears in Two Places with Different Pins (caution)

| File | Pin |
|---|---|
| `scripts/subset_selection/requirements.txt` | `torch>=2.0.0` |
| `tests/requirements-gpu.txt` | `torch` (unpinned, from PyTorch CUDA index) |

These serve different purposes (CPU-compatible vs CUDA-specific) but could diverge. Consider adding a comment to `tests/requirements-gpu.txt` explaining the relationship:

```diff
  --index-url https://download.pytorch.org/whl/cu121
+ # CUDA-enabled PyTorch for GPU CI runners
+ # Must be compatible with torch>=2.0.0 in scripts/subset_selection/requirements.txt
  torch
  torchvision
  torchaudio
```

### 4. `jupyter` in Dev Dependencies is Heavy (optional cleanup)

`requirements-dev.txt` includes `jupyter>=1.0.0`, which is a metapackage that installs `notebook`, `qtconsole`, `ipywidgets`, and more. If only `ipykernel` is needed for notebook execution in CI (which `papermill` already handles), this could be replaced:

```diff
  papermill>=2.4.0
  ipykernel>=6.20.0
- jupyter>=1.0.0
```

This would reduce the transitive dependency tree significantly. Verify by checking if any CI workflow or test directly invokes `jupyter` (vs just `papermill`).

## Dependency Graph (Logical)

```
requirements-dev.txt (Dev/CI)
├── ruff            — formatting
├── nbstripout      — notebook output stripping
├── pytest          — test runner
├── nbformat        — notebook parsing
├── papermill       — notebook execution
├── ipykernel       — Python kernel for papermill
└── jupyter         — (potentially removable)

kubeflow-pipelines/docling-standard/requirements.txt (Pipeline compilation)
├── docling         — document conversion engine
├── kfp             — Kubeflow Pipelines SDK
├── kfp-kubernetes  — KFP Kubernetes extensions
├── boto3           — S3 access
└── tesserocr       — OCR engine bindings

kubeflow-pipelines/docling-vlm/requirements.txt (Pipeline compilation)
├── docling         — document conversion engine (VLM mode)
├── kfp             — Kubeflow Pipelines SDK
├── kfp-kubernetes  — KFP Kubernetes extensions
└── boto3           — S3 access

scripts/subset_selection/requirements.txt (Subset selection)
├── torch           — tensor operations, GPU compute
├── transformers    — model loading (Arctic encoder)
├── numpy           — numerical arrays
├── datasets        — HuggingFace dataset loading
├── h5py            — HDF5 embedding storage
├── submodlib-py    — facility location optimization
├── jinja2          — text templating
└── tqdm            — progress bars

tests/requirements-gpu.txt (GPU CI)
├── torch           — CUDA-enabled PyTorch
├── torchvision     — (transitive, for CUDA wheel index)
└── torchaudio      — (transitive, for CUDA wheel index)
```

## Summary

| Metric | Value | Assessment |
|---|---|---|
| Total unique direct deps | 21 | Healthy for ~3K lines across 3 subsystems |
| Max deps in one file | 8 (`subset_selection`) | Reasonable for ML workload |
| Unused deps detected | 0 | Clean |
| Pinning strategy | Mixed (exact for KFP, floor for scripts) | Appropriate per context |
| Actionable items | 3 | Pin `submodlib-py`, comment `torch` overlap, consider dropping `jupyter` |
