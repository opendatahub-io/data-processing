# Code Navigability Refactoring Guide

> Target: All source files under 500 lines for AI agent comprehension.
> Reference: [AI Bug Automation Readiness - Code Navigability](https://github.com/ugiordan/ai-bug-automation-readiness/blob/main/docs/TEAM_ACTION_GUIDE.md#6-code-navigability-5)

## Audit Results

### Files Over 500 Lines

| File | Lines | Action |
|---|---|---|
| `scripts/subset_selection/subset_selection.py` | **930** | Split into 4 focused modules |

### Files Under Threshold (no action needed)

| File | Lines |
|---|---|
| `kubeflow-pipelines/common/components.py` | 412 |
| `scripts/subset_selection/encoders/arctic_encoder.py` | 207 |
| `kubeflow-pipelines/docling-standard/standard_components.py` | 207 |
| `kubeflow-pipelines/docling-vlm/vlm_components.py` | 194 |
| All other `.py` files | <160 |

### KFP Components Note

`components.py` (412 lines) cannot be trivially split because each `@dsl.component` function must be self-contained for KFP serialization (see [ADR-001](docs/adr/001-kfp-component-serialization.md)). The 4 components in this file (`import_pdfs`, `create_pdf_splits`, `download_docling_models`, `docling_chunk`) are already logically independent — they just share a file for organizational convenience. At 412 lines, it's comfortably under the threshold.

---

## Refactoring Plan: `subset_selection.py` (930 → 4 files, all under 500)

### Current Structure

```
subset_selection.py (930 lines)
├── Lines 46-174:   5 config dataclasses
├── Lines 176-585:  DataProcessor class (orchestration)
├── Lines 587-697:  _process_dataset_shard() (multiprocessing worker)
├── Lines 700-743:  _merge_shard_files() (HDF5 merge utility)
├── Lines 746-858:  process_folds_with_gpu() (multiprocessing worker)
├── Lines 861-869:  get_supported_encoders() (utility)
└── Lines 872-931:  subset_datasets() (public API entry point)
```

### Proposed Structure

```
scripts/subset_selection/
├── __init__.py                    (update re-exports)
├── config.py                  NEW (~130 lines) — Configuration dataclasses
├── embedding_workers.py       NEW (~160 lines) — GPU/CPU embedding generation
├── selection_workers.py       NEW (~115 lines) — Fold-level facility location
├── subset_selection.py        SHRINK (~350 lines) — DataProcessor + public API
├── cli.py                         (no changes)
├── encoders/                      (no changes)
└── utils/                         (no changes)
```

### File-by-File Breakdown

---

#### 1. `config.py` (~130 lines) — NEW

**What moves here**: All 5 dataclasses from lines 46–174.

**Contents**:
- `BasicConfig` (lines 46–86)
- `EncoderConfig` (lines 89–101)
- `TemplateConfig` (lines 104–116)
- `SystemConfig` (lines 119–131)
- `ProcessingConfig` (lines 134–174)

**Imports needed**:
```python
from dataclasses import dataclass, field
from typing import Dict, List, Union
import logging

from .utils.subset_selection_utils import get_default_num_gpus

logger = logging.getLogger(__name__)
```

**Why this is a clean cut**: Config classes have no dependencies on `DataProcessor`, `torch`, `numpy`, or any heavy imports. They only depend on `get_default_num_gpus` from utils.

---

#### 2. `embedding_workers.py` (~160 lines) — NEW

**What moves here**: The two standalone functions for embedding generation (lines 587–743).

**Contents**:
- `_process_dataset_shard()` (lines 587–697) — Multiprocessing worker that generates embeddings on a single GPU/CPU
- `_merge_shard_files()` (lines 700–743) — Merges HDF5 shard files into a single embeddings file

**Imports needed**:
```python
import logging
import os

from jinja2 import BaseLoader, Environment
from tqdm import tqdm
import h5py
import numpy as np
import torch

from .encoders import get_encoder_class

logger = logging.getLogger(__name__)
```

**Why this is a clean cut**: Both functions are standalone (not methods on `DataProcessor`). They are called via `multiprocessing.Pool.map()`, so they must be top-level functions in a module — moving them to their own module is natural. `DataProcessor.generate_embeddings()` will import them:

```python
from .embedding_workers import _process_dataset_shard, _merge_shard_files
```

---

#### 3. `selection_workers.py` (~115 lines) — NEW

**What moves here**: The fold-processing worker function (lines 746–858).

**Contents**:
- `process_folds_with_gpu()` (lines 746–858) — Multiprocessing worker that runs facility location optimization on assigned folds

**Imports needed**:
```python
import gc
import logging
import math

import torch

from .utils.subset_selection_utils import compute_pairwise_dense

logger = logging.getLogger(__name__)
```

**Why this is a clean cut**: This function is a standalone multiprocessing worker (called via `Pool.map()`). It has a single focused responsibility: compute similarity matrices and run `FacilityLocationFunction` on assigned folds. Its only internal dependency is `compute_pairwise_dense` from utils.

```python
# In subset_selection.py:
from .selection_workers import process_folds_with_gpu
```

---

#### 4. `subset_selection.py` (~350 lines) — SHRINK (from 930)

**What remains**:
- `DataProcessor` class (lines 176–585) — orchestration logic
- `get_supported_encoders()` (lines 861–869) — utility
- `subset_datasets()` (lines 872–931) — public API entry point

**Updated imports** (replacing inline code with module imports):
```python
from .config import BasicConfig, EncoderConfig, TemplateConfig, SystemConfig, ProcessingConfig
from .embedding_workers import _process_dataset_shard, _merge_shard_files
from .selection_workers import process_folds_with_gpu
```

**What changes in `DataProcessor`**:
- `generate_embeddings()` method calls `_process_dataset_shard` and `_merge_shard_files` — these are already standalone functions, just imported from a different module now
- `select_subsets()` method calls `process_folds_with_gpu` — same pattern

---

#### 5. `__init__.py` — UPDATE

Add re-exports for the new modules so the public API doesn't change:

```python
from .config import (
    BasicConfig,
    EncoderConfig,
    TemplateConfig,
    SystemConfig,
    ProcessingConfig,
)
from .subset_selection import (
    DataProcessor,
    get_supported_encoders,
    subset_datasets,
)
```

This ensures existing imports like `from scripts.subset_selection import subset_datasets` continue to work.

---

### Migration Checklist

- [ ] Create `config.py` with all 5 dataclasses
- [ ] Create `embedding_workers.py` with `_process_dataset_shard` and `_merge_shard_files`
- [ ] Create `selection_workers.py` with `process_folds_with_gpu`
- [ ] Remove moved code from `subset_selection.py`
- [ ] Add imports for moved code in `subset_selection.py`
- [ ] Update `__init__.py` re-exports
- [ ] Move the `multiprocessing.set_start_method('spawn')` call to `subset_selection.py` (it stays in the module that creates `Pool`)
- [ ] Move the `logging.basicConfig()` call to `subset_selection.py` only (remove duplicates)
- [ ] Verify `cli.py` imports still work (it imports `subset_datasets` from `__init__.py`)
- [ ] Run `pytest tests/ -v` to verify no regressions
- [ ] Run `make format-python-check` to verify formatting

### Risks and Mitigations

| Risk | Mitigation |
|---|---|
| **Multiprocessing pickling** — Workers must be importable top-level functions | Already the case; moving to separate modules doesn't change this |
| **Circular imports** — `config.py` depends on `utils`, `subset_selection.py` depends on `config.py` | No cycle: `utils` → `config` → `embedding_workers` / `selection_workers` → `subset_selection` (linear dependency chain) |
| **Public API breakage** — External code imports from `__init__.py` | Updated `__init__.py` re-exports all public symbols from new locations |
| **Test breakage** — Tests reference `subset_selection.py` directly | Tests use `subset_datasets()` via `__init__.py`, not direct file imports |

### Line Count Summary

| File | Before | After |
|---|---|---|
| `subset_selection.py` | 930 | ~350 |
| `config.py` | — | ~130 |
| `embedding_workers.py` | — | ~160 |
| `selection_workers.py` | — | ~115 |
| **Total** | **930** | **~755** (slight increase from import boilerplate) |

All files well under the 500-line threshold. The largest file drops from 930 to ~350 lines.
