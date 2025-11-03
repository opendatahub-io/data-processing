# Subset Selection Scripts

This package provides functionality for selecting diverse subsets of datasets using facility location maximization with embedding-based similarity.

## Overview

The subset selection scripts use advanced machine learning techniques to identify representative samples from large datasets. This is particularly useful for:
- Reducing dataset size while maintaining diversity
- Selecting training data that covers the full distribution
- Creating validation/test sets that represent the full dataset

## Installation

Install the required dependencies:

```bash
pip install -r scripts/subset_selection/requirements.txt
```

## Usage

### Command Line Interface (Recommended)

The easiest way to use subset selection is through the CLI:

```bash
# Basic usage - Select 10% and 50% subsets
python -m scripts.subset_selection.cli \
  --input path/to/dataset.jsonl \
  --subset-sizes "0.1,0.5" \
  --output-dir output/

# Absolute counts - Select exactly 1000 and 5000 samples
python -m scripts.subset_selection.cli \
  --input path/to/dataset.jsonl \
  --subset-sizes "1000,5000" \
  --output-dir output/

# Small dataset (< 100k samples) - adjust epsilon and num_folds
python -m scripts.subset_selection.cli \
  --input path/to/small_dataset.jsonl \
  --subset-sizes "0.5" \
  --epsilon 0.1 \
  --num-folds 10 \
  --output-dir output/

# Multiple files combined
python -m scripts.subset_selection.cli \
  --input file1.jsonl file2.jsonl file3.jsonl \
  --subset-sizes "0.25,0.5" \
  --combine-files \
  --output-dir output/

# Testing mode (no GPU required)
python -m scripts.subset_selection.cli \
  --input dataset.jsonl \
  --subset-sizes "0.1" \
  --testing-mode \
  --output-dir output/
```

#### CLI Options

```
Required:
  --input <file> [<file> ...]    Input file(s) to process (JSONL, JSON, CSV, Parquet)
  --subset-sizes <sizes>         Comma-separated sizes (e.g., "0.1,0.5" or "1000,5000")

Optional:
  --output-dir <dir>             Output directory (default: output)
  --batch-size <int>             Batch size for processing (default: 100000)
  --num-folds <int>              Number of folds/partitions (default: 50)
  --epsilon <float>              Optimization parameter (default: 160.0)
  --num-gpus <int>               Number of GPUs to use (default: auto-detect)
  --combine-files                Combine multiple input files before processing
  --testing-mode                 Enable CPU mode for testing
  --encoder-type <str>           Encoder type (default: arctic)
  --encoder-model <str>          Model name (default: Snowflake/snowflake-arctic-embed-l-v2.0)
  --template-name <str>          Template name (default: conversation)
  --seed <int>                   Random seed (default: 42)
```

### Python API

You can also use subset selection directly in Python:

```python
from scripts import subset_datasets

# Select subsets from your dataset
subset_datasets(
    input_files=["path/to/your/dataset.jsonl"],
    subset_sizes=[0.1, 0.5],  # 10% and 50% of the dataset
)
```

### Advanced Python Configuration

```python
from scripts import (
    subset_datasets,
    BasicConfig,
    EncoderConfig,
    TemplateConfig,
    SystemConfig
)

# Configure subset selection
subset_datasets(
    input_files=["dataset1.jsonl", "dataset2.jsonl"],
    subset_sizes=[1000, 5000],  # Select 1000 and 5000 samples
    output_dir="output",
    batch_size=100000,
    num_folds=50,
    combine_files=False,
    epsilon=160.0,
    encoder_type="arctic",
    encoder_model="Snowflake/snowflake-arctic-embed-l-v2.0",
    template_name="conversation",
)
```

## Configuration

### BasicConfig Parameters

- **`output_dir`**: Directory for output files (default: `"output"`)
- **`batch_size`**: Batch size for processing (default: `100000`)
- **`num_folds`**: Number of folds/partitions for subset selection (default: `50`)
  - The dataset is divided into folds for parallel processing across GPUs
  - **Recommendations based on dataset size:**
    - < 1,000 samples: Use `5-10` folds
    - 1,000-10,000 samples: Use `10-20` folds
    - 10,000-100,000 samples: Use `20-50` folds
    - \> 100,000 samples: Use `50-100` folds (default: 50)
  - More folds = better parallelization but higher memory usage per fold
  - Use fewer folds for small datasets to ensure each fold has enough samples
- **`combine_files`**: Whether to combine multiple input files (default: `False`)
- **`epsilon`**: Epsilon parameter for the LazierThanLazyGreedy optimizer (default: `160.0`)
  - Controls the trade-off between optimization quality and speed
  - **Recommendations based on dataset size:**
    - < 1,000 samples: Use `0.01-0.1`
    - 1,000-10,000 samples: Use `0.1-1.0`
    - 10,000-100,000 samples: Use `1.0-10.0`
    - \> 100,000 samples: Use `160.0` (default)

### EncoderConfig Parameters

- `encoder_type`: Type of encoder to use (default: "arctic")
- `encoder_model`: Model name for the encoder
- `instruction`: Custom instruction for embedding generation
- `testing_mode`: Enable testing mode with CPU support (default: False)

### TemplateConfig Parameters

- `template_name`: Name of the template to use (default: "conversation")
- `templates`: Custom templates for text formatting

### SystemConfig Parameters

- `num_gpus`: Number of GPUs to use (auto-detected by default)
- `seed`: Random seed for reproducibility (default: 42)
- `max_retries`: Maximum number of retries on failure (default: 3)
- `retry_delay`: Delay between retries in seconds (default: 30)

## Package Structure

```
scripts/
├── __init__.py              # Top-level package initialization
└── subset_selection/
    ├── __init__.py          # Subset selection package initialization
    ├── subset_selection.py  # Main subset selection logic
    ├── cli.py              # Command-line interface
    ├── requirements.txt    # Package dependencies
    ├── README.md          # This file
    ├── encoders/
    │   ├── __init__.py     # Encoder registry
    │   └── arctic_encoder.py  # Arctic embedding encoder
    └── utils/
        ├── __init__.py     # Utils initialization
        └── subset_selection_utils.py  # Utility functions
```

## Supported Encoders

Currently supported encoders:
- `arctic`: Snowflake Arctic Embed models

To see all supported encoders:

```python
from scripts import get_supported_encoders
print(get_supported_encoders())
```

## Output Files

The script generates several output files:

1. **Embeddings**: Stored in HDF5 format in `{output_dir}/{dataset_name}/embeddings/`
2. **Metadata**: NPZ files containing indices and gains for each subset
3. **Subset Files**: Dataset subsets in the original file format (JSON, CSV, Parquet)


## Quick Start Example

Using your data file:

```bash
# Navigate to project root
cd /Users/roburishabh/Github/odh-data-processing

# Run subset selection - Select 10% and 50% subsets
python -m scripts.subset_selection.cli \
  --input scripts/subset_selection/data/combined_cut_50x.jsonl \
  --subset-sizes "0.1,0.5" \
  --output-dir scripts/subset_selection/data/output \
  --epsilon 0.1 \
  --num-folds 10

# Check results
ls scripts/subset_selection/data/output/
```

## Notes

- **Dataset Size**: Subset selection is optimized for datasets >100k samples
  - For smaller datasets, adjust `--epsilon` and `--num-folds` accordingly
- **GPU Requirement**: GPU acceleration is required for production use
  - Use `--testing-mode` for CPU fallback (testing only, slower)
- **Multiple GPUs**: Automatically detects and utilizes all available GPUs
  - Override with `--num-gpus` flag if needed
- **Memory**: Each fold processes independently, so more folds = less memory per fold
- **Performance**: 
  - Larger epsilon values = faster but potentially lower quality
  - More folds = better GPU utilization but more overhead

## Acknowledgments

This subset selection implementation is based on the **DataCurate4LLMs** project by Krishna Teja KK:

- **Repository**: https://github.com/krishnatejakk/DataCurate4LLMs
- **Author**: [@krishnatejakk](https://github.com/krishnatejakk)

We've adapted their subset selection approach for integration with the Open Data Hub ecosystem, maintaining compatibility with the original Apache-2.0 license. Special thanks to Krishna Teja KK for making this valuable work open source.
