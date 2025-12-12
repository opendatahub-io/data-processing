#!/usr/bin/env python3
"""
Command-line interface for subset selection.
"""

import argparse
import sys

from .subset_selection import subset_datasets


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Select diverse subsets from datasets using facility location optimization"
    )
    
    # Required arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        nargs="+",
        help="Input file(s) to process (JSONL, JSON, CSV, or Parquet)",
    )
    parser.add_argument(
        "--subset-sizes",
        type=str,
        required=True,
        help="Comma-separated subset sizes (e.g., '0.1,0.5' for 10%% and 50%%, or '1000,5000' for absolute counts)",
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100000,
        help="Batch size for processing (default: 100000)",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=50,
        help="Number of folds for subset selection (default: 50)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=160.0,
        help="Epsilon parameter for optimization (default: 160.0 for large datasets, use 0.1-1.0 for small)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect all available)",
    )
    parser.add_argument(
        "--combine-files",
        action="store_true",
        help="Combine multiple input files before processing",
    )
    parser.add_argument(
        "--testing-mode",
        action="store_true",
        help="Enable testing mode (allows CPU usage, for testing only)",
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="arctic",
        help="Encoder type to use (default: arctic)",
    )
    parser.add_argument(
        "--encoder-model",
        type=str,
        default="Snowflake/snowflake-arctic-embed-l-v2.0",
        help="Encoder model name (default: Snowflake/snowflake-arctic-embed-l-v2.0)",
    )
    parser.add_argument(
        "--template-name",
        type=str,
        default="conversation",
        help="Template name to use (default: conversation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()
    
    # Parse subset sizes
    subset_sizes = []
    for size_str in args.subset_sizes.split(","):
        size_str = size_str.strip()
        if "." in size_str:
            subset_sizes.append(float(size_str))
        else:
            subset_sizes.append(int(size_str))

    print("=="*100)
    print(f"Starting subset selection...")
    print(f"  Input files: {args.input}")
    print(f"  Subset sizes: {subset_sizes}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Number of folds: {args.num_folds}")
    print(f"  Epsilon: {args.epsilon}")
    
    # Build kwargs
    kwargs = {
        "output_dir": args.output_dir,
        "batch_size": args.batch_size,
        "num_folds": args.num_folds,
        "epsilon": args.epsilon,
        "combine_files": args.combine_files,
        "encoder_type": args.encoder_type,
        "encoder_model": args.encoder_model,
        "template_name": args.template_name,
        "seed": args.seed,
    }
    
    if args.num_gpus is not None:
        kwargs["num_gpus"] = args.num_gpus
    
    try:
        subset_datasets(
            input_files=args.input,
            subset_sizes=subset_sizes,
            testing_mode=args.testing_mode,
            **kwargs,
        )
        print(f"\n✓ Subset selection complete! Results saved to {args.output_dir}")
        return 0
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())