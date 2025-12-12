"""
Subset selection package for data processing.

This package provides functionality for selecting diverse subsets of datasets
using facility location maximization with embedding-based similarity.
"""

from .subset_selection import (
    BasicConfig,
    DataProcessor,
    EncoderConfig,
    ProcessingConfig,
    SystemConfig,
    TemplateConfig,
    get_supported_encoders,
    subset_datasets,
)

__all__ = [
    "BasicConfig",
    "DataProcessor",
    "EncoderConfig",
    "ProcessingConfig",
    "SystemConfig",
    "TemplateConfig",
    "get_supported_encoders",
    "subset_datasets",
]

