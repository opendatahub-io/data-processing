# Data Processing Pipeline Overview

This document describes the standard data processing pipeline for converting PDF documents into structured formats suitable for retrieval-augmented generation (RAG) workflows.

| Format | Extension | Use Case |
|---|---|---|
| Docling JSON | .json | Lossless document representation |
| Markdown | .md | Human-readable output |
| Chunked JSONL | .jsonl | RAG-ready semantic chunks |

The pipeline supports multiple output formats including Markdown and structured JSON for downstream consumption.
