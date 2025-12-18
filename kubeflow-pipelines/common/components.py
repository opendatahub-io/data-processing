from typing import List

from kfp import dsl

from .constants import DOCLING_BASE_IMAGE, PYTHON_BASE_IMAGE


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
    packages_to_install=["boto3", "requests"],  # Required for S3 and HTTP downloads
)
def import_pdfs(
    output_path: dsl.Output[dsl.Artifact],
    filenames: str,
    base_url: str,
    from_s3: bool = False,
    s3_secret_mount_path: str = "/mnt/secrets",
):
    """
    Import PDF filenames (comma-separated) from specified URL or S3 bucket.

    Args:
        filenames: List of PDF filenames to import.
        base_url: Base URL of the PDF files.
        output_path: Path to the output directory for the PDF files.
        from_s3: Whether or not to import from S3.
        s3_secret_mount_path: Path to the secret mount path for the S3 credentials.
    """
    import os  # pylint: disable=import-outside-toplevel
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    import boto3  # pylint: disable=import-outside-toplevel
    import requests  # pylint: disable=import-outside-toplevel

    filenames_list = [name.strip() for name in filenames.split(",") if name.strip()]
    if not filenames_list:
        raise ValueError(
            "filenames must contain at least one filename (comma-separated)"
        )

    output_path_p = Path(output_path.path)
    output_path_p.mkdir(parents=True, exist_ok=True)

    if from_s3:
        if not os.path.exists(s3_secret_mount_path):
            raise ValueError(
                f"Secret for S3 should be mounted in {s3_secret_mount_path}"
            )

        s3_endpoint_url_secret = "S3_ENDPOINT_URL"
        s3_endpoint_url_file_path = os.path.join(
            s3_secret_mount_path, s3_endpoint_url_secret
        )
        if os.path.isfile(s3_endpoint_url_file_path):
            with open(s3_endpoint_url_file_path) as f:
                s3_endpoint_url = f.read()
        else:
            raise ValueError(
                f"Key {s3_endpoint_url_secret} not defined in secret {s3_secret_mount_path}"
            )

        s3_access_key_secret = "S3_ACCESS_KEY"
        s3_access_key_file_path = os.path.join(
            s3_secret_mount_path, s3_access_key_secret
        )
        if os.path.isfile(s3_access_key_file_path):
            with open(s3_access_key_file_path) as f:
                s3_access_key = f.read()
        else:
            raise ValueError(
                f"Key {s3_access_key_secret} not defined in secret {s3_secret_mount_path}"
            )

        s3_secret_key_secret = "S3_SECRET_KEY"
        s3_secret_key_file_path = os.path.join(
            s3_secret_mount_path, s3_secret_key_secret
        )
        if os.path.isfile(s3_secret_key_file_path):
            with open(s3_secret_key_file_path) as f:
                s3_secret_key = f.read()
        else:
            raise ValueError(
                f"Key {s3_secret_key_secret} not defined in secret {s3_secret_mount_path}"
            )

        s3_bucket_secret = "S3_BUCKET"
        s3_bucket_file_path = os.path.join(s3_secret_mount_path, s3_bucket_secret)
        if os.path.isfile(s3_bucket_file_path):
            with open(s3_bucket_file_path) as f:
                s3_bucket = f.read()
        else:
            raise ValueError(
                f"Key {s3_bucket_secret} not defined in secret {s3_secret_mount_path}"
            )

        s3_prefix_secret = "S3_PREFIX"
        s3_prefix_file_path = os.path.join(s3_secret_mount_path, s3_prefix_secret)
        if os.path.isfile(s3_prefix_file_path):
            with open(s3_prefix_file_path) as f:
                s3_prefix = f.read()
        else:
            raise ValueError(
                f"Key {s3_prefix_secret} not defined in secret {s3_secret_mount_path}"
            )

        if not s3_endpoint_url:
            raise ValueError("S3_ENDPOINT_URL must be provided")

        if not s3_bucket:
            raise ValueError("S3_BUCKET must be provided")

        s3_client = boto3.client(
            "s3",
            endpoint_url=s3_endpoint_url,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
        )

        for filename in filenames_list:
            orig = f"{s3_prefix.rstrip('/')}/{filename.lstrip('/')}"
            dest = output_path_p / filename
            print(f"import-test-pdfs: downloading {orig} -> {dest} from s3", flush=True)
            s3_client.download_file(s3_bucket, orig, dest)
    else:
        if not base_url:
            raise ValueError("base_url must be provided")

        for filename in filenames_list:
            url = f"{base_url.rstrip('/')}/{filename.lstrip('/')}"
            dest = output_path_p / filename
            print(f"import-test-pdfs: downloading {url} -> {dest}", flush=True)
            with requests.get(url, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

    print("import-test-pdfs: done", flush=True)


@dsl.component(
    base_image=PYTHON_BASE_IMAGE,
)
def create_pdf_splits(
    input_path: dsl.Input[dsl.Artifact],
    num_splits: int,
) -> List[List[str]]:
    """
    Create a list of PDF splits.

    Args:
        input_path: Path to the input directory containing PDF files.
        num_splits: Number of splits to create.
    """
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    input_path_p = Path(input_path.path)

    all_pdfs = [path.name for path in input_path_p.glob("*.pdf")]
    all_splits = [all_pdfs[i::num_splits] for i in range(num_splits)]
    filled_splits = list(filter(None, all_splits))
    return filled_splits


@dsl.component(
    base_image=DOCLING_BASE_IMAGE,
)
def download_docling_models(
    output_path: dsl.Output[dsl.Artifact],
    pipeline_type: str = "standard",
    remote_model_endpoint_enabled: bool = False,
):
    """
    Download Docling models based on pipeline type and configuration.

    This unified component handles model downloading for different pipeline types:
    - standard : Download traditional Docling models (layout, tableformer, easyocr)
    - vlm : Download Docling VLM models (smolvlm, smoldocling) for local inference
             When remote_model_endpoint_enabled=True, downloads minimal models for remote inference

    Args:
        output_path: Path to the output directory for Docling models
        pipeline_type: Type of pipeline (standard, vlm)
        remote_model_endpoint_enabled: Whether to download remote model endpoint models (VLM only)
    """
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    from docling.utils.model_downloader import (
        download_models,
    )  # pylint: disable=import-outside-toplevel

    output_path_p = Path(output_path.path)
    output_path_p.mkdir(parents=True, exist_ok=True)

    if pipeline_type == "standard":
        # Standard pipeline: download traditional models
        download_models(
            output_dir=output_path_p,
            progress=True,
            with_layout=True,
            with_tableformer=True,
            with_easyocr=False,
        )
    elif pipeline_type == "vlm" and remote_model_endpoint_enabled:
        # VLM pipeline with remote model endpoint: Download minimal required models
        # Only models set are what lives in fabianofranz repo
        # TODO: figure out what needs to be downloaded or removed
        download_models(
            output_dir=output_path_p,
            progress=False,
            force=False,
            with_layout=True,
            with_tableformer=True,
            with_code_formula=False,
            with_picture_classifier=False,
            with_smolvlm=False,
            with_smoldocling=False,
            with_smoldocling_mlx=False,
            with_granite_vision=False,
            with_easyocr=False,
        )
    elif pipeline_type == "vlm":
        # VLM pipeline with local models: Download VLM models for local inference
        # TODO: set models downloaded by model name passed into KFP pipeline ex: smoldocling OR granite-vision
        download_models(
            output_dir=output_path_p,
            with_smolvlm=True,
            with_smoldocling=True,
            progress=False,
            force=False,
            with_layout=False,
            with_tableformer=False,
            with_code_formula=False,
            with_picture_classifier=False,
            with_smoldocling_mlx=False,
            with_granite_vision=False,
            with_easyocr=False,
        )
    else:
        raise ValueError(
            f"Invalid pipeline_type: {pipeline_type}. Must be 'standard' or 'vlm'"
        )


@dsl.component(
    base_image=DOCLING_BASE_IMAGE,
)
def docling_chunk(
    input_path: dsl.Input[dsl.Artifact],
    output_path: dsl.Output[dsl.Artifact],
    max_tokens: int = 512,
    merge_peers: bool = True,
):
    """
    Chunk Docling documents using HybridChunker. Takes converted docling JSON files as input
    and produces chunked JSONL files with semantic chunks suitable for RAG.

    Output format is JSONL (one JSON object per line) for easy inspection and streaming.

    Args:
        input_path: Path to the input directory containing Docling JSON files
        output_path: Path to the output directory for the chunked JSONL files
        max_tokens: Maximum number of tokens per chunk
        merge_peers: Whether to merge smaller chunks at the same level
    """
    import json  # pylint: disable=import-outside-toplevel
    from datetime import datetime, timezone  # pylint: disable=import-outside-toplevel
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    # HybridChunker = Docling's smart chunking class that combines:
    # 1. Document structure awareness
    # 2. Token-based splitting
    from docling.chunking import HybridChunker  # pylint: disable=import-outside-toplevel
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )  # pylint: disable=import-outside-toplevel
    from docling_core.types import DoclingDocument  # pylint: disable=import-outside-toplevel
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    # Convert KFP artifact paths to Path objects
    input_path_p = Path(input_path.path)
    output_path_p = Path(output_path.path)
    output_path_p.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer for HybridChunker (new API)
    # Using a lightweight sentence-transformer model for tokenization
    EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        hf_tokenizer = AutoTokenizer.from_pretrained(
            EMBED_MODEL_ID,
            resume_download=True,
            timeout=60,
        )
        print(f"docling-chunk: loaded tokenizer from {EMBED_MODEL_ID}", flush=True)
    except Exception as e:
        print(f"docling-chunk: ERROR loading tokenizer: {e}", flush=True)
        raise RuntimeError(
            f"Failed to load tokenizer model {EMBED_MODEL_ID}. "
            "Ensure network access to HuggingFace Hub or pre-download the model."
        ) from e

    tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=max_tokens,
    )

    # Initialize Hybrid chunker with user-specified parameters
    # tokenizer: The tokenizer wrapper to use for counting tokens (includes max_tokens)
    # merge_peers: if true, smaller adjacent chunks will be merged together
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=merge_peers,
    )

    # Find all JSON files in the input directory
    json_files = list(input_path_p.glob("*.json"))
    if not json_files:
        print(f"docling-chunk: No JSON files found in {input_path_p}", flush=True)
        return

    print(
        f"docling-chunk: processing {len(json_files)} files with max_tokens={max_tokens} and merge_peers={merge_peers}",
        flush=True,
    )

    # Track processing results
    processed_count = 0
    skipped_files = []

    # Process each file
    for json_file in json_files:
        print(f"docling-chunk: processing {json_file}", flush=True)

        # Load and validate the JSON file
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                doc_data = json.load(f)
        except json.JSONDecodeError as e:
            print(
                f"docling-chunk: skipping {json_file.name} - invalid JSON: {e}",
                flush=True,
            )
            skipped_files.append((json_file.name, f"invalid JSON: {e}"))
            continue

        # Parse the JSON data into a DoclingDocument object
        # This validates that the JSON conforms to the DoclingDocument schema
        try:
            doc = DoclingDocument.model_validate(doc_data)
        except Exception as e:
            # Catches pydantic.ValidationError and any other validation issues
            print(
                f"docling-chunk: skipping {json_file.name} - not a valid DoclingDocument: {e}",
                flush=True,
            )
            skipped_files.append((json_file.name, f"validation failed: {e}"))
            continue

        # Chunk the document using HybridChunker
        chunks = list(chunker.chunk(dl_doc=doc))

        # Generate output filename: original_name_chunks.jsonl
        output_filename = f"{json_file.stem}_chunks.jsonl"
        output_file = output_path_p / output_filename

        # Get current timestamp in ISO format
        timestamp = datetime.now(timezone.utc).isoformat()

        # Chunking config (for reproducibility)
        chunking_config = {
            "max_tokens": max_tokens,
            "merge_peers": merge_peers,
            "tokenizer_model": EMBED_MODEL_ID,
        }

        # Write chunks as JSONL (one JSON object per line)
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, chunk in enumerate(chunks):
                # Get contextualized text for this chunk
                chunk_text = chunker.contextualize(chunk=chunk)

                # Build the chunk object
                chunk_obj = {
                    "timestamp": timestamp,
                    "source_document": json_file.name,
                    "chunk_index": idx,
                    "chunking_config": chunking_config,
                    "text": chunk_text,
                }

                # Write as a single line of JSON
                f.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")

        print(
            f"docling-chunk: saved {len(chunks)} chunks to {output_filename}",
            flush=True,
        )
        processed_count += 1

    # Report summary
    print(
        f"docling-chunk: done - processed {processed_count}/{len(json_files)} files",
        flush=True,
    )
    if skipped_files:
        print(
            f"docling-chunk: skipped {len(skipped_files)} invalid files:",
            flush=True,
        )
        for filename, reason in skipped_files:
            print(f"  - {filename}: {reason}", flush=True)
