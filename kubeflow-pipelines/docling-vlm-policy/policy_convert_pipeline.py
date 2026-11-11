import sys
from pathlib import Path

# Add parent directories to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "docling-standard"))

from common import (
    create_pdf_splits,
    docling_chunk,
    download_docling_models,
    import_pdfs,
)
from kfp import compiler, dsl
from policy_components import interpret_policy_artifacts
from standard_components import docling_convert_standard


@dsl.pipeline(
    name="data-processing-docling-vlm-policy-pipeline",
    description="Policy document conversion pipeline with artifact-type-specific VLM interpretation by the Data Processing Team",
)
def policy_convert_pipeline(
    num_splits: int = 3,
    pdf_from_s3: bool = False,
    pdf_filenames: str = "2203.01017v2.pdf",
    # URL source params
    pdf_base_url: str = "https://github.com/docling-project/docling/raw/v2.43.0/tests/data/pdf",
    # Docling conversion params
    docling_num_threads: int = 4,
    docling_timeout_per_document: int = 300,
    docling_ocr: bool = True,
    docling_table_mode: str = "accurate",
    # VLM interpretation params
    vlm_max_tokens: int = 1024,
    vlm_timeout: int = 120,
    # Chunking params
    docling_chunk_enabled: bool = False,
    docling_chunk_max_tokens: int = 512,
    docling_chunk_merge_peers: bool = True,
):
    from kfp import kubernetes  # pylint: disable=import-outside-toplevel

    secret_mount_path = "/mnt/secrets"

    # Step 1: Import PDFs from URL or S3
    importer = import_pdfs(
        filenames=pdf_filenames,
        base_url=pdf_base_url,
        from_s3=pdf_from_s3,
        s3_secret_mount_path=secret_mount_path,
    )
    importer.set_caching_options(False)
    kubernetes.use_secret_as_volume(
        importer,
        secret_name="data-processing-docling-pipeline",
        mount_path=secret_mount_path,
        optional=True,
    )

    # Step 2: Split PDFs for parallel processing
    pdf_splits = create_pdf_splits(
        input_path=importer.outputs["output_path"],
        num_splits=num_splits,
    )

    # Step 3: Download Docling models (standard pipeline for structural parsing)
    artifacts = download_docling_models(
        pipeline_type="standard",
        remote_model_endpoint_enabled=False,
    )
    artifacts.set_caching_options(False)

    with dsl.ParallelFor(pdf_splits.output) as pdf_split:
        # Step 4: Convert PDFs with standard pipeline, exporting images as
        # referenced PNGs so the VLM interpretation step can access them
        converter = docling_convert_standard(
            input_path=importer.outputs["output_path"],
            artifacts_path=artifacts.outputs["output_path"],
            pdf_filenames=pdf_split,
            image_export_mode="referenced",
            num_threads=docling_num_threads,
            timeout_per_document=docling_timeout_per_document,
            ocr=docling_ocr,
            table_mode=docling_table_mode,
        )
        converter.set_caching_options(False)
        converter.set_memory_request("1G")
        converter.set_memory_limit("6G")
        converter.set_cpu_request("500m")
        converter.set_cpu_limit("4")

        # Step 5: Classify and interpret artifacts with VLM, assemble
        # policy-optimized Markdown with artifact-type markers
        interpreter = interpret_policy_artifacts(
            input_path=converter.outputs["output_path"],
            vlm_secret_mount_path=secret_mount_path,
            vlm_max_tokens=vlm_max_tokens,
            vlm_timeout=vlm_timeout,
        )
        interpreter.set_caching_options(False)
        interpreter.set_memory_request("512M")
        interpreter.set_memory_limit("4G")
        interpreter.set_cpu_request("250m")
        interpreter.set_cpu_limit("2")
        kubernetes.use_secret_as_volume(
            interpreter,
            secret_name="data-processing-docling-pipeline",
            mount_path=secret_mount_path,
        )

        # Step 6: Optionally chunk the DoclingDocument JSON output
        with dsl.If(docling_chunk_enabled == True):  # noqa: E712
            chunker = docling_chunk(
                input_path=interpreter.outputs["output_path"],
                max_tokens=docling_chunk_max_tokens,
                merge_peers=docling_chunk_merge_peers,
            )
            chunker.set_caching_options(False)
            chunker.set_memory_request("512M")
            chunker.set_memory_limit("2G")
            chunker.set_cpu_request("250m")
            chunker.set_cpu_limit("2")


if __name__ == "__main__":
    output_yaml = "policy_convert_pipeline_compiled.yaml"
    compiler.Compiler().compile(policy_convert_pipeline, output_yaml)
    print(f"Docling VLM policy pipeline compiled to {output_yaml}")
