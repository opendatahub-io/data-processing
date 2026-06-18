import sys
from pathlib import Path
from typing import List

# Add the parent directory to Python path to find common
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.constants import DOCLING_BASE_IMAGE
from kfp import dsl


@dsl.component(
    base_image=DOCLING_BASE_IMAGE,
    packages_to_install=["docling-jobkit==1.6.0"],
)
def docling_convert_vlm(
    input_path: dsl.Input[dsl.Artifact],
    artifacts_path: dsl.Input[dsl.Artifact],
    output_path: dsl.Output[dsl.Artifact],
    pdf_filenames: List[str],
    image_export_mode: str = "embedded",
    timeout_per_document: int = 300,
    remote_model_enabled: bool = False,
    remote_model_secret_mount_path: str = "/mnt/secrets",
):
    """
    Convert a list of PDF files to JSON and Markdown using Docling (VLM Pipeline).

    Uses docling-jobkit's DoclingConverterManager for simplified converter initialization.

    Args:
        input_path: Path to the input directory containing PDF files.
        artifacts_path: Path to the directory containing Docling models.
        output_path: Path to the output directory for converted JSON and Markdown files.
        pdf_filenames: List of PDF file names to process.
        timeout_per_document: Timeout per document processing.
        image_export_mode: Mode to export images.
        remote_model_enabled: Whether or not to use a remote model.
        remote_model_secret_mount_path: Path to the remote model secret mount path.
    """
    import os  # pylint: disable=import-outside-toplevel
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    # docling-jobkit imports
    from docling_jobkit.convert.manager import (  # pylint: disable=import-outside-toplevel
        DoclingConverterManager,
        DoclingConverterManagerConfig,
    )
    from docling_jobkit.datamodel.convert import (  # pylint: disable=import-outside-toplevel
        ConvertDocumentsOptions,
        ProcessingPipeline,
        VlmModelApi,
    )
    from docling.datamodel.pipeline_options_vlm_model import (  # pylint: disable=import-outside-toplevel
        ResponseFormat,
    )
    from docling.datamodel.vlm_model_specs import (  # pylint: disable=import-outside-toplevel
        VlmModelType,
    )
    from docling_core.types.doc.base import (  # pylint: disable=import-outside-toplevel
        ImageRefMode,
    )

    if not pdf_filenames:
        raise ValueError(
            "pdf_filenames must be provided with the list of file names to process"
        )

    input_path_p = Path(input_path.path)
    artifacts_path_p = Path(artifacts_path.path)
    output_path_p = Path(output_path.path)
    output_path_p.mkdir(parents=True, exist_ok=True)

    input_pdfs = [input_path_p / name for name in pdf_filenames]
    print(
        f"docling-vlm-convert: starting with backend='vlm', files={len(input_pdfs)}",
        flush=True,
    )

    # Handle remote model configuration
    vlm_model_api = None
    if remote_model_enabled:
        if not os.path.exists(remote_model_secret_mount_path):
            raise ValueError(
                f"Secret for remote model should be mounted in {remote_model_secret_mount_path}"
            )

        def read_secret(secret_name: str) -> str:
            path = os.path.join(remote_model_secret_mount_path, secret_name)
            if os.path.isfile(path):
                with open(path) as f:
                    return f.read()
            raise ValueError(
                f"Key {secret_name} not defined in secret {remote_model_secret_mount_path}"
            )

        remote_endpoint_url = read_secret("REMOTE_MODEL_ENDPOINT_URL")
        remote_model_name = read_secret("REMOTE_MODEL_NAME")
        remote_api_key = read_secret("REMOTE_MODEL_API_KEY")

        if not remote_endpoint_url:
            raise ValueError(
                "remote_model_endpoint_url must be provided when remote_model_enabled is True"
            )

        # Configure VLM API model using docling-jobkit's VlmModelApi
        vlm_model_api = VlmModelApi(
            url=remote_endpoint_url,
            headers={"Authorization": f"Bearer {remote_api_key}"},
            params={"model_id": remote_model_name, "max_new_tokens": 400},
            prompt="OCR the full page to markdown.",
            timeout=600,
            response_format=ResponseFormat.MARKDOWN,
        )

    # Create manager config
    manager_config = DoclingConverterManagerConfig(
        artifacts_path=artifacts_path_p,
        enable_remote_services=remote_model_enabled,
    )

    # Create conversion options
    # For local inference, use SMOLDOCLING model; for remote, use the API model
    convert_options = ConvertDocumentsOptions(
        pipeline=ProcessingPipeline.VLM,
        document_timeout=float(timeout_per_document),
        image_export_mode=ImageRefMode(image_export_mode),
        vlm_pipeline_model=None if remote_model_enabled else VlmModelType.SMOLDOCLING,
        vlm_pipeline_model_api=vlm_model_api,
    )

    # Create manager and convert documents
    manager = DoclingConverterManager(manager_config)
    results = manager.convert_documents(sources=input_pdfs, options=convert_options)

    # Save outputs
    for result in results:
        doc_filename = result.input.file.stem

        output_json_path = output_path_p / f"{doc_filename}.json"
        print(f"docling-vlm-convert: saving {output_json_path}", flush=True)
        result.document.save_as_json(
            output_json_path, image_mode=ImageRefMode(image_export_mode)
        )

        output_md_path = output_path_p / f"{doc_filename}.md"
        print(f"docling-vlm-convert: saving {output_md_path}", flush=True)
        result.document.save_as_markdown(
            output_md_path, image_mode=ImageRefMode(image_export_mode)
        )

    print("docling-vlm-convert: done", flush=True)
