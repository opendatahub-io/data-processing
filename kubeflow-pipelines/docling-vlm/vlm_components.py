import sys
from pathlib import Path
from typing import List

# Add the parent directory to Python path to find common
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.constants import DOCLING_BASE_IMAGE
from kfp import dsl


@dsl.component(
    base_image=DOCLING_BASE_IMAGE,
)
def docling_convert_vlm(
    input_path: dsl.Input[dsl.Artifact],
    artifacts_path: dsl.Input[dsl.Artifact],
    output_path: dsl.Output[dsl.Artifact],
    pdf_filenames: List[str],
    num_threads: int = 4,
    image_export_mode: str = "embedded",
    timeout_per_document: int = 300,
    remote_model_enabled: bool = False,
    remote_model_secret_mount_path: str = "/mnt/secrets",
    accelerator_device: str = "auto",
):
    """
    Convert a list of PDF files to JSON and Markdown using Docling (VLM Pipeline).

    Args:
        input_path: Path to the input directory containing PDF files.
        artifacts_path: Path to the directory containing Docling models.
        output_path: Path to the output directory for converted JSON and Markdown files.
        pdf_filenames: List of PDF file names to process.
        num_threads: Number of threads to use per document processing.
        timeout_per_document: Timeout per document processing.
        image_export_mode: Mode to export images.
        remote_model_enabled: Whether or not to use a remote model.
        remote_model_secret_mount_path: Path to the remote model secret mount path.
    """
    import os
    from pathlib import Path

    from docling.datamodel.accelerator_options import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415, E402
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.base_models import (
        InputFormat,
    )  # pylint: disable=import-outside-toplevel  # noqa: PLC0415, E402
    from docling.datamodel.pipeline_options import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415, E402
        VlmPipelineOptions,
        smoldocling_vlm_conversion_options,
    )
    from docling.datamodel.pipeline_options_vlm_model import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415, E402
        ApiVlmOptions,
        ResponseFormat,
    )
    from docling.document_converter import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415, E402
        DocumentConverter,
        PdfFormatOption,
    )
    from docling.pipeline.vlm_pipeline import (
        VlmPipeline,
    )  # pylint: disable=import-outside-toplevel  # noqa: PLC0415, E402
    from docling_core.types.doc.base import (
        ImageRefMode,
    )  # pylint: disable=import-outside-toplevel  # noqa: PLC0415, E402

    input_path_p = Path(input_path.path)
    artifacts_path_p = Path(artifacts_path.path)
    output_path_p = Path(output_path.path)
    output_path_p.mkdir(parents=True, exist_ok=True)

    device_map = {
        "auto": AcceleratorDevice.AUTO,
        "cpu": AcceleratorDevice.CPU,
        "gpu": AcceleratorDevice.CUDA,
    }
    dev = accelerator_device.lower()
    if dev not in device_map:
        raise ValueError(
            f"Invalid accelerator_device: {accelerator_device}. Must be one of {sorted(device_map.keys())}"
        )

    input_pdfs = [input_path_p / name for name in pdf_filenames]

    print(
        f"docling-vlm-convert: starting with backend='vlm', files={len(input_pdfs)}",
        flush=True,
    )
    if not pdf_filenames:
        raise ValueError(
            "pdf_filenames must be provided with the list of file names to process"
        )

    allowed_image_export_modes = {e.value for e in ImageRefMode}
    if image_export_mode not in allowed_image_export_modes:
        raise ValueError(
            f"Invalid image_export_mode: {image_export_mode}. Must be one of {sorted(allowed_image_export_modes)}"
        )

    if remote_model_enabled:
        if not os.path.exists(remote_model_secret_mount_path):
            raise ValueError(
                f"Secret for remote model should be mounted in {remote_model_secret_mount_path}"
            )

        remote_model_endpoint_url_secret = "REMOTE_MODEL_ENDPOINT_URL"
        remote_model_endpoint_url_file_path = os.path.join(
            remote_model_secret_mount_path, remote_model_endpoint_url_secret
        )
        if os.path.isfile(remote_model_endpoint_url_file_path):
            with open(remote_model_endpoint_url_file_path) as f:
                remote_model_endpoint_url = f.read()
        else:
            raise ValueError(
                f"Key {remote_model_endpoint_url_secret} not defined in secret {remote_model_secret_mount_path}"
            )

        remote_model_name_secret = "REMOTE_MODEL_NAME"
        remote_model_name_file_path = os.path.join(
            remote_model_secret_mount_path, remote_model_name_secret
        )
        if os.path.isfile(remote_model_name_file_path):
            with open(remote_model_name_file_path) as f:
                remote_model_name = f.read()
        else:
            raise ValueError(
                f"Key {remote_model_name_secret} not defined in secret {remote_model_secret_mount_path}"
            )

        remote_model_api_key_secret = "REMOTE_MODEL_API_KEY"
        remote_model_api_key_file_path = os.path.join(
            remote_model_secret_mount_path, remote_model_api_key_secret
        )
        if os.path.isfile(remote_model_api_key_file_path):
            with open(remote_model_api_key_file_path) as f:
                remote_model_api_key = f.read()
        else:
            raise ValueError(
                f"Key {remote_model_api_key_secret} not defined in secret {remote_model_secret_mount_path}"
            )

        if not remote_model_endpoint_url:
            raise ValueError(
                "remote_model_endpoint_url must be provided when remote_model_enabled is True"
            )

        pipeline_options = VlmPipelineOptions(
            enable_remote_services=True,
        )
        pipeline_options.vlm_options = ApiVlmOptions(
            url=remote_model_endpoint_url,  # type: ignore[arg-type]
            params=dict(
                model_id=remote_model_name,
                parameters=dict(
                    max_new_tokens=400,
                ),
            ),
            prompt="OCR the full page to markdown.",
            timeout=600,
            response_format=ResponseFormat.MARKDOWN,
            headers={
                "Authorization": f"Bearer {remote_model_api_key}",
            },
        )
    else:
        pipeline_options = VlmPipelineOptions(
            vlm_options=smoldocling_vlm_conversion_options
        )

    pipeline_cls = VlmPipeline
    pipeline_options.artifacts_path = artifacts_path_p
    pipeline_options.document_timeout = float(timeout_per_document)
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=num_threads, device=device_map[dev]
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=pipeline_cls,
                pipeline_options=pipeline_options,
            )
        }
    )

    results = doc_converter.convert_all(input_pdfs, raises_on_error=True)

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
