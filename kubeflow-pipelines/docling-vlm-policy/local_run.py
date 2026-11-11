import os
import sys
from pathlib import Path
from typing import List

# Add parent directories to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "docling-standard"))

from common import (
    create_pdf_splits,
    docling_chunk,
    download_docling_models,
    import_pdfs,
)
from kfp import dsl, local
from policy_components import interpret_policy_artifacts
from standard_components import docling_convert_standard

VLM_ENV_KEYS = (
    "REMOTE_MODEL_ENDPOINT_URL",
    "REMOTE_MODEL_NAME",
    "REMOTE_MODEL_API_KEY",
)


@dsl.component(base_image="python:3.11")
def take_first_split(splits: List[List[str]]) -> List[str]:
    return splits[0] if splits else []


@dsl.pipeline()
def policy_convert_pipeline_local():
    """
    Local pipeline for testing policy document conversion with VLM interpretation.
    """
    importer = import_pdfs(
        filenames="2305.03393v1-pg9.pdf",
        base_url="https://github.com/docling-project/docling/raw/v2.43.0/tests/data/pdf",
    )

    pdf_splits = create_pdf_splits(
        input_path=importer.outputs["output_path"],
        num_splits=1,
    )

    artifacts = download_docling_models(
        pipeline_type="standard",
        remote_model_endpoint_enabled=False,
    )

    first_split = take_first_split(splits=pdf_splits.output)

    converter = docling_convert_standard(
        input_path=importer.outputs["output_path"],
        artifacts_path=artifacts.outputs["output_path"],
        pdf_filenames=first_split.output,
        image_export_mode="referenced",
    )

    interpreter = interpret_policy_artifacts(
        input_path=converter.outputs["output_path"],
    )

    docling_chunk(
        input_path=interpreter.outputs["output_path"],
        max_tokens=512,
        merge_peers=True,
    )


def main() -> None:
    vlm_env = {k: os.environ[k] for k in VLM_ENV_KEYS if k in os.environ}
    if not vlm_env:
        print(
            "Set REMOTE_MODEL_ENDPOINT_URL, REMOTE_MODEL_NAME, and "
            "REMOTE_MODEL_API_KEY environment variables to configure the VLM "
            "endpoint for local runs."
        )
    local.init(runner=local.DockerRunner(environment=vlm_env))
    policy_convert_pipeline_local()


if __name__ == "__main__":
    main()
