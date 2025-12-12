import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import create_pdf_splits, download_docling_models, import_pdfs
from kfp import dsl, local
from standard_components import docling_convert_standard


@dsl.component(base_image="python:3.11")
def take_first_split(splits: List[List[str]]) -> List[str]:
    return splits[0] if splits else []


@dsl.pipeline()
def convert_pipeline_local():
    importer = import_pdfs(
        filenames="2203.01017v2.pdf,2206.01062.pdf",
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

    docling_convert_standard(
        input_path=importer.outputs["output_path"],
        artifacts_path=artifacts.outputs["output_path"],
        pdf_filenames=first_split.output,
    )


def main() -> None:
    # Requires: pip install docker; and a Docker-compatible daemon (Docker or Podman socket)
    local.init(runner=local.DockerRunner())
    convert_pipeline_local()


if __name__ == "__main__":
    main()
