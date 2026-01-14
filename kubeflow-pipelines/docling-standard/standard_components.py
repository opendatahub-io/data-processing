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
def docling_convert_standard(
    input_path: dsl.Input[dsl.Artifact],
    artifacts_path: dsl.Input[dsl.Artifact],
    output_path: dsl.Output[dsl.Artifact],
    pdf_filenames: List[str],
    pdf_backend: str = "dlparse_v4",
    image_export_mode: str = "embedded",
    table_mode: str = "accurate",
    timeout_per_document: int = 300,
    ocr: bool = True,
    force_ocr: bool = False,
    ocr_engine: str = "easyocr",
    allow_external_plugins: bool = False,
    enrich_code: bool = False,
    enrich_formula: bool = False,
    enrich_picture_classes: bool = False,
    enrich_picture_description: bool = False,
):
    """
    Convert a list of PDF files to JSON and Markdown using Docling (Standard Pipeline).

    Uses docling-jobkit's DoclingConverterManager for simplified converter initialization.

    Args:
        input_path: Path to the input directory containing PDF files.
        artifacts_path: Path to the directory containing Docling models.
        output_path: Path to the output directory for converted JSON and Markdown files.
        pdf_filenames: List of PDF file names to process.
        pdf_backend: Backend to use for PDF processing.
        image_export_mode: Mode to export images.
        table_mode: Mode to detect tables.
        timeout_per_document: Timeout per document processing.
        ocr: Whether or not to use OCR if needed.
        force_ocr: Whether or not to force OCR.
        ocr_engine: Engine to use for OCR.
        allow_external_plugins: Whether or not to allow external plugins.
        enrich_code: Whether or not to enrich code.
        enrich_formula: Whether or not to enrich formula.
        enrich_picture_classes: Whether or not to enrich picture classes.
        enrich_picture_description: Whether or not to enrich picture description.
    """
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    # docling-jobkit imports
    from docling_jobkit.convert.manager import (  # pylint: disable=import-outside-toplevel
        DoclingConverterManager,
        DoclingConverterManagerConfig,
    )
    from docling_jobkit.datamodel.convert import (  # pylint: disable=import-outside-toplevel
        ConvertDocumentsOptions,
        ProcessingPipeline,
    )
    from docling.datamodel.pipeline_options import (  # pylint: disable=import-outside-toplevel
        PdfBackend,
        TableFormerMode,
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
        f"docling-standard-convert: starting with backend='{pdf_backend}', files={len(input_pdfs)}",
        flush=True,
    )

    # Create manager config with infrastructure settings
    manager_config = DoclingConverterManagerConfig(
        artifacts_path=artifacts_path_p,
        allow_external_plugins=allow_external_plugins,
    )

    # Create conversion options with all document processing settings
    convert_options = ConvertDocumentsOptions(
        pipeline=ProcessingPipeline.STANDARD,
        pdf_backend=PdfBackend(pdf_backend),
        table_mode=TableFormerMode(table_mode),
        do_ocr=ocr,
        force_ocr=force_ocr,
        ocr_engine=ocr_engine,
        document_timeout=float(timeout_per_document),
        image_export_mode=ImageRefMode(image_export_mode),
        do_code_enrichment=enrich_code,
        do_formula_enrichment=enrich_formula,
        do_picture_classification=enrich_picture_classes,
        do_picture_description=enrich_picture_description,
    )

    # Create manager and convert documents
    manager = DoclingConverterManager(manager_config)
    results = manager.convert_documents(sources=input_pdfs, options=convert_options)

    # Save outputs
    for result in results:
        doc_filename = result.input.file.stem

        output_json_path = output_path_p / f"{doc_filename}.json"
        print(f"docling-standard-convert: saving {output_json_path}", flush=True)
        result.document.save_as_json(
            output_json_path, image_mode=ImageRefMode(image_export_mode)
        )

        output_md_path = output_path_p / f"{doc_filename}.md"
        print(f"docling-standard-convert: saving {output_md_path}", flush=True)
        result.document.save_as_markdown(
            output_md_path, image_mode=ImageRefMode(image_export_mode)
        )

    print("docling-standard-convert: done", flush=True)
