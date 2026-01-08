import json
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import (
    create_pdf_splits,
    docling_chunk,
    download_docling_models,
    import_pdfs,
)
from kfp import dsl, local
from vlm_components import docling_convert_vlm

# =============================
# Test Configurations
# =============================
TEST_CONFIGS = {
    "baseline_cpu": {
        "docling_accelerator_device": "cpu",
        "docling_num_threads": 4,
        "docling_image_export_mode": "embedded",
    },
    "auto_detect": {
        "docling_accelerator_device": "auto",
        "docling_num_threads": 4,
        "docling_image_export_mode": "embedded",
    },
    "minimal_threads": {
        "docling_accelerator_device": "cpu",
        "docling_num_threads": 2,
        "docling_image_export_mode": "embedded",
    },
    "placeholder_images": {
        "docling_accelerator_device": "cpu",
        "docling_num_threads": 4,
        "docling_image_export_mode": "placeholder",
    },
    # Uncomment if you have GPU available
    "gpu_test": {
        "docling_accelerator_device": "gpu",
        "docling_num_threads": 4,
        "docling_image_export_mode": "embedded",
    },
}

FAILURE_SCENARIOS = {
    "invalid_device": {
        "docling_accelerator_device": "invalid_device",
        "docling_num_threads": 4,
        "docling_image_export_mode": "embedded",
        "should_fail": True,
        "expected_error": "Invalid accelerator_device",
    },
    "invalid_image_mode": {
        "docling_accelerator_device": "cpu",
        "docling_num_threads": 4,
        "docling_image_export_mode": "invalid_mode",
        "should_fail": True,
        "expected_error": "Invalid image_export_mode",
    },
}


# ================================
# Helper Components
# ================================
@dsl.component(base_image="python:3.11")
def take_first_split(splits: List[List[str]]) -> List[str]:
    """Extract the first split from the list of PDF splits."""
    return splits[0] if splits else []


# =================================
# Test Pipeline
# =================================
@dsl.pipeline()
def convert_pipeline_test(
    docling_accelerator_device: str = "auto",
    docling_num_threads: int = 4,
    docling_image_export_mode: str = "embedded",
    docling_timeout_per_document: int = 300,
):
    importer = import_pdfs(
        filenames="2305.03393v1-pg9.pdf",
        base_url="https://github.com/docling-project/docling/raw/v2.43.0/tests/data/pdf",
    )

    pdf_splits = create_pdf_splits(
        input_path=importer.outputs["output_path"],
        num_splits=1,
    )

    artifacts = download_docling_models(
        pipeline_type="vlm",
        remote_model_endpoint_enabled=False,
    )

    first_split = take_first_split(splits=pdf_splits.output)

    converter = docling_convert_vlm(
        input_path=importer.outputs["output_path"],
        artifacts_path=artifacts.outputs["output_path"],
        pdf_filenames=first_split.output,
        num_threads=docling_num_threads,
        image_export_mode=docling_image_export_mode,
        timeout_per_document=docling_timeout_per_document,
        accelerator_device=docling_accelerator_device,
    )

    docling_chunk(
        input_path=converter.outputs["output_path"],
        max_tokens=512,
        merge_peers=True,
    )


# =============================
# Output Validation
# =============================


def validate_output(output_dir: Path, test_name: str) -> dict:
    """
    Validate the output quality and format.

    Args:
        output_dir: Path to the output directory containing converted files.
        test_name: Name of the test for reporting.

    Returns:
        Dictionary containing validation results and checks.
    """
    validation_results = {
        "test_name": test_name,
        "passed": True,
        "checks": {},
    }

    # Check for JSON output
    json_files = list(output_dir.glob("*.json"))
    validation_results["checks"]["json_exists"] = len(json_files) > 0

    # Check for Markdown output
    md_files = list(output_dir.glob("*.md"))
    validation_results["checks"]["md_exists"] = len(md_files) > 0

    if json_files:
        json_path = json_files[0]
        try:
            with open(json_path) as f:
                data = json.load(f)

            # Required fields validation
            validation_results["checks"]["has_name"] = "name" in data
            validation_results["checks"]["has_pages"] = "pages" in data

            if "pages" in data:
                validation_results["checks"]["pages_count"] = len(data["pages"])
                validation_results["checks"]["has_content"] = len(data["pages"]) > 0

                # Check page structure
                if data["pages"]:
                    first_page = data["pages"][0]
                    validation_results["checks"]["page_has_number"] = (
                        "page_number" in first_page
                    )
                    validation_results["checks"]["page_has_size"] = "size" in first_page
        except Exception as e:
            validation_results["checks"]["json_parsing_error"] = str(e)
            validation_results["passed"] = False

    if md_files:
        md_path = md_files[0]
        try:
            content = md_path.read_text()
            validation_results["checks"]["md_length"] = len(content)
            validation_results["checks"]["md_not_empty"] = len(content) > 100
            validation_results["checks"]["md_has_headers"] = "#" in content
            validation_results["checks"]["md_no_error"] = not content.startswith(
                "Error"
            )
        except Exception as e:
            validation_results["checks"]["md_parsing_error"] = str(e)
            validation_results["passed"] = False

    # Overall pass/fail
    validation_results["passed"] = all(
        [
            validation_results["checks"].get("json_exists", False),
            validation_results["checks"].get("md_exists", False),
            validation_results["checks"].get("has_content", False),
        ]
    )

    return validation_results


# =============================
# Test Runner
# =============================


def run_test_scenario(test_name: str, config: dict, should_fail: bool = False) -> dict:
    """
    Run a single test scenario.

    Args:
        test_name: Name of the test scenario.
        config: Configuration dictionary for the test.
        should_fail: Whether the test is expected to fail.

    Returns:
        Dictionary containing test results.
    """
    print(f"\n{'=' * 60}")
    print(f"Running test: {test_name}")
    print(f"Config: {config}")
    print(f"Expected to fail: {should_fail}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # Filter out test metadata keys that shouldn't be passed to the pipeline
    pipeline_config = {
        k: v for k, v in config.items() if k not in ["expected_error", "should_fail"]
    }

    try:
        convert_pipeline_test(**pipeline_config)

        elapsed = time.time() - start_time

        if should_fail:
            print(f"❌ TEST FAILED: {test_name} - Expected failure but succeeded")
            return {
                "test_name": test_name,
                "status": "FAIL",
                "reason": "Expected to fail but succeeded",
                "elapsed_time": elapsed,
            }

        print(f"✅ TEST PASSED: {test_name} - Completed in {elapsed:.2f}s")
        return {
            "test_name": test_name,
            "status": "PASS",
            "elapsed_time": elapsed,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)

        if should_fail:
            expected_error = config.get("expected_error", "")
            # Check if pipeline failed as expected (validation worked)
            if "FAILURE" in error_msg or expected_error.lower() in error_msg.lower():
                print(f"✅ TEST PASSED: {test_name} - Failed as expected")
                print("   Pipeline correctly rejected invalid parameter")
                if expected_error:
                    print(f"   Expected error type: '{expected_error}'")
                return {
                    "test_name": test_name,
                    "status": "PASS",
                    "reason": "Failed as expected - validation working",
                    "error": error_msg[:200],
                    "elapsed_time": elapsed,
                }
            else:
                print(f"❌ TEST FAILED: {test_name} - Wrong error type")
                print(f"   Expected error containing: '{expected_error}'")
                print(f"   Got: {error_msg[:200]}")
                return {
                    "test_name": test_name,
                    "status": "FAIL",
                    "reason": "Wrong error type",
                    "error": error_msg,
                    "elapsed_time": elapsed,
                }

        print(f"❌ TEST FAILED: {test_name} - Unexpected error: {error_msg[:200]}")
        return {
            "test_name": test_name,
            "status": "FAIL",
            "error": error_msg,
            "elapsed_time": elapsed,
        }


# =============================
# Main Entry Point
# =============================


def main() -> None:
    """Main test runner entry point."""
    print("=" * 60)
    print("Starting VLM Pipeline Testing Suite")
    print(f"Total tests to run: {len(TEST_CONFIGS) + len(FAILURE_SCENARIOS)}")
    print("=" * 60)

    # Initialize Docker runner for all local pipeline executions
    local.init(runner=local.DockerRunner())

    results = []

    # Run normal test scenarios
    print("\n" + "=" * 60)
    print("PHASE 1: Normal Functionality Tests")
    print("=" * 60)
    for test_name, config in TEST_CONFIGS.items():
        result = run_test_scenario(test_name, config, should_fail=False)
        results.append(result)

    # Run failure scenarios
    print("\n" + "=" * 60)
    print("PHASE 2: Failure Scenario Tests")
    print("=" * 60)
    for test_name, config in FAILURE_SCENARIOS.items():
        should_fail = config.get("should_fail", True)
        result = run_test_scenario(test_name, config, should_fail=should_fail)
        results.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    total_time = sum(r.get("elapsed_time", 0) for r in results)

    print(f"\nTotal Tests: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total Time: {total_time:.2f}s")

    print("\n📋 Detailed Results:")
    for result in results:
        status_emoji = "✅" if result["status"] == "PASS" else "❌"

        print(
            f"{status_emoji} {result['test_name']}: {result['status']} "
            f"({result.get('elapsed_time', 0):.2f}s)"
        )
        if "error" in result:
            error_preview = (
                result["error"][:150] + "..."
                if len(result["error"]) > 150
                else result["error"]
            )
            print(f"   Error: {error_preview}")
        if "reason" in result:
            print(f"   Reason: {result['reason']}")

    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
