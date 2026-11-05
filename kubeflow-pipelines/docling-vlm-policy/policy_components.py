import sys
from pathlib import Path

# Add the parent directory to Python path to find common
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.constants import DOCLING_BASE_IMAGE
from kfp import dsl


@dsl.component(
    base_image=DOCLING_BASE_IMAGE,
)
def interpret_policy_artifacts(
    input_path: dsl.Input[dsl.Artifact],
    output_path: dsl.Output[dsl.Artifact],
    vlm_secret_mount_path: str = "/mnt/secrets",
    vlm_max_tokens: int = 1024,
    vlm_timeout: int = 120,
):
    """
    Classify and interpret non-text artifacts in policy documents using a VLM.

    Reads Docling conversion output (JSON, Markdown, and referenced image files),
    sends each extracted image to a VLM for artifact-type classification and
    type-specific structured interpretation, then assembles final Markdown with
    interpreted content replacing image references wrapped in artifact-type markers.

    Args:
        input_path: Path to directory containing Docling conversion output.
        output_path: Path to output directory for assembled policy Markdown,
            artifact metadata, and passthrough DoclingDocument JSON.
        vlm_secret_mount_path: Path to mounted Kubernetes secret containing
            REMOTE_MODEL_ENDPOINT_URL, REMOTE_MODEL_NAME, REMOTE_MODEL_API_KEY.
        vlm_max_tokens: Maximum tokens for each VLM interpretation response.
        vlm_timeout: Timeout in seconds per VLM API call.
    """
    import base64
    import json
    import os
    import re
    import shutil
    import time
    from pathlib import Path

    import requests

    CLASSIFICATION_PROMPT = (
        "You are analyzing an image extracted from a policy document. "
        "Classify this image into exactly one of the following categories. "
        "Respond with ONLY the category name, nothing else.\n\n"
        "Categories:\n"
        "- chart: Any type of chart, graph, or plot (bar, line, pie, scatter, etc.)\n"
        "- risk_matrix: A risk assessment matrix or heat map showing likelihood vs impact\n"
        "- decision_tree: A decision tree, flowchart showing decision logic with branches\n"
        "- form: A form, questionnaire, or template with fields to fill in\n"
        "- screenshot: A screenshot of a user interface or application\n"
        "- diagram: An architecture diagram, process flow, org chart, or network diagram\n"
        "- other: Any other type of image that doesn't fit the above categories"
    )

    INTERPRETATION_PROMPTS = {
        "risk_matrix": (
            "Convert this risk matrix into structured Markdown. "
            "Create a Markdown table that preserves all row headers, column headers, "
            "and cell values. Include severity levels and any color coding described as "
            "text labels (e.g., 'High Risk'). Add any legends, footnotes, or scoring "
            "criteria below the table."
        ),
        "decision_tree": (
            "Convert this decision tree into a nested Markdown list representing the "
            "complete decision logic. Use indentation to show hierarchy and branches. "
            "Each decision point should show the question or condition. "
            "Each branch should show the path taken (Yes/No, or named paths). "
            "Each leaf node should show the final outcome or action. "
            "Preserve all labels, conditions, thresholds, and outcomes exactly as shown."
        ),
        "chart": (
            "Describe this chart in structured Markdown. Include:\n"
            "- **Chart type**: (bar, line, pie, etc.)\n"
            "- **Title**: The chart title if visible\n"
            "- **Axes**: X-axis and Y-axis labels and units\n"
            "- **Data summary**: A Markdown table with the key data points and values\n"
            "- **Trends**: Notable trends, peaks, or patterns in the data\n"
            "- **Annotations**: Any callouts, labels, or annotations on the chart"
        ),
        "form": (
            "Convert this form into structured Markdown. "
            "Create a Markdown table with 'Field' and 'Value/Description' columns. "
            "Include all field labels and any pre-filled values. "
            "For checkboxes, indicate [x] checked or [ ] unchecked. "
            "For dropdowns, show the selected value if visible. "
            "Include section headers, instructions, and help text."
        ),
        "screenshot": (
            "Describe this UI screenshot as a structured step-by-step walkthrough. "
            "Identify the application or interface shown. "
            "Describe visible elements: navigation, buttons, menus, form fields, "
            "status indicators, data displays, and notifications. "
            "Note the current state or action being performed. "
            "If it shows a workflow or sequence, use numbered steps."
        ),
        "diagram": (
            "Describe this diagram in structured Markdown. "
            "List all components and their labels. "
            "Describe relationships and connections between components. "
            "Note directional flows (arrows) and their labels. "
            "Identify groupings, layers, or zones. "
            "Use a nested list to represent the hierarchy, or a table for "
            "components and connections. "
            "Preserve any annotations, version numbers, or metadata shown."
        ),
        "other": (
            "Describe the content of this image in structured Markdown. "
            "Preserve all visible text, numbers, labels, and structural information. "
            "Use appropriate Markdown formatting — tables for tabular data, "
            "lists for hierarchical or sequential content, and headings for sections. "
            "Capture any legends, footnotes, or annotations."
        ),
    }

    VALID_TYPES = set(INTERPRETATION_PROMPTS.keys())

    # --- Read VLM credentials from mounted secret or environment variables ---
    def read_secret(name):
        fpath = os.path.join(vlm_secret_mount_path, name)
        if os.path.isfile(fpath):
            with open(fpath) as f:
                return f.read().strip()
        env_val = os.environ.get(name)
        if env_val is not None:
            return env_val
        raise ValueError(
            f"Key {name} not found in secret mount {vlm_secret_mount_path} "
            f"or environment variable {name}"
        )

    if not os.path.exists(vlm_secret_mount_path):
        print(
            f"Secret mount path {vlm_secret_mount_path} not found, "
            "falling back to environment variables",
            flush=True,
        )

    vlm_endpoint_url = read_secret("REMOTE_MODEL_ENDPOINT_URL")
    vlm_model_name = read_secret("REMOTE_MODEL_NAME")
    vlm_api_key = read_secret("REMOTE_MODEL_API_KEY")

    if not vlm_endpoint_url.endswith("/v1/chat/completions"):
        vlm_endpoint_url = vlm_endpoint_url.rstrip("/")
        if not vlm_endpoint_url.endswith("/v1"):
            vlm_endpoint_url += "/v1"
        vlm_endpoint_url += "/chat/completions"

    input_path_p = Path(input_path.path)
    output_path_p = Path(output_path.path)
    output_path_p.mkdir(parents=True, exist_ok=True)

    # --- VLM call helper with retries ---
    def call_vlm(image_b64, prompt, max_tokens):
        payload = {
            "model": vlm_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }
        headers = {}
        if vlm_api_key:
            headers["Authorization"] = f"Bearer {vlm_api_key}"

        for attempt in range(3):
            try:
                resp = requests.post(
                    vlm_endpoint_url,
                    json=payload,
                    headers=headers,
                    timeout=vlm_timeout,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError, IndexError) as e:
                if attempt < 2:
                    wait = 2 ** (attempt + 1)
                    print(
                        f"  VLM call attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait}s...",
                        flush=True,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"VLM call failed after 3 attempts: {e}") from e

    def resolve_image_path(img_ref, doc_name):
        """Try multiple strategies to find the image file."""
        candidates = [
            input_path_p / img_ref,
            input_path_p / doc_name / img_ref,
            input_path_p / Path(img_ref).name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        found = list(input_path_p.rglob(Path(img_ref).name))
        if found:
            return found[0]
        return None

    # --- Process each document ---
    md_files = sorted(input_path_p.glob("*.md"))
    if not md_files:
        print(
            "interpret-policy-artifacts: no markdown files found, nothing to process",
            flush=True,
        )
        return

    print(
        f"interpret-policy-artifacts: processing {len(md_files)} document(s)",
        flush=True,
    )

    img_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    for md_file in md_files:
        doc_name = md_file.stem
        print(f"interpret-policy-artifacts: processing {doc_name}", flush=True)

        md_content = md_file.read_text(encoding="utf-8")
        matches = list(img_pattern.finditer(md_content))

        if not matches:
            print(f"  {doc_name}: no images found, copying markdown as-is", flush=True)
            shutil.copy2(md_file, output_path_p / f"{doc_name}_policy.md")
            json_src = input_path_p / f"{doc_name}.json"
            if json_src.exists():
                shutil.copy2(json_src, output_path_p / f"{doc_name}.json")
            continue

        print(f"  {doc_name}: found {len(matches)} image(s) to interpret", flush=True)

        artifacts_metadata = []
        assembled_md = md_content

        for idx, match in enumerate(matches):
            img_ref = match.group(2)
            full_match_text = match.group(0)

            img_path = resolve_image_path(img_ref, doc_name)
            if img_path is None:
                print(f"  {doc_name}: image {img_ref} not found, skipping", flush=True)
                continue

            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            # Step 1: Classify artifact type
            print(
                f"  {doc_name}: classifying image {idx + 1}/{len(matches)}...",
                flush=True,
            )
            raw_type = call_vlm(img_b64, CLASSIFICATION_PROMPT, max_tokens=20)
            artifact_type = raw_type.strip().lower().replace(" ", "_")
            if artifact_type not in VALID_TYPES:
                artifact_type = "other"
            print(
                f"  {doc_name}: image {idx + 1} classified as '{artifact_type}'",
                flush=True,
            )

            # Step 2: Interpret with type-specific prompt
            print(
                f"  {doc_name}: interpreting image {idx + 1} as {artifact_type}...",
                flush=True,
            )
            interpretation = call_vlm(
                img_b64,
                INTERPRETATION_PROMPTS[artifact_type],
                max_tokens=vlm_max_tokens,
            )

            # Step 3: Build replacement with artifact markers
            replacement = (
                f"<!-- artifact: {artifact_type} -->\n"
                f"{interpretation.strip()}\n"
                f"<!-- /artifact -->"
            )
            assembled_md = assembled_md.replace(full_match_text, replacement, 1)

            artifacts_metadata.append(
                {
                    "index": idx,
                    "type": artifact_type,
                    "source_image": str(img_ref),
                    "interpretation": interpretation.strip(),
                }
            )

        # Save assembled policy markdown
        policy_md_path = output_path_p / f"{doc_name}_policy.md"
        policy_md_path.write_text(assembled_md, encoding="utf-8")
        print(f"  {doc_name}: saved {policy_md_path.name}", flush=True)

        # Save artifact metadata in a subdirectory to avoid conflicts with
        # the chunker, which globs *.json and validates as DoclingDocument
        metadata_dir = output_path_p / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        artifacts_json_path = metadata_dir / f"{doc_name}_artifacts.json"
        with open(artifacts_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "document": doc_name,
                    "total_artifacts": len(artifacts_metadata),
                    "artifacts": artifacts_metadata,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"  {doc_name}: saved {artifacts_json_path.name}", flush=True)

        # Copy DoclingDocument JSON for downstream consumers and chunking
        json_src = input_path_p / f"{doc_name}.json"
        if json_src.exists():
            shutil.copy2(json_src, output_path_p / f"{doc_name}.json")

    print("interpret-policy-artifacts: done", flush=True)
