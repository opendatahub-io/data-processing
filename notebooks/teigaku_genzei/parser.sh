#!/bin/sh
set -e

source="source/"
output="docparser/"
# code_path="../sdg_hub/scripts/docparser_v2.py"
code_path="../sdg_hub/examples/knowledge_tuning/instructlab/docparser_v2.py"

# Validate inputs
if [ ! -f "${code_path}" ]; then
    echo "Error: Python script not found at ${code_path}" >&2
    exit 1
fi

python ${code_path}  --input-dir ${source} --output-dir ${output}
