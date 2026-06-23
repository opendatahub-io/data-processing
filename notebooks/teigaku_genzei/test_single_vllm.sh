#!/bin/sh

if [ -z "$1" ]; then
    echo "./test_single_vllm.sh {model_path}"
    echo "e.g., ./test_single_vllm.sh path_to/granite-3.3-8b-instruct-teigaku-genzei-interp"
    exit 1
fi
MODEL_TEST_PATH="$1" # /proj/instructlab/checkpoints/${MODEL_TEST_NAME}
MODEL_TEST_NAME=`basename ${MODEL_TEST_PATH}`
# MODEL_TEST_NAME=granite-3.3-8b-instruct-teigaku-genzei-interp
# MODEL_TEST_NAME=granite-3.3-8b-instruct-teigaku-genzei-ibm-v2-interp

# JUDGE_MODULE="metrics.llm_as_judge.direct.rits.llama3_3_70b"
# JUDGE_MODULE="metrics.llm_as_judge.direct.rits.llama4_maverick"
# JUDGE_MODULE="metrics.llm_as_judge.direct.rits.phi_4"
# JUDGE_MODULE="metrics.llm_as_judge.direct_positional_bias.rits.llama3_3_70b"
# JUDGE_MODULE="metrics.llm_as_judge.direct_positional_bias.rits.llama4_maverick"
JUDGE_MODULE="metrics.llm_as_judge.direct_positional_bias.rits.phi_4"

# SERVER_TYPE="vllm-remote"
# SERVER_TYPE="openai"
SERVER_TYPE="vllm-remote"
SERVER_API_KEY="dummy"
# DATA_CLASSIFICATION="public"
DATA_CLASSIFICATION="proprietary"

TEST_DIR=./qa_table
TEST_NAMES="0024001-021"
OUT_DIR=./tmp/single_${JUDGE_MODULE}__${MODEL_TEST_NAME}

mkdir -p ${OUT_DIR}

for TEST in ${TEST_NAMES}; do
    TEST_FILE=${TEST_DIR}/${TEST}.csv
    OUT_RESULT_PREFIX=${OUT_DIR}/result_${TEST}
    OUT_GLOBAL_PREFIX=${OUT_DIR}/global_${TEST}

    python ./test_qa.py \
        --test_file ${TEST_FILE} \
        --out_result_file ${OUT_RESULT_PREFIX}_after.csv \
        --out_global_score_file ${OUT_GLOBAL_PREFIX}_after.csv \
        --judge_module ${JUDGE_MODULE} \
        --server_type ${SERVER_TYPE} \
        --api_key ${SERVER_API_KEY} \
        --data_classification ${DATA_CLASSIFICATION} \
        --model_path ${MODEL_TEST_PATH}
    # python ./test_qa.py \
    #     --test_file ${TEST_FILE} \
    #     --out_result_file ${OUT_RESULT_PREFIX}_before.csv \
    #     --out_global_score_file ${OUT_GLOBAL_PREFIX}_before.csv \
    #     --judge_module ${JUDGE_MODULE} \
    #     --model_path ibm-granite/granite-3.3-8b-instruct
done
