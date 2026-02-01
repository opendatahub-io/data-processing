# Import required components

import sys
from typing import cast, Any, Hashable, Sequence, Collection
import json
import argparse
import os

from unitxt import evaluate, create_dataset, load_dataset
from unitxt.blocks import Task, InputOutputTemplate, TaskCard
from unitxt.inference import HFAutoModelInferenceEngine, OpenAiInferenceEngine, VLLMRemoteInferenceEngine, ListWithMetadata, TextGenerationInferenceOutput
from unitxt.metric_utils import EvaluationResults
from unitxt.operators import Copy, Set
from unitxt.loaders import LoadFromDictionary, LoadCSV
from datasets import Dataset, DatasetDict
import pandas as pd
import dotenv

import catalog_util
import jsonl_util

# TODO:
# load test CSV
# prompt template config
# metrics: ROUGE
# metrics: BLEU or SacreBLEU (reference can be multiple)
# metrics: BERTScore
# metrics: relevance by llm-as-a-judge
# output global score as a file.


MAX_OUTPUT_TOKENS = 1024

COL_TITLE = "Title"
COL_QUESTION = "Question"
COL_ANSWER = "Answer"
COL_LANGUAGE = "target_language"
COL_E_CONTEXT = "Context"
COL_GROUND_TRUTH = "Ground Truth"
# COL_E_QUESTION = "question"
# COL_E_ANSWER = "answer"

ARG_MODEL_PATH="model_path"
ARG_TEST_FILE="test_file"
ARG_OUT_RESULT_FILE="out_result_file"
ARG_TEST_LANGUAGE="test_language"
ARG_OUT_GLOBAL_SCORE_FILE="out_global_score_file"
ARG_JUDGE_MODULE="judge_module"
ARG_API_URL: str = "api_url"
ARG_API_KEY: str = "api_key"
ARG_SERVER_TYPE: str = "server_type"
ARG_DATA_CLASSIFICATION: str = "data_classification"
ARG_INSTRUCTION: str = "instruction"
ARG_PREDICTION_FILE: str = "prediction_file"
ARG_PREDICTION_ONLY: str = "prediction_only"

INSTRUCTION_DEFAULT = {
    "ja": "以下の質問に答えてください。",
    "en": "Answer the following question.",
}


def str_to_bool(s):
    return s.strip().lower() in ("true", "1", "yes", "y", "on")

def config():
    parser = argparse.ArgumentParser(description='test a model with a zero-shot QA task')
    parser.add_argument('--' + ARG_MODEL_PATH, type=str, default="Qwen/Qwen1.5-0.5B-Chat")
    parser.add_argument('--' + ARG_TEST_FILE, type=str, default=None)
    parser.add_argument('--' + ARG_TEST_LANGUAGE, type=str, default="ja")
    parser.add_argument('--' + ARG_OUT_RESULT_FILE, type=str, default=None)
    parser.add_argument('--' + ARG_OUT_GLOBAL_SCORE_FILE, type=str, default=None)
    parser.add_argument('--' + ARG_JUDGE_MODULE, type=str, default="metrics.llm_as_judge.direct.rits.phi_4")
    parser.add_argument('--' + ARG_API_URL, type=str, default="http://0.0.0.0:8000/v1") 
    parser.add_argument('--' + ARG_API_KEY, type=str, default=None)
    parser.add_argument('--' + ARG_SERVER_TYPE, type=str, default="hf", choices=["hf", "openai", "vllm-remote"])
    parser.add_argument('--' + ARG_DATA_CLASSIFICATION, type=str, default="proprietary", choices=["public", "proprietary"])
    parser.add_argument('--' + ARG_INSTRUCTION, type=str, default=None)
    parser.add_argument('--' + ARG_PREDICTION_FILE, type=str, default=None)
    parser.add_argument('--' + ARG_PREDICTION_ONLY, type=str_to_bool, default=False, choices=[False, True])
    # parser.add_argument('--device', type=str, default="cuda")

    args = parser.parse_args()

    args = vars(args)
    
    return args

def load_test_data(input_file: str, test_language: str) ->  list[dict[Hashable, Any]]:
    data_df = pd.read_csv(input_file, encoding="utf8", dtype="object")
    data = data_df.to_dict("records")
    return data

def load_test_data2(input_file: str) -> list[dict[str, Any]]: # list[dict[str, str]]:
    data = [
        {"Title":"1-1", "Question": "What is the capital of Texas?", "Answer": "Austin", "target_language": "ja"},
        {"Title":"1-2", "Question": "What is the color of the sky?", "Answer": "Blue", "target_language": "ja"},
    ]
    return data

def write_json_file(file_path: str, data: Any):
    with open(file_path, 'w') as file:
        json.dump(data, file, ensure_ascii=False)
    return

def write_results_json(file_path: str, results: EvaluationResults) -> None:
    results_list = [ {"source": res["source"], "score": res["score"], "prediction": res["prediction"]} for res in results ]
    # results_list = results.instance_scores
    write_json_file(file_path, results_list)
    return

def flatten_score(score_dict: dict[Any, Any]) -> dict[Any, Any]:

    flat_score = { k:v for (k,v) in score_dict.items() if (not isinstance(v, Collection)) or (isinstance(v, str))}
    return flat_score

def convert_results_to_df(results: EvaluationResults) -> pd.DataFrame:
    flatten_results = [flatten_score(res["score"]["instance"]) | {"source": res["source"], "prediction": res["prediction"]} for res in results]
    out_df = pd.DataFrame.from_records(flatten_results).sort_index(axis=1)
    return out_df

def write_results_csv(output_file: str, results: EvaluationResults) -> None:

    # flatten_results = [flatten_score(res["score"]["instance"]) | {"source": res["source"], "prediction": res["prediction"]} for res in results]
    # out_df = pd.DataFrame.from_records(flatten_results).sort_index(axis=1)
    out_df = convert_results_to_df(results)
    # out_df = results.instance_scores.to_df()
    out_df.to_csv(output_file, encoding="utf8", index=False)
    return


def get_lm_as_a_judge_metrics_list(judge_name: str, criteria: dict[str, str]) -> list[str]:
    lmaaj_metrics = [ 
        judge_name + f"[context_fields=[{fields}],criteria=metrics.llm_as_judge.direct.criteria.{criterion}]"
        for (criterion, fields) in criteria.items()
    ]
    return lmaaj_metrics

def register_unitxt_local_catalog_items():
    # if catalog_util.is_artifact_in_catalog("metrics.llm_as_judge.direct.rits.phi_4", catalog_path=catalog_util.LOCAL_CATALOG_PATH):
    # if catalog_util.is_artifact_in_catalog("metrics.llm_as_judge.direct.rits.llama4_maverick", catalog_path=catalog_util.LOCAL_CATALOG_PATH):
    # if catalog_util.is_artifact_in_catalog("metrics.llm_as_judge.direct.criteria.correctness_based_on_ground_truth2", catalog_path=catalog_util.LOCAL_CATALOG_PATH):
    import llm_as_judge_direct_rits_phi
    llm_as_judge_direct_rits_phi.register(catalog_util.LOCAL_CATALOG_PATH)
    import llm_as_judge_direct_rits_llama4
    llm_as_judge_direct_rits_llama4.register(catalog_util.LOCAL_CATALOG_PATH)
    import llm_as_judge_direct_criteria_correctness2
    llm_as_judge_direct_criteria_correctness2.register(catalog_util.LOCAL_CATALOG_PATH)
    import llm_as_judge_direct_positional_bias
    llm_as_judge_direct_positional_bias.register(catalog_util.LOCAL_CATALOG_PATH)
    return

def do_inference(model_path: str, api_key: str, api_url: str, server_type: str, dataset: Dataset) -> ListWithMetadata[str] | ListWithMetadata[TextGenerationInferenceOutput]:
    model = OpenAiInferenceEngine(
        model_name=model_path,
        max_tokens=MAX_OUTPUT_TOKENS,
        credentials={} if api_key is None else {"api_key": api_key},
        base_url=api_url,
        # use_cache=False,
        # use_fast_tokenizer=False,
    ) if server_type == "openai" else VLLMRemoteInferenceEngine(
        model_name=model_path,
        max_tokens=MAX_OUTPUT_TOKENS,
        credentials={} if api_key is None else {"api_key": api_key},
        base_url=api_url,
        data_classification_policy=["public", "proprietary"],
        # use_cache=False,
        # use_fast_tokenizer=False,
    ) if server_type == "vllm-remote" else HFAutoModelInferenceEngine(
        model_name=model_path,
        max_new_tokens=MAX_OUTPUT_TOKENS,
        # use_cache=False,
        # use_fast_tokenizer=False,
    )
    predictions_new = model(dataset)
    return predictions_new

def main():
    # if len(sys.argv) != 3:
    #     print("Usage: python script.py <input_json_file> <output_csv_file>")
    #     sys.exit(1)
    args = config()

    input_file = args[ARG_TEST_FILE]
    output_file = args[ARG_OUT_RESULT_FILE]
    model_path = args[ARG_MODEL_PATH]
    test_language = args[ARG_TEST_LANGUAGE]
    output_global_score_file = args[ARG_OUT_GLOBAL_SCORE_FILE]
    lmaaj_judge_name = args[ARG_JUDGE_MODULE]
    api_url = args[ARG_API_URL]
    api_key = args[ARG_API_KEY]
    server_type = args[ARG_SERVER_TYPE]
    data_classification = args[ARG_DATA_CLASSIFICATION]
    instruction = args[ARG_INSTRUCTION] if args[ARG_INSTRUCTION] is not None else INSTRUCTION_DEFAULT[test_language]
    predictions_file = args[ARG_PREDICTION_FILE]
    is_prediction_only = args[ARG_PREDICTION_ONLY]

    dotenv.load_dotenv()

    register_unitxt_local_catalog_items()
    
    # Question-answer dataset
    data = load_test_data(input_file, test_language) # [:2]

    # Define the task and evaluation metric
    # lmaaj_criteria = ["answer_relevance", "coherence", "conciseness", "correctness_based_on_ground_truth"]
    lmaaj_criteria = {
        "answer_relevance": COL_QUESTION, 
        "coherence": COL_QUESTION, 
        "conciseness": COL_QUESTION, 
        "correctness_based_on_ground_truth2": ",".join([COL_QUESTION, COL_GROUND_TRUTH]), 
        # "correctness_based_on_ground_truth": ",".join([COL_QUESTION, COL_GROUND_TRUTH]),
        "reference_document_faithfulness": ",".join([COL_E_CONTEXT, COL_QUESTION]),
        "consistency": ",".join([COL_E_CONTEXT, COL_QUESTION]),
    }

    # lmaaj_judge_name = "metrics.llm_as_judge.direct.rits.llama3_3_70b"
    # lmaaj_judge_name = "metrics.llm_as_judge.direct.rits.mixtral_large"
    # lmaaj_judge_name = "metrics.llm_as_judge.direct.rits.llama4_maverick"
    lmaaj_metrics_list = get_lm_as_a_judge_metrics_list(lmaaj_judge_name, lmaaj_criteria)
    # TODO: lmaaj assumes "context" in the input field. try using the gt answers as the context (do not pass it to the template)
    # TODO: set prefix if multiple judges are used.

    task = Task(
        input_fields={COL_QUESTION: str, COL_LANGUAGE: str},
        reference_fields={COL_ANSWER: str},
        prediction_type=str,
        metrics=[
            "metrics.rouge",
            "metrics.sacrebleu",
            # "metrics.bert_score.deberta_large_mnli",
        ] + lmaaj_metrics_list,
    )

    # Create a template to format inputs and outputs
    template = InputOutputTemplate(
        instruction=instruction,
        # input_format="{Question}",
        # output_format="{Answer}",
        input_format="{" + COL_QUESTION + "}",
        output_format="{" + COL_ANSWER + "}",
        postprocessors=[
            # "processors.lower_case", 
            "processors.to_string_stripped",
            Copy(field="task_data/" + COL_ANSWER, to_field="task_data/" + COL_E_CONTEXT),
            Copy(field="task_data/" + COL_ANSWER, to_field="task_data/" + COL_GROUND_TRUTH),
            # Copy(field="task_data/" + COL_QUESTION, to_field="task_data/" + COL_E_QUESTION),
            # Copy(field="task_data/" + COL_ANSWER, to_field="task_data/" + COL_E_ANSWER),
        ],
    )

    card = TaskCard(
        loader=LoadFromDictionary(data={"test": data}, data_classification_policy=[data_classification]),
        task=task,
        templates=[template],
        preprocess_steps=[
            Set(fields={(COL_LANGUAGE): test_language}),
        ],
    )
    dataset_tmp = load_dataset(
        card=card,
        format="formats.chat_api",
        split="test",
    )
    # Prepare the dataset
    # dataset_tmp = create_dataset(
    #     task=task,
    #     template=template,
    #     format="formats.chat_api",
    #     test_set=data,
    #     split="test",
    # )
    dataset = cast(Dataset, dataset_tmp)
    # print(dataset)
    print(type(dataset))
    # print(dataset["task_data"])

    # Set up the model (supports Hugging Face, WatsonX, OpenAI, etc.)
    # model = HFAutoModelInferenceEngine(
    #     model_name="Qwen/Qwen1.5-0.5B-Chat", max_new_tokens=32
    # )
    # model = OpenAiInferenceEngine(
    #     model_name=model_path,
    #     max_tokens=MAX_OUTPUT_TOKENS,
    #     credentials={} if api_key is None else {"api_key": api_key},
    #     base_url=api_url,
    #     # use_cache=False,
    #     # use_fast_tokenizer=False,
    # ) if server_type == "openai" else VLLMRemoteInferenceEngine(
    #     model_name=model_path,
    #     max_tokens=MAX_OUTPUT_TOKENS,
    #     credentials={} if api_key is None else {"api_key": api_key},
    #     base_url=api_url,
    #     data_classification_policy=["public", "proprietary"],
    #     # use_cache=False,
    #     # use_fast_tokenizer=False,
    # ) if server_type == "vllm-remote" else HFAutoModelInferenceEngine(
    #     model_name=model_path,
    #     max_new_tokens=MAX_OUTPUT_TOKENS,
    #     # use_cache=False,
    #     # use_fast_tokenizer=False,
    # )

    # Generate predictions and evaluate
    predictions_loaded = jsonl_util.read_jsonl_file(predictions_file) if predictions_file is not None and os.path.exists(predictions_file) else None
    if predictions_loaded is not None:
        print("Using predictions from file ", predictions_file)

    # predictions_new = model(dataset) if predictions_loaded is None else None
    predictions_new = do_inference(model_path, api_key, api_url, server_type, dataset) if predictions_loaded is None else None
    
    if (predictions_new is not None) and (predictions_file is not None): 
        jsonl_util.write_jsonl_file(predictions_file, predictions_new)
        print("Wrote predictions to", predictions_file)
    if is_prediction_only:
        sys.exit(0)

    predictions = predictions_new if predictions_new is not None else predictions_loaded
    # print(predictions)
    results = evaluate(predictions=predictions, data=dataset)

    # Print results
    print("Global Results:\n", results.global_scores.summary)
    # print("Instance Results:\n", results.instance_scores.summary)

    # TODO:
    # Create results_df first. include bias score column for each criteria.
    # - How to get the mapping from the choice to the score?
    # -> {criteria_name}_criteria column in the result contains a JSON string with a field "option_map".
    # -> All the samples have the same value in this column.
    # Compute biased score
    # Compute biased score average
    # Include those into the result and the global result
    if output_global_score_file is not None:
        out_global_df = results.global_scores.to_df().drop(index=["score"]).sort_index()
        out_global_df.to_csv(output_global_score_file, encoding="utf8")

    write_results_csv(output_file, results)
    # write_results_json(output_file, results)
    return

if __name__ == "__main__":
    main()
