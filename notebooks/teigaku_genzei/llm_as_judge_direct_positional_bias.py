from typing import List, Dict, Any
from unitxt.llm_as_judge import LLMJudgeDirect, CriteriaWithOptions
from unitxt import add_to_catalog
from unitxt.inference import CrossProviderInferenceEngine, RITSInferenceEngine

from llm_as_judge_direct_rits_llama4 import RITSInferenceEngineFixed
import llm_as_judge_direct_rits_llama4

class LLMJudgeDirectPositionalBias(LLMJudgeDirect):
    # def prepare(self):
    #     super().prepare()
    #     print(f"LLMJudgeDirectPositionalBias.prepare() criteria: {self.criteria}")
    #     print(f"LLMJudgeDirectPositionalBias.prepare() inference_engine: {self.inference_engine}")
    #     if self.criteria is not None:
    #         self.ci_scores = [
    #             self.criteria.name,
    #             self.criteria.name + "_positional_bias_score",
    #             self.criteria.name + "_positional_bias_average",
    #         ]
    #         print(f"LLMJudgeDirectPositionalBias.prepare() ci_scores: {self.ci_scores}")

    POSTFIX_SCORE = "_positional_bias_score"
    POSTFIX_AVERAGE = "_positional_bias_average"
    POSTFIX_SELECTED_OPTION = "_positional_bias_selected_option"

    def compute(
        self,
        references: List[List[str]],
        predictions: List[str],
        task_data: List[Dict[str, Any]],
    ) -> List[Dict]:

        results = super().compute(references, predictions, task_data)

        score_name = self.criteria.name if self.criteria is not None else self.main_score
        forward_score_name = score_name
        backward_score_name = score_name + LLMJudgeDirectPositionalBias.POSTFIX_SCORE
        average_score_name = score_name + LLMJudgeDirectPositionalBias.POSTFIX_AVERAGE
        backward_selected_option_name = score_name + LLMJudgeDirectPositionalBias.POSTFIX_SELECTED_OPTION

        if self.criteria is not None:
            self.ci_scores = [
                forward_score_name,
                backward_score_name,
                average_score_name,
            ]
            print(f"LLMJudgeDirectPositionalBias.prepare() ci_scores: {self.ci_scores}")
            self.reduction_map = {"mean": [self.main_score] + self.ci_scores}

        def get_pb_score(
                result: Dict[str, str], 
                criteria: CriteriaWithOptions,
                fallback_score: float,
                default_pb_score: float,
                ) -> float:
            """
            Calculate the PB score based on the evaluation results.

            PB (positional bias) is the difference of the selection of an option between forward- and backward-ordered options.
            To mitigate PB, we compute the average of the scores associated with the selected options from forward- and backward-ordered options.

            This function takes evaluation results, criteria with options, a fallback score, and an out-of-range score as input and returns the calculated PB score.

            :param result: A dictionary containing the evaluation results.
            :param criteria: A CriteriaWithOptions object holding the criteria.
            :param fallback_score: The fallback score to use if criteria are not present or backward selected option is not present.
                Note that this is the case where no LLM Judge evaluation with backward options is available.
            :param default_pb_score: The score to use if the selected option is not found in criteria.option_map.
                Note that this is an erroneous case; this score is used only when the judge's inference result does not contain a meaningful option value.
            :return: The calculated PB score.
            """
            option_map = criteria.option_map if (criteria is not None) and (criteria.option_map is not None) else None
            if option_map is None:
                return fallback_score
            selected_option = result.get(backward_selected_option_name)
            score = option_map.get(selected_option, default_pb_score) if selected_option is not None else fallback_score
            return score

        default_pb_score = 0.0 # This value is intentionally set to 0 to indicate that the judge's inference result does not contain a meaningful option value.
        pb_scores = [
            get_pb_score(result, self.criteria, result.get(self.main_score, default_pb_score), default_pb_score) for result in results
        ]
        pb_results = [
            result | {
                backward_score_name: pb_score,
                average_score_name: (result.get(self.main_score, default_pb_score) + pb_score) / 2,
            }
            for result, pb_score in zip(results, pb_scores, strict=True)
        ]
        return pb_results
    


def register(catalog_path: str) -> None:

    laaj_direct_rits_phi4 = LLMJudgeDirectPositionalBias(
        inference_engine=RITSInferenceEngine(
            max_tokens=1024,
            seed=42,
            temperature=0,
            model_name="microsoft/phi-4",
        ),
        evaluator_name="MICROSOFT_PHI_4",
        generate_summaries=False,
    )
    add_to_catalog(laaj_direct_rits_phi4, name="metrics.llm_as_judge.direct_positional_bias.rits.phi_4", overwrite=True, catalog_path=catalog_path)

    laaj_direct_rits_llama4_maverick = LLMJudgeDirectPositionalBias(
        inference_engine=RITSInferenceEngineFixed(
            max_tokens=1024,
            seed=42,
            temperature=0,
            model_name=llm_as_judge_direct_rits_llama4.MODEL_NAME,
            base_url=RITSInferenceEngineFixed.get_base_url_from_model_name_v(llm_as_judge_direct_rits_llama4.ENDPOINT_NAME, "/v1"),
            # base_url=RITSInferenceEngine.get_base_url_from_model_name(llm_as_judge_direct_rits_llama4.ENDPOINT_NAME) + "/v1",
        ),
        evaluator_name="LLAMA3_4_MAVERICK",
        generate_summaries=False,
    )
    add_to_catalog(laaj_direct_rits_llama4_maverick, name="metrics.llm_as_judge.direct_positional_bias.rits.llama4_maverick", overwrite=True, catalog_path=catalog_path)

