import logging

from unitxt.inference import CrossProviderInferenceEngine, RITSInferenceEngine
from unitxt.llm_as_judge import LLMJudgeDirect
from unitxt import add_to_catalog
from unitxt import get_logger

logger = get_logger()

# TODO:
# This is a replacement of metrics.llm_as_judge.direct.rits.llama4_maverick in Unitxt 1.24.0
# to avoid a bug that causes a connection failure from CrossProviderInferenceEngine. See Issue #13.

MODEL_NAME="meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
ENDPOINT_NAME="llama-4-mvk-17b-128e-fp8"

class RITSInferenceEngineFixed(RITSInferenceEngine):

    @staticmethod
    def get_base_url_from_model_name_v(model_name: str, version: str):
        return RITSInferenceEngine.get_base_url_from_model_name(model_name) + version
    
    def prepare_engine(self):
        # inference endpoint need the '/v1' path
        self.base_url = (
            RITSInferenceEngineFixed.get_base_url_from_model_name_v(self.model_name, "/v1")
            # RITSInferenceEngine.get_base_url_from_model_name(self.model_name) + "/v1"
        ) if self.base_url is None else self.base_url
        logger.info(f"Created RITS inference engine (fixed) with base url: {self.base_url}")
        super(RITSInferenceEngine, self).prepare_engine()
    

def register(catalog_path: str) -> None:


    laaj_direct_rits_phi4 = LLMJudgeDirect(
        inference_engine=RITSInferenceEngineFixed(
            max_tokens=1024,
            seed=42,
            temperature=0,
            model_name=MODEL_NAME,
            # base_url=RITSInferenceEngine.get_base_url_from_model_name(ENDPOINT_NAME) + "/v1",
            base_url=RITSInferenceEngineFixed.get_base_url_from_model_name_v(ENDPOINT_NAME, "/v1"),

        ),
        # inference_engine=CrossProviderInferenceEngine(
        #     max_tokens=1024,
        #     seed=42,
        #     temperature=0,
        #     provider="rits",
        #     model="meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        # ),
        evaluator_name="LLAMA3_4_MAVERICK", # TODO: LLAMA4_4_MAVERICK ???
        generate_summaries=False,
    )

    add_to_catalog(laaj_direct_rits_phi4, name="metrics.llm_as_judge.direct.rits.llama4_maverick", overwrite=True, catalog_path=catalog_path)
