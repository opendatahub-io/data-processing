from unitxt.inference import CrossProviderInferenceEngine, RITSInferenceEngine
from unitxt.llm_as_judge import LLMJudgeDirect
from unitxt import add_to_catalog

def register(catalog_path: str) -> None:
    laaj_direct_rits_phi4 = LLMJudgeDirect(
        inference_engine=RITSInferenceEngine(
            max_tokens=1024,
            seed=42,
            temperature=0,
            model_name="microsoft/phi-4",
        ),
        # inference_engine=CrossProviderInferenceEngine(
        #     max_tokens=1024,
        #     seed=42,
        #     temperature=0,
        #     provider="rits",
        #     model="microsoft/phi-4",
        # ),
        evaluator_name="MICROSOFT_PHI_4",
        generate_summaries=False,
    )

    add_to_catalog(laaj_direct_rits_phi4, name="metrics.llm_as_judge.direct.rits.phi_4", overwrite=True, catalog_path=catalog_path)
