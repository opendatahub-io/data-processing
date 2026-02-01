from unitxt import add_to_catalog
from unitxt.llm_as_judge_constants import CriteriaOption, CriteriaWithOptions

# Source
# https://www.unitxt.ai/en/latest/catalog/catalog.metrics.llm_as_judge.direct.criteria.correctness_based_on_ground_truth.html
# Note: Using numeric option names ("3", "2", "1") instead of semantic names
# ("correct", "partially correct", "incorrect") to work around a post-processing issue
# with the original correctness_based_on_ground_truth criterion.
# We observed that in the case of some judge models, 
# unnecessary output (e.g., "The answer is correct") can be included. These should be removed and only "correct" 
# should be left as the normalized output. However, the normalization operator MatchClosestOption,
# or more specifically, difflib.get_close_matches() fails to do so when the three option names are similar,
# and can be confused with different option name.

def register(catalog_path: str) -> None:

    correctness_based_on_ground_truth2 = CriteriaWithOptions(
        name="correctness_based_on_ground_truth2",
        description="Does the response correctly convey the same factual information as the ground truth?",
        options=[
            CriteriaOption(
                # name="correct",
                name="3",
                description="The response conveys the same factual meaning as the ground truth. Minor rewording, synonyms, or grammatical differences are acceptable. The response is relevant to the question and does not introduce unrelated or misleading information.",
            ),
            CriteriaOption(
                # name="incomplete",
                name="2",
                description="The response contains some correct information but is incomplete or lacks essential details. It may also contain minor inaccuracies or extraneous information that slightly misrepresents the ground truth.",
            ),
            CriteriaOption(
                # name="wrong",
                name="1",
                description="The response does not align with the ground truth. It either presents incorrect, unrelated, or misleading information, or omits key details that change the intended meaning.",
            ),
        ],
        option_map={
            # "correct": 1.0,
            # "incomplete": 0.5,
            # "wrong": 0.0,
            "3": 1.0,
            "2": 0.5,
            "1": 0.0,
        },
    )

    add_to_catalog(correctness_based_on_ground_truth2, name="metrics.llm_as_judge.direct.criteria.correctness_based_on_ground_truth2", overwrite=True, catalog_path=catalog_path)
