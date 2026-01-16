
import sys
import argparse
from typing import cast, Any, Hashable, Sequence, Collection
import json
import pandas as pd

import jsonl_util

ARG_INPUT_CONTEXT_FILE="context_file"
ARG_INPUT_ICL_FILE="icl_file"
ARG_OUTPUT_SEED_FILE="out_seed_file"
ARG_JOIN_METHOD="join_method"
OPT_JOIN_METHOD_CARTESIAN="cartesian"
OPT_JOIN_METHOD_SLIDE="slide"

def config():
    parser = argparse.ArgumentParser(description='make a seed sample file from contexts and ICL samples')
    parser.add_argument('--' + ARG_INPUT_CONTEXT_FILE, type=str, required=True)
    parser.add_argument('--' + ARG_INPUT_ICL_FILE, type=str, required=True)
    parser.add_argument('--' + ARG_OUTPUT_SEED_FILE, type=str, required=True)
    parser.add_argument(
        '--' + ARG_JOIN_METHOD, 
        type=str, 
        default=OPT_JOIN_METHOD_SLIDE, 
        choices=[OPT_JOIN_METHOD_SLIDE, OPT_JOIN_METHOD_CARTESIAN],
    )
    # parser.add_argument('--device', type=str, default="cuda")

    args = parser.parse_args()
    return vars(args)

def select_icl_sample(icl_df: pd.DataFrame, average_context_length: int, index: int) -> pd.Series:
    if icl_df.empty:
        raise ValueError("ICL sample DataFrame is empty.")
    step = max(1, average_context_length)
    idx = int(index)
    ret_index = (idx - step) % len(icl_df)
    return cast(pd.Series, icl_df.iloc[ret_index])


def make_seed(context_df: pd.DataFrame, icl_list: list[Any], opt_join_method: str) -> pd.DataFrame:
    icl_df = pd.DataFrame(icl_list)

    if opt_join_method == OPT_JOIN_METHOD_SLIDE:
        # TODO: 
        # assumption. 
        # ICL list is sorted in the order of the original FAQ document. 
        # Each ICL doc sample is taken from an answer of the FAQ document.
        # Contexts are also derived from the FAQ document.
        # We should avoid ICL samples that are too much similar to a context to be assigned to that context.
        # average_context_length = len(icl_list) / len(context_df)
        # assigned_icl_df = context_df.apply(lambda x: select_icl_sample(icl_df, int(average_context_length), x["qindex"]), axis=1)
        if len(context_df) == 0:
            raise ValueError("No contexts loaded.")
        if len(icl_df) == 0:
            raise ValueError("No ICL samples loaded.")
        average_context_length = round(len(icl_df) / len(context_df))
        assigned_icl_df = context_df.apply(
            lambda x: select_icl_sample(icl_df, int(average_context_length), int(x["qindex"])),
            axis=1,
        )
        context_icl_df = pd.concat([
            assigned_icl_df,
            context_df.drop(columns=["qindex"]), 
        ], axis=1)
    elif opt_join_method == OPT_JOIN_METHOD_CARTESIAN:
        # TODO:
        # outer-join ICL and Contexts.
        # This is for augmentation of the seed samples.
        tmp_list = [
            pd.concat([
                icl_df,
                pd.DataFrame(sr.to_dict(), index=icl_df.index), 
            ], axis=1) for (idx, sr) in context_df.iterrows()
        ]
        context_icl_df = pd.concat(tmp_list, axis=0).reset_index(drop=True).drop(columns=["qindex"])
    else:
        raise argparse.ArgumentError(None, f"Invalid {ARG_JOIN_METHOD}: {opt_join_method}")
    # schema
    # "document_outline": "...",
    # "document_title": "0024001-021.md",
    # "domain": "gensen",
    # "icl_document": "...",
    # "icl_query_1": "...",
    # "icl_response_1": "...",
    # "icl_query_2": "...",
    # "icl_response_2": "...",
    # "icl_query_3": "...",
    # "icl_response_3": "...",
    # "document": "..."
    out_df = pd.concat([
        pd.DataFrame({
            "document_outline": "令和６年分所得税の定額減税", 
            "document_title": "0024001-021.md",
            "domain": "gensen",
        }, index=context_icl_df.index),
        context_icl_df,
    ], axis=1).rename(columns={
        "context": "document",
    })
    return out_df

def main()-> None:

    args = config()
    input_context_path = args[ARG_INPUT_CONTEXT_FILE]
    input_icl_path = args[ARG_INPUT_ICL_FILE]
    output_context_path = args[ARG_OUTPUT_SEED_FILE]
    opt_join_method = args[ARG_JOIN_METHOD]

    context_df = pd.read_csv(input_context_path, encoding="utf8")[["context", "qindex"]]
    icl_list = jsonl_util.read_jsonl_file(input_icl_path)

    out_df = make_seed(context_df, icl_list, opt_join_method)
    
    out_df.to_json(output_context_path, orient="records", lines=True, force_ascii=False)

if __name__ == "__main__":

    main()
