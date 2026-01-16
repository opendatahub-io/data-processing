import sys
from typing import cast, Any, Hashable, Sequence, Collection
import json
import argparse
import pandas as pd
import itertools

import jsonl_util
import context_util

# ARG_OPTION="glossary_option"
ARG_SHORT_CONTEXT="short_context"
ARG_INPUT_QA_FILE="qa_file"
# ARG_INPUT_GLOSSARY_FILE="glossary_file"
ARG_INPUT_CONTEXT_FILE="context_file"
ARG_OUTPUT_ICL_FILE="out_icl_file"
# OPT_GLOSSARY_APPENDIX="appendix"
# OPT_GLOSSARY_HEADER="header"
# OPT_GLOSSARY_NONE="none"
NUM_ICL_QA_EXAMPLES=3

def config():
    # print("Usage: python script.py <option> <input_qa.csv> <input_glossary.csv> <output_context.csv")
    parser = argparse.ArgumentParser(description="Make in-context learning examples from QAs and glossaries.")
    # parser.add_argument(
    #     '--' + ARG_OPTION, 
    #     type=str, 
    #     default=OPT_GLOSSARY_APPENDIX, 
    #     choices=[OPT_GLOSSARY_APPENDIX, OPT_GLOSSARY_HEADER, OPT_GLOSSARY_NONE],
    # )
    parser.add_argument('--' + ARG_SHORT_CONTEXT, action="store_true", default=False)
    parser.add_argument('--' + ARG_INPUT_QA_FILE, type=str, required=True, metavar="input_qa.csv")
    # parser.add_argument('--' + ARG_INPUT_GLOSSARY_FILE, type=str, required=True, metavar="input_glossary.csv")
    parser.add_argument('--' + ARG_INPUT_CONTEXT_FILE, type=str, required=True, metavar="input_context.csv")
    parser.add_argument('--' + ARG_OUTPUT_ICL_FILE, type=str, required=True, metavar="output_icl.jsonl")

    args = parser.parse_args()

    args = vars(args)
    
    return args

def generate_QA_combinations(row_sr: pd.Series, size: int)-> list[tuple[int, list[int]]]:
    q_list = cast(list[int], json.loads(row_sr["qlist"]))
    if len(q_list) >= size:
        comb_triples = itertools.combinations(q_list, size)
        comb_triples_list = [(cast(int, row_sr.name), list(comb)) for comb in comb_triples]
        return comb_triples_list
    if len(q_list) > 0:
        comb_triples_list = [(cast(int, row_sr.name), (q_list + [q_list[0]] * (size - len(q_list))))]
        return comb_triples_list
    return []

def compose_short_context(qa_df: pd.DataFrame, section_index: int, qa_index: list[int])-> str:
    sub_qa_df = qa_df.loc[qa_index]
    return context_util.compose_context(section_index, sub_qa_df)

def generate_ICL_example(
        context_qa_index: tuple[int, list[int]], 
        qa_df: pd.DataFrame, 
        context_df: pd.DataFrame, 
        short_context: bool=False,
    )-> dict[str, Any]:

    # TODO:
    # add a mode to compress the size of the context (i.e., the document).
    # re-create a document from QAs in context_qa_index[1]
    document = (
        context_df.loc[context_qa_index[0]]["context"] 
        if not short_context else 
        compose_short_context(qa_df, context_qa_index[0], context_qa_index[1])
    )
    icl_document = { "icl_document": document }
    icl_qa_list = [
        {
            "icl_query_" + str(i + 1): qa_df.loc[qa]["Question"].removeprefix("問 ").strip(),
            "icl_response_" + str(i + 1): qa_df.loc[qa]["Answer"].removeprefix("[Ａ]").strip(),
        } for (i, qa) in enumerate(context_qa_index[1])
    ]
    icl_qa_dict = {
        k: v for qa in icl_qa_list for (k, v) in qa.items()
    }
    icl_example = { **icl_document, **icl_qa_dict }
    return icl_example

def make_icl(qa_df: pd.DataFrame, context_df: pd.DataFrame, short_context: bool) -> list[dict[str, Any]]:

    ncontext_df = context_df.query("section != -1")
   
    icl_qa_index_sr = ncontext_df.apply(lambda row: generate_QA_combinations(row, NUM_ICL_QA_EXAMPLES), axis=1)
    flatten_icl_qa_index_list = list(itertools.chain.from_iterable(icl_qa_index_sr))
    icl_list = [generate_ICL_example(index_tuple, qa_df, ncontext_df, short_context) for index_tuple in flatten_icl_qa_index_list]

    return icl_list

def main()-> None:

    args = config()

    input_qa_path = args[ARG_INPUT_QA_FILE]
    input_context_path = args[ARG_INPUT_CONTEXT_FILE]
    output_icl_path = args[ARG_OUTPUT_ICL_FILE]
    short_context = args[ARG_SHORT_CONTEXT]

    qa_df = pd.read_csv(input_qa_path, encoding="utf8")
    context_df = pd.read_csv(input_context_path, encoding="utf8")

    icl_list = make_icl(qa_df, context_df, short_context)

    jsonl_util.write_jsonl_file(output_icl_path, icl_list)

    # TODO: 
    # Support for glossary-type context.
    # The current context (ICL document) size seems to be too large.
    # - Select three as context, assign corresponding QAs
    #   - Combination(14,3)=14x13x12/3x2x1=14x13x2=2(10+4)x(10+3)=2(100+70+12)=364 patterns. too much.
    # - For each sections, select three as context and assign corresponding QAs.
    # - Select 5-6 as context, assign 3 corresponding QAs by enumerate all the possible combinations.
    # - 

if __name__ == "__main__":
    main()
