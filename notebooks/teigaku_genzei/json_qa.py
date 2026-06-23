# TODO:
# Read a given JSON file and extract fields that corresponds to QA pairs.
# Save the QA pairs to a CSV file.
# Both the input file name and the output file name are specified as command line arguments.

# Assisted by watsonx Code Assistant 
import json
import csv
import sys
import re
from typing import Any

import json_util


def get_text_block(text_list: list[Any], begin_pos: int, end_pos: int) -> str:

    def is_body_text(x: dict[str, Any]) -> bool:
        return x["content_layer"] == "body" 

    def is_section_title(x: dict[str, Any]) -> bool:
        return x["label"] == "section_header" and re.match(r"^[【（].*[）】]$", x["text"]) is not None

    block_list = text_list[begin_pos:end_pos]
    block_tmp_itr = filter(lambda x: is_body_text(x) and not is_section_title(x), block_list)
    block_text_list = [block["text"] for block in block_tmp_itr]
    # page numbers can be eliminated by attributes.
    # obj["label"] in {"text", "list_item", "section_header", "page_footer"}
    # For page numbers.
    #   obj["content_layer"] == "furniture" and obj["label"] == "page_footer"
    # For document body texts.
    #   obj["content_layer"] == "body" and obj["label"] == "text" or "list_item"
    # For 問 and [Ａ] (the first line) and section headers like 【各人別控除事績簿】
    #   obj["content_layer"] == "body" and obj["label"] == "section_header"
    # For title like 11ー１  各人別控除事績簿の作成の要否
    #   obj["content_layer"] == "body" and obj["label"] == "list_item"

    return "".join(block_text_list)

def extract_qa_pairs(data):
    qa_pairs = []
    text_list = data["texts"]
    title_list = filter(lambda x: json_util.is_title(x[1]["text"]) and not json_util.is_toc(x[1]), enumerate(text_list))
    qhead_list = filter(lambda x: json_util.is_qhead(x[1]["text"]), enumerate(text_list))
    ahead_list = filter(lambda x: json_util.is_ahead(x[1]["text"]), enumerate(text_list))
    title_pos_list = [ti for (ti, title) in title_list]
    qhead_pos_list = [qi for (qi, qhead) in qhead_list]
    ahead_pos_list = [ai for (ai, ahead) in ahead_list]
    atail_pos_list = title_pos_list[1:] + [len(text_list)]

    for (ti, qhi, ahi, ati) in zip(title_pos_list, qhead_pos_list, ahead_pos_list, atail_pos_list, strict=True):
        title = text_list[ti]["text"]
        question = get_text_block(text_list, qhi, ahi)
        answer = get_text_block(text_list, ahi, ati)
        qa_pairs.append((title, question, answer))
    return qa_pairs

def save_to_csv(qa_pairs, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Title', 'Question', 'Answer'])
        for qa in qa_pairs:
            writer.writerow(qa)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_json_file> <output_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    data = json_util.read_json_file(input_file)
    qa_pairs = extract_qa_pairs(data)
    save_to_csv(qa_pairs, output_file)

    print(f"QA pairs have been successfully saved to {output_file}")
