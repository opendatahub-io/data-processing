# TODO:
# Read a given JSON file and extract fields that corresponds to the glossary.
# Save the glossary pairs to a CSV file.
# Both the input file name and the output file name are specified as command line arguments.

# Assisted by watsonx Code Assistant 
import json
import csv
import sys
import re
from typing import Any

import json_util


def get_desc_block(text_list: list[Any], begin_pos: int) -> str:

    def is_body_text(x: dict[str, Any]) -> bool:
        return x["content_layer"] == "body" and x["label"] == "text"

    def has_desc(text_list: list[Any], pos: int) -> bool:
        if pos + 1 >= len(text_list):
            return False
        the_text = text_list[pos + 1]
        return is_body_text(the_text) and not json_util.is_title(the_text["text"])

    desc_tmp = text_list[begin_pos + 1]["text"] if has_desc(text_list, begin_pos) else ""
    return desc_tmp

def extract_glossary(data):

    def is_section_title(x: dict[str, Any]) -> bool:
        return x["label"] == "section_header" and re.match(r"^[【（].*[）】]$", x["text"]) is not None

    glossary_list = []
    text_list = data["texts"]
    term_list = filter(lambda x: is_section_title(x[1]), enumerate(text_list))
    term_pos_list = [ti for (ti, title) in term_list]
    # print(title_pos_list)

    for ti in term_pos_list:
        term = text_list[ti]["text"]
        desc = get_desc_block(text_list, ti)
        if len(desc) > 0:
            glossary_list.append((term, desc))
    return glossary_list

def save_to_csv(qa_pairs, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Term', 'Description'])
        for qa in qa_pairs:
            writer.writerow(qa)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_glossary.py <input_json_file> <output_csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    data = json_util.read_json_file(input_file)
    qa_pairs = extract_glossary(data)
    save_to_csv(qa_pairs, output_file)

    print(f"Glossary pairs have been successfully saved to {output_file}")
