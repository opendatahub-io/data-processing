import sys
from typing import cast, Any

import jsonl_util

def extract_icl(obj: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    ret = {key: obj[key] for key in keys}
    return ret

def compute_dict_hash(obj: dict[str, str], keys: list[str]) -> str:
    value_list = [obj[k] for k in keys]
    value_hash = "::".join(value_list)
    return value_hash

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_icl_from_seed.py <input_jsonl_file> <output_jsonl_file>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    data_list = jsonl_util.read_jsonl_file(input_file_path)

    print(len(data_list))
    print(data_list[0])
    obj_list = cast(list[dict[str, str]], data_list)

    # TODO: remove "document", "document_title", "domain", "document_outline"
    selected_keys = [
        "icl_document",
        # "icl_query_1",
        # "icl_response_1",
        # "icl_query_2",
        # "icl_response_2",
        # "icl_query_3",
        # "icl_response_3",
    ]
    # selected_keys = [
    #     "document"
    # ]
    icl_list = [
        extract_icl(obj, selected_keys) for obj in obj_list
    ]
    hash_list = [
        (compute_dict_hash(obj, selected_keys), obj) for obj in icl_list
    ]
    unique_icl_dict: dict[str, dict[str, Any]] = {
        h: obj for (h, obj) in hash_list
    }
    unique_icl_list = [obj for (_, obj) in unique_icl_dict.items()]

    print(len(unique_icl_list))

    jsonl_util.write_jsonl_file(output_file_path, unique_icl_list)

