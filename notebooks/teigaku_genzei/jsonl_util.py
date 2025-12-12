
from typing import Any
import json



def write_jsonl_file(file_path: str, data_list: list[Any]):
    with open(file_path, 'w', encoding='utf-8') as file:
        for data in data_list:
            file.write(json.dumps(data, ensure_ascii=False) + '\n')

def read_jsonl_file(file_path: str) -> list[Any]:
    with open(file_path, 'r', encoding="utf-8") as file:
        data_list = [
            json.loads(line) for line in file
        ]
        return data_list
