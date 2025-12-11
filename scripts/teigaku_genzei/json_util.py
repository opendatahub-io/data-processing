
from typing import cast, Any, Hashable, Sequence, Collection
import re
import json

def is_title(text: str) -> bool:
    return re.match(r"[0-9０-９]+[－ー][0-9０-９]+", text) is not None

def is_toc(obj: dict[str, Any]) -> bool:
    return re.match(r"^#/pictures/", obj["parent"]["$ref"]) is not None

def get_title_match(text: str) -> re.Match | None:
    return re.match(r"([0-9０-９]+)[－ー]([0-9０-９]+)", text)

def is_qhead(text: str) -> bool:
    return re.match(r"^問", text) is not None

def is_ahead(text: str) -> bool:
    return re.match(r"^\[Ａ\]", text) is not None

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

