from typing import Any
import sys
import yaml
# import csv
import jsonl_util



def conv_seed_to_cqa(seed: Any) -> dict[str, Any]:

    qa_obj_list = [
        {
            "question": seed[f"icl_query_{i+1}"],
            "answer": seed[f"icl_response_{i+1}"],
        }
        for i in range(3)
    ]
    cqa_obj = {
        "context": seed["icl_document"],
        "questions_and_answers": qa_obj_list,
    }
    return cqa_obj

def main() -> None:

    if len(sys.argv) < 3:
        print("Usage: seed_jsonl_2_qna_yaml.py <input_seed.jsonl> <output_qna.yaml>")
        sys.exit(1)
    
    input_file = sys.argv[1]  # seed.jsonl
    output_file = sys.argv[2]  # qna.yaml

    seed_list = jsonl_util.read_jsonl_file(input_file)
    cqa_list = [conv_seed_to_cqa(seed) for seed in seed_list]

    qna_template_dict = {
        "created_by": "IBM",
        "version": 3,
        "domain": "Tax payment",
        "seed_examples": cqa_list,
        "document_outline": "Teigaku Genzei example.", 
        "document": {
            "repo": 'https://example.com/example/taxonomy',
            "commit": "0000000000000000000000000000000000000000",
            "patterns": [ "example.pdf" ],
        },
    }

    with open(output_file, "wb") as out_f:
        yaml.safe_dump(qna_template_dict, out_f, allow_unicode=True, encoding="utf8", default_flow_style=False, default_style="|", sort_keys=False)

    return

if __name__ == "__main__":
    main()

# output format (YAML)
"""
created_by: IBM
version: 3
domain: gensen
seed_examples:
  # 01.md 令和6年分 年末調整のしかた
  - context: |
    questions_and_answers:
      - question: |
          年末調整手続の電子化を行うことのメリットは何ですか？
        answer: |
          給与の支払者（勤務先）及び給与所得者（従業員）それぞれにおいて、書類の作成や確認、保管などの業務全般が大幅に効率化されます。
        location: 01-3
document_outline: |
  令和６年分 年末調整のしかた
document:
  repo: 'https://example.com/example/taxonomy'
  commit: 0000000000000000000000000000000000000000
  patterns:
    - nencho_all.pdf

"""
