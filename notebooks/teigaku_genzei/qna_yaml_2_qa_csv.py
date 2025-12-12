
# load YAML file.
# for each "questions_and_answers", select "question" and "answer".
# save as a CSV file.

# import sys
# import yaml
# # import csv
# import pandas as pd

# input_file = sys.argv[1]
# output_file = sys.argv[2]

# # load YAML file
# with open(input_file, "r") as file:
#     data = yaml.load(file, Loader=yaml.FullLoader)
#     # separate data
#     print(data)
#     qa_list = [
#         {
#             "Title": f"{isample + 1}-{iqa}",
#             "Question": qa["question"], 
#             "Answer": qa["answer"]
#         }
#         for (isample,sample) in enumerate(data["seed_examples"]) for (iqa, qa) in enumerate(sample["questions_and_answers"])
#     ]
#     df = pd.DataFrame(qa_list)
#     df.to_csv(output_file, index=False, encoding="utf8") 

import sys
import yaml
import pandas as pd

def main():
    if len(sys.argv) != 3:
        print("Usage: qna_yaml_2_qa_csv.py <input_yaml> <output_csv>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
 
    try:
        # load YAML file
        with open(input_file, "r") as file:
            data = yaml.safe_load(file)
        
        # Validate structure
        if "seed_examples" not in data:
            raise ValueError("YAML file missing 'seed_examples' key")
        
        # Extract Q&A pairs
        qa_list = [
            {
                "Title": f"{isample + 1}-{iqa}",
                "Question": qa["question"], 
                "Answer": qa["answer"]
            }
            for (isample, sample) in enumerate(data["seed_examples"]) 
            for (iqa, qa) in enumerate(sample["questions_and_answers"])
        ]
        
        # Write CSV
        df = pd.DataFrame(qa_list)
        df.to_csv(output_file, index=False, encoding="utf8")
        print(f"Successfully wrote {len(qa_list)} Q&A pairs to {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML format - {e}", file=sys.stderr)
        sys.exit(1)
    except (KeyError, ValueError) as e:
        print(f"Error: Invalid YAML structure - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
