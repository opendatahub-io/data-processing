#!/bin/sh

# python ./make_seed.py context/0024001-021_context.csv tmp/icl.jsonl tmp/seed_ja.jsonl
python ./make_seed.py --context_file context/0024001-021_context.csv --icl_file icl/icl.jsonl --out_seed_file tmp/seed_ja.jsonl --join_method cartesian
