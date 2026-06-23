#!/bin/sh

DATA_PREFIX_LIST="0024001-021 0024004-072_01"

for f in $DATA_PREFIX_LIST; do
    python ./make_context.py \
        --glossary_option appendix \
        --qa_file qa_table/${f}.csv \
        --glossary_file qa_table/${f}_glossary.csv \
        --out_context_file context/${f}_context.csv
done

# ARG_OPTION="glossary_option"
# ARG_INPUT_QA_FILE="qa_file"
# ARG_INPUT_GLOSSARY_FILE="glossary_file"
# ARG_OUTPUT_CONTEXT_FILE="out_context_file"
# OPT_GLOSSARY_APPENDIX="appendix"
# OPT_GLOSSARY_HEADER="header"
# OPT_GLOSSARY_NONE="none"
