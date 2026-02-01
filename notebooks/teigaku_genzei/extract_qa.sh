#!/bin/sh

source_dir="docparser"
source_qa="0024001-021 0024004-072_01"
qa_table_dir="qa_table"

for sqa in ${source_qa}; do
    sf=${source_dir}/${sqa}.json
    qf=${qa_table_dir}/${sqa}.csv
    gf=${qa_table_dir}/${sqa}_glossary.csv

    python ./json_qa.py ${sf} ${qf}
    python ./json_glossary.py ${sf} ${gf}
done

