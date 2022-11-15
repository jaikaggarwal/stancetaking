#!/bin/bash

counter=0
for d in /ais/hal9000/datasets/reddit/stance_analysis/*_files/; do
    echo "$d"
    files="$(find $d -type f)"
    for f in $files; do
        shuf $f -n 50000 -o /ais/hal9000/datasets/reddit/stance_analysis/wang2vec_sample/${counter}.txt
        counter=$((counter+1))
        # echo "$counter"
    done
done
echo "$counter"