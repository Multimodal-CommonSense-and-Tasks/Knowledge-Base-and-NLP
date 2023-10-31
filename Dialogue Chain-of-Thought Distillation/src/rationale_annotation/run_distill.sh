#!/bin/bash

for ((i=72; i<=10000; i++)); do
    python src/rationale_annotation/distill_rationale_soda.py --split=train --n_frag=10000 --frag_index=$i
done
