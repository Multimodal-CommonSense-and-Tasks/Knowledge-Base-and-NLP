#!/bin/bash

for ((i=0; i<=100; i++)); do
    python src/rationale_annotation/distill_rationale_cicero.py --split=train --n_frag=100 --frag_index=$i --n_shot=5

done
