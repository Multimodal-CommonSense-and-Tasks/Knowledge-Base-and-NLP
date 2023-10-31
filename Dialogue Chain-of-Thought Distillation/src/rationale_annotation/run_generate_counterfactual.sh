#!/bin/bash

for ((i=0; i<=100; i++)); do
    python src/rationale_annotation/generate_counter_factual_thoughts.py --split=train --n_frag=100 --frag_index=$i --n_shot=5

done

