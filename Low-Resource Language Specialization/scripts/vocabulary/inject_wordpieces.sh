#!/bin/bash

ARGS=( "$@" )
BASE_VOCAB=("${ARGS[0]}")
NEW_VOCAB=("${ARGS[1]}")
# OUTPUT should be a new file
OUTPUT=("${ARGS[2]}")

NEW_LENGTH=`wc -l < $NEW_VOCAB`
if [ $NEW_LENGTH -gt 99 ]; then
    echo "New vocab length is too long - $NEW_LENGTH"
    exit
fi

# output is 1st line of base, 2-100th of new, and remainder of base
head -n 1 $BASE_VOCAB >> $OUTPUT
cat $NEW_VOCAB >> $OUTPUT
tail -n +101 $BASE_VOCAB >> $OUTPUT

BASE_LENGTH=`wc -l < $BASE_VOCAB`
OUTPUT_LENGTH=`wc -l < $OUTPUT`
if [ $BASE_LENGTH -ne $OUTPUT_LENGTH ]; then
    echo "Something went wrong - base length $BASE_LENGTH but output length $OUTPUT_LENGTH"
fi