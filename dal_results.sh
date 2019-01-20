#!/bin/bash
#
# Compute DAL results by picking best model on validation data and then
# evaluating that model on the train/test sets computing the accuracy
#
from="kamiak" # kamiak or cv, depending on where you trained
models="$from-models"
logs="$from-logs"

if [[ -z $1 ]]; then
    echo "Usage:"
    echo "  ./dal_results.sh flat --features=al"
    exit 1
fi

# First argument is the suffix, then the rest are arguments for the training
suffix="$1"
models="$models-$suffix"
logs="$logs-$suffix"
shift

out="dal_results_$suffix.txt"
echo "Args: $@" > "$out"
./dal_eval.py --modeldir="$models" --logdir="$logs" "$@" | tee -a "$out"
