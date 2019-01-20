#!/bin/bash
#
# Compute DAL results by picking best model on validation data and then
# evaluating that model on the train/test sets computing the accuracy
#
from="kamiak" # kamiak or cv, depending on where you trained
models="$from-models"
logs="$from-logs"

./dal_eval.py --features=al --modeldir="$models" --logdir="$logs" | tee dal_results.txt
