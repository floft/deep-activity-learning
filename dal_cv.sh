#!/bin/bash
#
# Run several DAL instances at one time -- multiple folds, multiple targets
#
. /scripts/threading
thread_init
_thread_count=3 # it's one more, so it'll do 4 at once

logs="cv-logs"
models="cv-models"
mem="0.2" # 0.8/4 = 0.2
features="al"
model="flat"
datasets=(hh{101..130})
folds=({0..2})

# Do all the homes
for target in "${datasets[@]}"; do
    # Check the dataset exists
    if [[ ! -f datasets/al_${target}_train_0.tfrecord ]]; then
        echo "Skipping $target"
        continue
    fi

    # Do all the folds for each home
    for fold in "${folds[@]}"; do
        echo "Processing $target fold $fold"
        ./dal.py \
            --features="$features" \
            --logdir="$logs" --modeldir="$models" \
            --fold="$fold" --target="$target" \
            --"$model" \
            --gpu-mem="$mem" --debug-num="$fold" &
        thread_wait
    done
done

thread_finish
