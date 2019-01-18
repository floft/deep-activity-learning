#!/bin/bash
#
# Grab the accuracy results from all the targets/folds from tensorboard
# after running ./dal_cv.sh that trains the networks
#
results="cv-results"
logs="cv-logs"
models="cv-models"
datasets=(hh{101..130})
folds=({0..2})

# Run tensorboard to process the logs
tensorboard --logdir "$logs" &
tb="$!"
trap "echo \"Exiting...\"; kill $tb; exit 1" 2 15

echo "Press return when the results show up in tensorboard."
read -r line

download_results() {
    target="$1"
    fold="$2"

    prefix="http://localhost:6006/data/plugin/scalars/scalars?tag=accuracy_task%2F"
    suffix="&experiment=&format=csv"
    curl "${prefix}source%2Ftraining&run=${target}-flat-${fold}${suffix}" \
        > "${results}/${target}-${fold}-source-training.csv"
    echo $?
    curl "${prefix}source%2Fvalidation&run=${target}-flat-${fold}${suffix}" \
        > "${results}/${target}-${fold}-source-validation.csv"
    curl "${prefix}target%2Ftraining&run=${target}-flat-${fold}${suffix}" \
        > "${results}/${target}-${fold}-target-training.csv"
    curl "${prefix}target%2Fvalidation&run=${target}-flat-${fold}${suffix}" \
        > "${results}/${target}-${fold}-target-validation.csv"
}

# Grab all the results from tensorboard
mkdir -p "$results"

# All homes
for target in "${datasets[@]}"; do
    # Check the dataset exists
    if [[ ! -f datasets/al_${target}_train_0.tfrecord ]]; then
        echo "Skipping $target"
        continue
    fi

    # All folds
    for fold in "${folds[@]}"; do
        echo "Processing $target fold $fold"
        download_results "$target" "$fold"
    done
done

# Kill tensorboard
trap - 2 15
kill $tb
