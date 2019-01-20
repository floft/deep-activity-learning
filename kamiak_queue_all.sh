#!/bin/bash
#
# Start up all the kamiak_train.srun jobs for all datasets in the cross validation
#
if [[ -z $1 ]]; then
    echo "Usage:"
    echo "  ./kamiak_queue_all.sh flat --dataset=al.zip --features=al --flat # flat model"
    echo "  ./kamiak_queue_all.sh flat-da --dataset=al.zip --features=al --flat --adapt # flat model with adaptation"
    echo "  Note: outputs to kamiak-{models,logs}-suffix where suffix is the first argument"
    exit 1
fi

. kamiak_config.sh

# We'll run all on these datasets
datasets=(hh{{101..106},{108..120},{122..130}})

# First argument is the suffix, then the rest are arguments for the training
suffix="$1"
modelFolder="$modelFolder-$suffix"
logFolder="$logFolder-$suffix"
shift

for target in "${datasets[@]}"; do
    echo "Queueing $target"
    sbatch kamiak_train.srun --target="$target" \
        --logdir="$logFolder" --modeldir="$modelFolder" "$@"
done
