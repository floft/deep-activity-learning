#!/bin/bash
#
# Start up all the kamiak_train_single.srun jobs for all datasets in the cross
# validation. This version splits by fold.
#
if [[ -z $1 ]]; then
    echo "Usage:"
    echo "  ./kamiak_queue_all_folds.sh flat --dataset=al.zip --features=al --flat"
    exit 1
fi

. kamiak_config.sh

# We'll run all on these datasets
datasets=(hh{{101..106},{108..120},{122..130}})
folds=(0..2)

# First argument is the suffix, then the rest are arguments for the training
suffix="$1"
modelFolder="$modelFolder-$suffix"
logFolder="$logFolder-$suffix"
shift

for target in "${datasets[@]}"; do
    echo "Queueing $target"
    sbatch kamiak_train_single.srun --target="$target" \
        --logdir="$logFolder" --modeldir="$modelFolder" \
        --fold=0 --debug-num=0 "$@"
    sbatch kamiak_train_single.srun --target="$target" \
        --logdir="$logFolder" --modeldir="$modelFolder" \
        --fold=1 --debug-num=1 "$@"
    sbatch kamiak_train_single.srun --target="$target" \
        --logdir="$logFolder" --modeldir="$modelFolder" \
        --fold=2 --debug-num=2 "$@"
done
