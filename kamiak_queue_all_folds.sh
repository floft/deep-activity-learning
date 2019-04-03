#!/bin/bash
#
# Start up all the kamiak_train_single.srun jobs for all datasets in the cross
# validation (over just folds, no target specified). This version splits by fold.
#
if [[ -z $1 ]]; then
    echo "Usage:"
    echo "  ./kamiak_queue_all_folds.sh flat --dataset=al.zip --features=al --model=flat"
    exit 1
fi

. kamiak_config.sh

# We'll run all on these folds
folds=(0..2)

# First argument is the suffix, then the rest are arguments for the training
suffix="$1"
modelFolder="$modelFolder-$suffix"
logFolder="$logFolder-$suffix"
shift

sbatch -J "$suffix" kamiak_train_single.srun \
    --logdir="$logFolder" --modeldir="$modelFolder" \
    --fold=0 "$@"
sbatch -J "$suffix" kamiak_train_single.srun \
    --logdir="$logFolder" --modeldir="$modelFolder" \
    --fold=1 "$@"
sbatch -J "$suffix" kamiak_train_single.srun \
    --logdir="$logFolder" --modeldir="$modelFolder" \
    --fold=2 "$@"
