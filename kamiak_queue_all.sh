#!/bin/bash
#
# Start up all the kamiak_train.srun jobs
#
features="al"
model="flat"
datasets=(hh{{101..106},{108..120},{122..130}})

for target in "${datasets[@]}"; do
    echo "Queueing $target"
    sbatch kamiak_train.srun --features="$features" --target="$target" --"$model"
done
