#!/bin/bash
#
# It's easy to overwhelm tensorboard by giving it too many files.
# Thus, take a subset of them.
#
dir="$1"
port="$2"

if [[ -z $dir || ! -d $dir ]]; then
    echo "Usage: ./tensorboard_some.sh kamiak-logs-flat [port]"
    exit 1
fi

# Pick a subset
logdir="$(for i in "$dir"/hh{103..106}-*-{0..2}/; do
    [[ -d $i ]] && echo -n $(basename $i):$i,
done | sed 's/,$//g')"

# Default port
[[ -z $port ]] && port="6006"

# Run
tensorboard --logdir="$logdir" --port="$port"
