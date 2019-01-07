#!/bin/bash
#
# Use the al.py script to output AL features and labels for each of the smart
# home datasets
#
if [[ -z $1 ]]; then
    echo "Usage: time ./export_data.sh ../raw/*/*.txt"
    exit 1
fi

echo "Updating sensor/activity lists"
sensor_names="$(cat "$@" | cut -d' ' -f3 | sort -u | tr '\n' ' ')"
activity_labels="$(cat "$@" | cut -d' ' -f5 | sort -u | tr '\n' ' ')"
sed -i "s#^sensors .*\$#sensors $sensor_names#g" "al.config"
sed -i "s#^activities .*\$#activities $activity_labels#g" "al.config"

. /scripts/threading
thread_init

for i; do
    echo "Processing $i"

    # e.g. if ../raw/rw/rw101.txt this will make a "rw" directory
    dir=$(basename -- $(dirname -- "$i"))
    mkdir -p "$dir"

    # e.g. if ../raw/rw/rw101.txt this will get "rw101"
    filename=$(basename -- "$i")
    filename="${filename%.*}"

    # e.g. if ../raw/rw/rw101.txt this will output to a file rw/rw101.hdf5
    python2 al.py al.config "$i" "${dir}/${filename}.hdf5" &

    thread_wait
done

thread_finish
