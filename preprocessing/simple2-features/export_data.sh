#!/bin/bash
#
# Use the simple2_features.py script to output simple AL features and labels for
# each of the smart home datasets
#
if [[ -z $1 ]]; then
    echo "Usage: time ./export_data.sh ../raw/*/*.txt"
    exit 1
fi

# Skip computation since we do it in al-features, just copy that one
echo "Updating sensor/activity lists"
cp ../al-features/al.config .

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
    python3 simple2_features.py al.config "$i" "${dir}/${filename}.hdf5" &

    thread_wait
done

thread_finish
