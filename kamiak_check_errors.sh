#!/bin/bash
#
# Check for errors like "UnknownError" that CUDA sometimes gives
#
cd slurm_logs
error_files="$(grep -R UnknownError | sed -r 's/^([^:]*):.*$/\1/g' | sort -u | \
    tr '\n' ' ')"
out_files="$(sed 's/\.err/\.out/g' <<< "$error_files")"

echo "Head"
head -n 1 $out_files

echo "Tail"
tail -n 25 $out_files
