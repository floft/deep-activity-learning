#!/bin/bash
#
# Convert from format like Relax="begin" to format with Relax at end of lines
#

# Compile converter
[[ ! -e convert ]] && gcc convert1.c -o convert

# Run on all files
for folder in ../orig/*/; do
    folderbase="$(basename "$folder")"
    converted="./$folderbase"
    mkdir -p "$converted"

    for file in "$folder"/*.txt; do
        filebase="$(basename "$file")"
        in="$folder/$filebase"
        out="$converted/$filebase"
        echo "Converting $in to $out"
        ./convert "$in" > "$out"
    done
done
