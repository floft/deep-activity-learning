#!/bin/bash
# Show the class [im]balance
# Percentages code: https://stackoverflow.com/a/3802965
echo "Class [im]balance:"
cat preprocessing/translated/hh/*.txt | cut -d' ' -f5 | sort | uniq -c | \
    awk '{array[$2]=$1; sum+=$1} END { for (i in array) printf "%-25s %-15d %6.2f%%\n", i, array[i], array[i]/sum*100}' | sort -r -k2,2 -n
