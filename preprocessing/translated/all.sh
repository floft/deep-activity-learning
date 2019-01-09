#!/bin/bash
#
# Concatenate files for one massive training dataset
#
#rm -rf hh/half{1,2}.txt
#cat $(ls hh/hh1{01..15}.txt 2>/dev/null) > hh/half1.txt
#cat $(ls hh/hh1{16..30}.txt 2>/dev/null) > hh/half2.txt
cat hh/*.txt > hh/all.txt

rm -rf rw/all.txt
cat rw/*.txt > rw/all.txt
