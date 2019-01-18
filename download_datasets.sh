#!/bin/bash
#
# Download CASAS smart home datasets
#
mkdir -p preprocessing/orig/hh
cd preprocessing/orig/hh

# Download single-resident hh* datasets if not already downloaded
# For the full list, see http://casas.wsu.edu/datasets/
wget -nc http://casas.wsu.edu/datasets/hh{{101..106},{108..120},{122..130}}.zip

# Extract
for i in *.zip; do
    [[ ! -e ${i//.zip/}/ ]] && unzip $i || echo "Already unzipped $i"
done

# Move all of the hh101/ann.txt files into hh101.txt, etc.
for i in */; do
    f=${i//\//}
    [[ -e $f/ann.txt ]] && mv $f/ann.txt $f.txt
done

# Convert from format like Relax="begin" to format with Relax at end of lines
cd ../../raw
./generate_raw.sh
