#!/bin/bash
# Remove old generated dataset
rm -rf preprocessing/{{al,simple}-features,translated}/{hh,rw} datasets/*.hdf5

# Generate datasets again from raw data
cd preprocessing/translated
./translate.sh ../raw/*/*.txt

cd ../al-features
./export_data.sh ../translated/*/*.txt

cd ../simple-features
./export_data.sh ../translated/*/*.txt

cd ../../datasets
python3 generate.py

cd ..
