#!/bin/bash
# Remove old generated dataset
rm -rf preprocessing/{al,simple}-features/{hh,rw} datasets/*.hdf5

# Generate datasets again from raw data
cd preprocessing/al-features
./export_data.sh ../raw/*/*.txt

cd ../simple-features
./export_data.sh ../raw/*/*.txt

cd ../../datasets
python3 generate.py

cd ..
