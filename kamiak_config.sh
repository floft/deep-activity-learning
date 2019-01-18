#
# Config file for running on high performance cluster with Slurm
#
dir="/data/vcea/garrett.wilson/dal"
program="dal.py"
modelFolder="kamiak-models"
logFolder="kamiak-logs"
compressedDataset="al.zip"
dataset=("datasets/al"*)
#compressedDataset="simple.zip"
#dataset=("datasets/simple"*)
#compressedDataset="simple2.zip"
#dataset=("datasets/simple2"*)

# Connecting to the remote server
# Note: this is rsync, so make sure all paths have a trailing slash
remotedir="$dir/"
remotessh="kamiak"
localdir="/home/garrett/Documents/School/19_Spring/Research/deep-activity-learning/"
