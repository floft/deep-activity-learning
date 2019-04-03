# Deep Activity Learning (DAL)

Goals:

- Compare random forest activity learning (AL) with deep activity learning
- Compare previous AL features with simpler feature vector
- Add domain adaptation and generalization to deep network for further
  improvement

Steps:

- Preprocess data extracting the desired features, creating time-series windows,
  and cross validation train/test splits (see *generate_datasets.sh*,
  *preprocessing/*, etc.)
- Run and compare AL (*al.py*) and DAL (*dal.py*) on the datasets

## Datasets

This is designed to work on smart home datasets in the formats of those on the
CASAS website. To download some smart home data to *preprocessing/orig*, convert
into the appropriate annotated format, and output to *preprocessing/raw*, run:

    ./download_datasets.sh

## Preprocessing

To apply activity label and sensor translations, generate the desired feature
representations and time-series windows, and create the .tfrecord files:

    ./generate_datasets.sh

**Note:** a lot of the Bash scripts use my multi-threading/processing script
[/scripts/threading](https://floft.net/code/bash-threading/) to drastically
speed up the preprocessing, so you'll want to either remove those statements or
download the script.

## Running

### AL

To train AL (uses random forests) and compute the results:

    ./al_results.sh

### DAL

If running locally on your computer, run the cross validation training:

    ./dal_cv.sh

If running on a cluster (after editing *kamiak_config.sh*):

    ./kamiak_upload.sh
    ./kamiak_queue_all.sh flat --dataset=al.zip --features=al --model=flat
    ./kamiak_queue_all.sh flat-da --dataset=al.zip --features=al --model=flat --adapt

    # on your computer
    ./kamiak_tflogs.sh # during training, to download the logs/models

Then, to pick the best models based on the validation results above (unless
using domain adaptation, then pick the last model) and evaluate on the entire
train and test sets for comparison with AL:

    ./dal_results.sh flat --features=al # set "from" to either kamiak or cv
    ./dal_results.sh flat-da --features=al --last
