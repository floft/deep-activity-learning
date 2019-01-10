#!/usr/bin/env python3
"""
Generate the cross validation folds on the various files
"""
import os
import sys
import h5py
import pathlib
import numpy as np

from sklearn.model_selection import TimeSeriesSplit

def load_hdf5(filename):
    """
    Load x,y data from hdf5 file
    """
    d = h5py.File(filename, "r")
    features = np.array(d["features"])
    labels = np.array(d["labels"])
    return features, labels

def create_windows(x, y, window_size, overlap=True):
    """
    Concatenate along dim-1 to meet the desired window_size. We'll skip any
    windows that reach beyond the end.

    Two options (examples for window_size=5):
        Overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
            label of example 4; and window 1 will be 1,2,3,4,5 and the label of
            example 5
        No overlap - e.g. window 0 will be a list of examples 0,1,2,3,4 and the
            label of example 4; and window 1 will be 5,6,7,8,9 and the label of
            example 9
    """
    windows_x = []
    windows_y = []
    i = 0

    while i < len(y)-window_size:
        # Make it (1,window_size,# features)
        window_x = np.expand_dims(np.concatenate(x[i:i+window_size], axis=0), axis=0)
        window_y = y[i+window_size-1]

        windows_x.append(window_x)
        windows_y.append(window_y)

        # Where to start the next window
        if overlap:
            i += 1
        else:
            i += window_size

    windows_x = np.vstack(windows_x)
    windows_y = np.hstack(windows_y)

    return windows_x, windows_y

def process_data(x, y, window_size, overlap):
    # Expand dimensions to be (# examples, 1, # features)
    x = np.expand_dims(x, axis=1).astype(np.float32)

    # Above we expanded to be window_size=1, so if that's the case, we're
    # already done
    if window_size != 1:
        x, y = create_windows(x, y, window_size, overlap)

    return x, y

def cross_validation_indices(folds, x):
    # Indices for each cross validation fold -- must recalculate since each
    # file is a different size
    tscv = TimeSeriesSplit(n_splits=folds)

    train_indices = []
    test_indices = []

    # Train/test split before calculating windows since if overlap=True, we
    # still don't want overlap between the train/test splits -- only within
    # the train set and within the test set
    for train, test in tscv.split(x):
        train_indices.append(train)
        test_indices.append(test)

    assert len(train_indices) == len(test_indices), \
        "Number of train/test folds must match"

    return train_indices, test_indices

def create_dataset(
        dir_name,
        prefix,
        window_size,
        subdir="hh",
        seed=0,
        overlap=True,
        folds=3,
        files=None):

    # If not specified, use all the files
    if files is None:
        files = pathlib.Path(os.path.join(dir_name, subdir)).glob("*.hdf5")
        paths = [(x.stem, str(x)) for x in files]
    else:
        # e.g. files should be ["hh101", "hh102", ...]
        paths = [(x, os.path.join(dir_name, subdir) + "/" + x + ".hdf5") for x in files]

    # For each dataset we want to create
    for name, f in paths:
        x, y = load_hdf5(f)
        assert len(x) == len(y), "Must have label for each feature vector"

        # Output to file
        out = h5py.File(prefix+"_"+name+".hdf5", "w")

        #
        # All data -- no cross validation when doing domain adaptation, it's
        # instead leave-one-out cross validation, where we train on multiple
        # homes' data and then test on a different one
        #
        # NOTE actually we still need cross validation (and thus don't need this)
        # since we want an estimate of both
        #   1) on houses with labeled data, how well would AL perform after
        #      we stop training on labeling data
        #   2) on houses with only unlabeled data, how well would AL perform after
        #      we stop the adaptation process (training with domain adaptation)
        #
        #x_all, y_all = process_data(x, y, window_size, overlap)
        #out.create_dataset("features", data=x_all, compression="gzip")
        #out.create_dataset("labels", data=y_all, compression="gzip")

        #
        # Split data into time-series folds with scikit-learn
        # (see https://scikit-learn.org/stable/modules/cross_validation.html#timeseries-cv)
        #
        train_indices, test_indices = cross_validation_indices(folds, x)

        for fold in range(len(train_indices)):
            # Do train and test sets separate to reduce memory usage
            train = train_indices[fold]
            x_train, y_train = process_data(x[train], y[train], window_size, overlap)
            out.create_dataset(str(fold)+"/features_train", data=x_train, compression="gzip")
            out.create_dataset(str(fold)+"/labels_train", data=y_train, compression="gzip")

            test = test_indices[fold]
            x_test, y_test = process_data(x[test], y[test], window_size, overlap)
            out.create_dataset(str(fold)+"/features_test", data=x_test, compression="gzip")
            out.create_dataset(str(fold)+"/labels_test", data=y_test, compression="gzip")

if __name__ == "__main__":
    create_dataset("../al-features", "al",
        window_size=1) # each uses window of size 30
    create_dataset("../simple-features", "simple",
        window_size=30)
