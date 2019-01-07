#!/usr/bin/env python3
"""
Generate the cross validation folds on the various datasets
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

def create_dataset(
        dir_name,
        subdir="hh",
        seed=0,
        window_size=100,
        overlap=False,
        folds=3,
        files=["hh101", "hh102", "hh103", "hh104", "hh105"]):
    """
    First dataset to try
    """
    paths = [os.path.join(dir_name, subdir) + "/" + x + ".hdf5" for x in files]
    xs = []
    ys = []

    for f in paths:
        x, y = load_hdf5(f)
        assert len(x) == len(y), "Must have label for each feature vector"

        xs.append(x)
        ys.append(y)

    # Expand dimensions to be (# examples, 1, # features)
    for i in range(len(xs)):
        xs[i] = np.expand_dims(xs[i], axis=1).astype(np.float32)

    # Above we expanded to be window_size=1, so if that's the case, we're
    # already done
    if window_size != 1:
        for i in range(len(xs)):
            xs[i], ys[i] = create_windows(xs[i], ys[i], window_size, overlap)

    # Gives, e.g. dataset = {
    #   "hh101": {
    #       "features_train": [ fold 0 x, fold 1 x, fold 2 x ],
    #       "features_test":  [ fold 0 x, fold 1 x, fold 2 x ],
    #       "labels_train":   [ fold 0 y, fold 1 y, fold 2 y ],
    #       "labels_test":    [ fold 0 y, fold 1 y, fold 2 y ],
    #   },
    #   "hh102": { ... }, ...
    # }
    dataset = {}

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        d = {
            "features_train": [],
            "features_test": [],
            "labels_train": [],
            "labels_test": [],
        }

        # Indices for each cross validation fold -- must recalculate since each
        # file is a different size
        tscv = TimeSeriesSplit(n_splits=folds)

        train_indices = []
        test_indices = []

        for train, test in tscv.split(x):
            train_indices.append(train)
            test_indices.append(test)

        assert len(train_indices) == len(test_indices), "Number of train/test folds must match"

        for j in range(len(train_indices)):
            train = train_indices[j]
            test = test_indices[j]

            d["features_train"].append(x[train])
            d["features_test"].append(x[test])
            d["labels_train"].append(y[train])
            d["labels_test"].append(y[test])

        dataset[files[i]] = d

    return dataset

def write_dataset(datasets, prefix, folds=3):
    """
    Write out the data to a .hdf5 file

    Format: {0,1,2}/{features,labels}_{train,test}
    Example:
        f=h5py.File("simple_hh101.hdf5")
        np.array(f.get("0/features_train")) # train x for fold 0
    """
    for name, dataset in datasets.items():
        out = h5py.File(prefix+"_"+name+".hdf5", "w")

        for k, d in dataset.items():
            for fold in range(folds):
                out.create_dataset(str(fold)+"/"+k, data=np.array(d[fold]), compression="gzip")

if __name__ == "__main__":
    al = create_dataset("../preprocessing/al-features")
    write_dataset(al, "al")

    simple = create_dataset("../preprocessing/simple-features")
    write_dataset(simple, "simple")
