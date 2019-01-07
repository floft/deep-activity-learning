#!/usr/bin/env python3
"""
Random forest activity learning (the baseline)
"""
import h5py
import pathlib
import numpy as np

from sklearn.ensemble import RandomForestClassifier

def load_hdf5_full(filename):
    """
    Load x,y data from hdf5 file -- create dictionary of all data in file
    """
    d = {
            "features_train": [],
            "features_test": [],
            "labels_train": [],
            "labels_test": [],
        }
    data = h5py.File(filename, "r")

    for fold in data.keys():
        d["features_train"].append(np.array(data[fold]["features_train"]))
        d["features_test"].append(np.array(data[fold]["features_test"]))
        d["labels_train"].append(np.array(data[fold]["labels_train"]))
        d["labels_test"].append(np.array(data[fold]["labels_test"]))

    return d

def load_hdf5(filename):
    """
    Load x,y data from hdf5 file -- don't load everything, better for large files
    """
    return h5py.File(filename, "r")

def compute_accuracy(y_hat, y):
    # TODO maybe exclude Other_Activity for now?
    # TODO maybe move all .py into root dir so I can share code?
    assert len(y_hat) == len(y), "y must be same length as y"
    return sum(y_hat == y)/len(y)

def predict_random_forest(data):
    results = []

    for fold in data.keys():
        # Load only data we need to save memory
        x_train = np.array(data[fold]["features_train"])
        y_train = np.array(data[fold]["labels_train"])

        # Flatten time/feature dims of x since RF doesn't work on multi-dim data
        x_train = x_train.reshape([x_train.shape[0], -1])

        rf = RandomForestClassifier(
            n_estimators=60,
            max_features=8,
            bootstrap=True,
            criterion="entropy",
            min_samples_split=20,
            max_depth=None,
            n_jobs=8,
            class_weight='balanced')

        # Train
        rf.fit(x_train, y_train)

        # Read this data later from the file to save memory
        x_test = np.array(data[fold]["features_test"])
        y_test = np.array(data[fold]["labels_test"])

        # Flatten time/feature dims of x since RF doesn't work on multi-dim data
        x_test = x_test.reshape([x_test.shape[0], -1])

        # Test
        y_predicted = rf.predict(x_test)

        # Accuracy on test data
        results.append(compute_accuracy(y_predicted, y_test))

    return np.array(results)

if __name__ == "__main__":
    for feature_set in ["al", "simple"]:
        print("Feature set:", feature_set)

        for f in pathlib.Path("datasets").glob(feature_set+"_*.hdf5"):
            data = load_hdf5(f)
            results = predict_random_forest(data)
            print(str(f)+":", results, "Avg:", np.mean(results), "Std:", np.std(results))

        print()
