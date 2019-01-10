#!/usr/bin/env python3
"""
Random forest activity learning (the baseline)
"""
import h5py
import pathlib
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from load_data import load_hdf5, load_data_home_da

def compute_accuracy(y_hat, y):
    assert len(y_hat) == len(y), "y must be same length as y"
    return sum(y_hat == y)/len(y)

def train_rf(x, y):
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
    rf.fit(x, y)

    return rf

def predict_random_forest(data):
    results = []

    for fold in data.keys():
        # Load only data we need to save memory
        x_train = np.array(data[fold]["features_train"])
        y_train = np.array(data[fold]["labels_train"])

        # Flatten time/feature dims of x since RF doesn't work on multi-dim data
        x_train = x_train.reshape([x_train.shape[0], -1])

        # Train Random Forest
        rf = train_rf(x_train, y_train)

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
    #
    # Testing individual dataset files
    #
    # for feature_set in ["al", "simple"]:
    #     print("Feature set:", feature_set)

    #     for f in pathlib.Path("datasets").glob(feature_set+"_*.hdf5"):
    #         data = load_hdf5(f)
    #         results = predict_random_forest(data)
    #         print(str(f)+":", results, "Avg:", np.mean(results), "Std:", np.std(results))

    #     print()

    #
    # Domain adaptation testing
    #
    folds = 3
    feature_set = "al"
    targets = ["hh101", "hh102", "hh103"]

    for target in targets:
        train_results_a = []
        test_results_a = []
        train_results_b = []
        test_results_b = []

        for fold in range(folds):
            train_data_a, train_labels_a, \
            test_data_a, test_labels_a, \
            train_data_b, train_labels_b, \
            test_data_b, test_labels_b = \
                load_data_home_da(fold, target, feature_set)

            # Reshape since RF doesn't handle higher dimensional data
            train_data_a = train_data_a.reshape([train_data_a.shape[0], -1])
            test_data_a = test_data_a.reshape([test_data_a.shape[0], -1])
            train_data_b = train_data_b.reshape([train_data_b.shape[0], -1])
            test_data_b = test_data_b.reshape([test_data_b.shape[0], -1])

            # Train
            rf = train_rf(train_data_a, train_labels_a)

            # Test on train/test and A/B
            train_results_a.append(compute_accuracy(rf.predict(train_data_a), train_labels_a))
            test_results_a.append(compute_accuracy(rf.predict(test_data_a), test_labels_a))
            train_results_b.append(compute_accuracy(rf.predict(train_data_b), train_labels_b))
            test_results_b.append(compute_accuracy(rf.predict(test_data_b), test_labels_b))

        train_results_a = np.array(train_results_a)
        test_results_a = np.array(test_results_a)
        train_results_b = np.array(train_results_b)
        test_results_b = np.array(test_results_b)

        print(target)
        print("  Train A: ", train_results_a,
            "Avg:", np.mean(train_results_a),
            "Std:", np.std(train_results_a))
        print("  Test A: ", test_results_a,
            "Avg:", np.mean(test_results_a),
            "Std:", np.std(test_results_a))
        print("  Train B: ", train_results_b,
            "Avg:", np.mean(train_results_b),
            "Std:", np.std(train_results_b))
        print("  Test B: ", test_results_b,
            "Avg:", np.mean(test_results_b),
            "Std:", np.std(test_results_b))
        print()
