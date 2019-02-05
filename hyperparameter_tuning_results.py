#!/usr/bin/env python3
"""
Analyze the results from the hyperparameter tuning
"""
import pathlib
import numpy as np
import pandas as pd

def get_tuning_files(dir_name, prefix="dal_results_cv-"):
    """ Get all the hyperparameter evaluation result files """
    files = []
    matching = pathlib.Path(dir_name).glob(prefix+"*.txt")

    for m in matching:
        name = m.stem.replace(prefix, "")
        file = str(m)
        files.append((name, file))

    return files

def beginning_match(match, line):
    """ Does the first x=len(match) chars of line match the match string """
    return line[:len(match)] == match

def parse_file(filename):
    """
    Get all of the data from the file

    Several parts:
        - Best validation accuracy per target/fold at a particular step
        - target/fold train/test A/B accuracy
        - averages of train/test A/B accuracies
    """
    in_validation = False
    in_traintest = False
    in_averages = False

    validation = []
    traintest = []
    averages = []

    valid_header = "Target,Fold,Model,Best Step,Accuracy at Step"
    traintest_header = "Target,Fold,Train A,Test A,Train B,Test B"
    averages_header = "Dataset,Avg,Std"

    with open(filename) as f:
        for line in f:
            l = line.strip()

            if beginning_match(valid_header, l):
                in_validation = True
                in_traintest = False
                in_averages = False
            elif beginning_match(traintest_header, l):
                in_validation = False
                in_traintest = True
                in_averages = False
            elif beginning_match(averages_header, l):
                in_validation = False
                in_traintest = False
                in_averages = True
            elif len(l) > 0:
                values = l.split(",")

                if in_validation:
                    validation.append((values[0], int(values[1]),
                        values[2], int(values[3]), float(values[4])))
                elif in_traintest:
                    traintest.append((values[0], int(values[1]),
                        float(values[2]), float(values[3]), float(values[4]), float(values[5])))
                elif in_averages:
                    averages.append((values[0], float(values[1]), float(values[2])))
            else:
                # Empty lines ends a section
                in_validation = False
                in_traintest = False
                in_averages = False

    validation = pd.DataFrame(data=validation, columns=valid_header.split(","))
    traintest = pd.DataFrame(data=traintest, columns=traintest_header.split(","))
    averages = pd.DataFrame(data=averages, columns=averages_header.split(","))

    return validation, traintest, averages

def compute_mean_std(df, name):
    # ddof=0 is the numpy default, ddof=1 is Pandas' default
    return df[name].mean(), df[name].std(ddof=0)

def compute_val_stats(df):
    return compute_mean_std(df, "Accuracy at Step")

def compute_eval_stats(df):
    names = ["Train A", "Test A", "Train B", "Test B"]
    data = [[name]+list(compute_mean_std(df, name)) for name in names]
    return pd.DataFrame(data=data, columns=["Dataset", "Avg", "Std"])

def all_stats(files, recompute_averages=False, sort_on_test=False, sort_on_b=False):
    stats = []

    for name, file in files:
        validation, traintest, averages = parse_file(file)

        if recompute_averages:
            averages = compute_eval_stats(traintest)

        validavg = compute_val_stats(validation)

        stats.append({
            "name": name,
            "file": file,
            "validation": validation,
            "traintest": traintest,
            "averages": averages,
            "validavg": validavg,
        })

    if sort_on_test:
        # Sort by test accuracy (i.e. cheating)
        stats.sort(key=lambda x: x["averages"][x["averages"]["Dataset"] == "Test A"]["Avg"].values[0])
    elif sort_on_b:
        # Sort by test accuracy on domain B (i.e. cheating)
        stats.sort(key=lambda x: x["averages"][x["averages"]["Dataset"] == "Test B"]["Avg"].values[0])
    else:
        # Sort by validation accuracy
        stats.sort(key=lambda x: x["validavg"][0])

    return stats

if __name__ == "__main__":
    files = get_tuning_files(".")

    # Best on validation data
    best = all_stats(files)[-1]
    print("Best on Validation -", best["name"])
    print(best["averages"])
    print()

    # Best on testing data (i.e. cheating)
    best = all_stats(files, sort_on_test=True)[-1]
    print("Best on Test A (cheating) -", best["name"])
    print(best["averages"])
    print()

    # Best on testing data on domain B (i.e. cheating)
    best = all_stats(files, sort_on_b=True)[-1]
    print("Best on Test B (cheating) -", best["name"])
    print(best["averages"])
