#!/usr/bin/env python3
"""
Delete all models that aren't either:
    1) the last one (used for continuing training or if using adaptation)
    2) the one with the highest validation accuracy
"""
import os
import sys
import pathlib

from eval_utils import get_files_to_keep, delete_models_except

def get_model_dir(log_dir):
    model_dir = log_dir.replace("logs", "models")

    assert os.path.exists(model_dir), \
        "Could not find model_dir by replacing \"logs\" with \"models\": "+log_dir

    return model_dir

if __name__ == "__main__":
    logs = sys.argv[1:]

    # Usage
    if len(logs) == 0:
        print("Usage: ./delete_excess_models.py logdir1/* logdir2/* ...")
        sys.exit(1)

    # Delete all but the last and best models
    for log_dir in logs:
        best, last = get_files_to_keep(log_dir)
        model_dir = get_model_dir(log_dir)
        delete_models_except(model_dir, best, last)
