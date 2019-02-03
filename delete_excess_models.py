#!/usr/bin/env python3
"""
Delete all models that aren't either:
    1) the last one (used for continuing training or if using adaptation)
    2) the one with the highest validation accuracy
"""
import os
import sys
import pathlib

from dal_eval import get_step_from_log, get_checkpoint

def get_files_to_keep(log_dir):
    best, _ = get_step_from_log(log_dir, last=False)
    last, _ = get_step_from_log(log_dir, last=True)

    return best, last

def get_model_dir(log_dir):
    model_dir = log_dir.replace("logs", "models")

    assert os.path.exists(model_dir), \
        "Could not find model_dir by replacing \"logs\" with \"models\": "+log_dir

    return model_dir

def delete_models_except(model_dir, best, last):
    files = pathlib.Path(model_dir).glob("*")

    keep = [
        "checkpoint",
        "graph.pbtxt",
        "events.out",
        "model.ckpt-"+str(best),
        "model.ckpt-"+str(last),
        # Make sure we don't get rid of the +/- ones if we're off by one
        # with the validation accuracy
        "model.ckpt-"+str(best-1),
        "model.ckpt-"+str(last-1),
        "model.ckpt-"+str(best+1),
        "model.ckpt-"+str(last+1),
    ]

    for f in files:
        # Skip deleting this file if one of the keep strings is in the name
        found = False

        for k in keep:
            if k in str(f.name):
                found = True
                break

        # Otherwise, delete it
        if not found:
            print("Deleting", f)
            os.remove(str(f))

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
