"""
Functions for handling logs/models to help with evaluation
"""
import os
import sys
import pathlib
import tensorflow as tf

from file_utils import last_modified, get_best_valid_accuracy, write_best_valid_accuracy

def get_step_from_log(log_dir, last, tag='accuracy_task/source/validation',
        warn=True):
    """
    Get the highest accuracy and the step of that highest accuracy from the
    latest log file in the specified log_dir. If last==True, then get the last
    accuracy rather than the highest.

    Optionally warn if the file is still being written to (i.e. training is
    still running).
    """
    # Open the log file generated during training, find the step with the
    # highest validation accuracy
    logfile = last_modified(log_dir, "*.tfevents*")

    # If the log file doesn't exist
    if logfile is None:
        return None, None

    task_accuracy = []
    try:
        for e in tf.compat.v1.train.summary_iterator(logfile):
            for v in e.summary.value:
                if v.tag == tag:
                    task_accuracy.append((e.step, v.simple_value))
    except tf.errors.DataLossError:
        if warn:
            # Skip DataLossErrors since it's probably since we're still writing to
            # the file (i.e. if we run eval during training).
            print("Warning: DataLossError -- found " + str(len(task_accuracy)) \
                + ", skipping remainder of file", file=sys.stderr)
            sys.stderr.flush()

    # Sort by accuracy -- but only if we didn't choose to use the last model.
    # In that case, the ...[-1] will pick the last one, so all we have to do
    # is not sort this.
    if not last:
        task_accuracy.sort(key=lambda tup: tup[1])

    # If no data in the log file, then ignore it
    if len(task_accuracy) == 0:
        return None, None

    assert len(task_accuracy) > 0, \
        "task_accuracy empty for log "+logfile+": "+str(task_accuracy)

    max_accuracy = task_accuracy[-1][1]
    max_accuracy_step = task_accuracy[-1][0]

    return max_accuracy_step, max_accuracy

def get_checkpoint(model_dir, step):
    """
    Load corresponding checkpoint -- if it doesn't exist, then it probably
    saved the iter-1 as a checkpoint. For example, most evaluations are at
    iterations like 84001 but the checkpoint was at iteration 84000.

    Returns the checkpoint filename and the step we actually loaded (may or
    may not be the one specified, but should be +/- 1)
    """
    ckpt = os.path.join(model_dir, "model.ckpt-"+str(step))

    if not os.path.exists(ckpt+".index"):
        step -= 1
        ckpt = os.path.join(model_dir, "model.ckpt-"+str(step))

        assert os.path.exists(ckpt+".index"), \
            "could not find model checkpoint "+ckpt

    return ckpt, step

def get_files_to_keep(log_dir, warn=True):
    """ Get both the best and last model files to keep """
    best, _ = get_step_from_log(log_dir, last=False, warn=warn)
    last, _ = get_step_from_log(log_dir, last=True, warn=warn)

    return best, last

def delete_models_except(model_dir, best, last):
    """ Delete all the model files except for the specified best and last ones """
    # Skip deleting files if we couldn't find the best/last, i.e. error on the
    # side of not deleting stuff.
    if best is None or last is None:
        return

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

class RemoveOldCheckpoints:
    """
    Remove checkpoints that are not the best or the last one so we don't waste
    tons of disk space
    """
    def __init__(self, log_dir, model_dir):
        self.log_dir = log_dir
        self.model_dir = model_dir

    def before_save(self):
        """
        Do this right before saving, to make sure we don't mistakingly delete
        the one we just saved in after_save. This will in effect keep the last
        two instead of just the last one.
        """
        # Don't warn since we know there will be a DataLossError since we're
        # still training and the file isn't complete yet.
        best, last = get_files_to_keep(self.log_dir, warn=False)
        delete_models_except(self.model_dir, best, last)

    def after_save(self):
        """
        Keep track of the best accuracy we've gotten on the validation data.
        This file is used for automated hyperparameter tuning.

        This is done after saving since possibly the latest checkpoint had the
        highest validation accuracy.
        """
        # Get previous best if available
        previous_best = get_best_valid_accuracy(self.log_dir)

        # Get new best
        _, best_accuracy = get_step_from_log(self.log_dir, last=False, warn=False)

        # Only if we got some accuracy (e.g. log might not exist)
        if best_accuracy is not None:
            # Write if new best is better than previous best
            if previous_best is None or best_accuracy > previous_best:
                write_best_valid_accuracy(self.log_dir, best_accuracy)
