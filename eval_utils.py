"""
Functions for handling logs/models to help with evaluation
"""
import os
import re
import sys
import pathlib
import tensorflow as tf

def last_modified_number(dir_name, glob):
    """
    Looks in dir_name at all files matching glob and takes number
    from the one last modified
    """
    files = pathlib.Path(dir_name).glob(glob)
    files = sorted(files, key=lambda cp:cp.stat().st_mtime)

    if len(files) > 0:
        # Get number from filename
        regex = re.compile(r'\d+')
        numbers = [int(x) for x in regex.findall(str(files[-1]))]
        assert len(numbers) == 1, "Could not determine number from last modified file"
        last = numbers[0]

        return last

    return None

def last_modified(dir_name, glob):
    """
    Looks in dir_name at all files matching glob and returns the file last
    modified
    """
    files = pathlib.Path(dir_name).glob(glob)
    files = sorted(files, key=lambda cp:cp.stat().st_mtime)

    if len(files) > 0:
        return str(files[-1])

    return None

def get_step_from_log(log_dir, last, tag='accuracy_task/source/validation'):
    # Open the log file generated during training, find the step with the
    # highest validation accuracy
    logfile = last_modified(log_dir, "*.tfevents*")

    # If the log file doesn't exist
    if logfile is None:
        return None, None

    task_accuracy = []
    try:
        for e in tf.train.summary_iterator(logfile):
            for v in e.summary.value:
                if v.tag == tag:
                    task_accuracy.append((e.step, v.simple_value))
    except tf.errors.DataLossError:
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

    assert len(task_accuracy) > 0, \
        "task_accuracy empty for log"+logfile+": "+str(task_accuracy)

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

def get_files_to_keep(log_dir):
    best, _ = get_step_from_log(log_dir, last=False)
    last, _ = get_step_from_log(log_dir, last=True)

    return best, last

def delete_models_except(model_dir, best, last):
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
