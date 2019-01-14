#!/usr/bin/env python3
"""
Create the .tfrecord files for domain adaptation training/testing

When doing the training directly on loaded numpy arrays, we run out of memory.
Thus, we'll save as .tfrecord files which TensorFlow doesn't need to load into
memory all at once thus allowing us to process larger datasets.
"""
import os
import copy
import pathlib
import numpy as np
import tensorflow as tf

from pool import run_job_pool
from load_data import one_hot, load_hdf5, ALConfig, shuffle_together_calc

def create_tf_example(x, y):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'x': tf.train.Feature(float_list=tf.train.FloatList(value=x.flatten())),
        'y': tf.train.Feature(float_list=tf.train.FloatList(value=y)),
    }))
    return tf_example

def write_tfrecord(filename, x, y):
    """ Output to TF record file """
    assert len(x) == len(y)
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    with tf.python_io.TFRecordWriter(filename, options=options) as writer:
        for i in range(len(x)):
            tf_example = create_tf_example(x[i], y[i])
            writer.write(tf_example.SerializeToString())

def write_tfrecord_config(filename, num_features, num_classes, time_steps, x_dims):
    with open(filename, 'w') as f:
        f.write("num_features "+str(num_features)+"\n")
        f.write("num_classes "+str(num_classes)+"\n")
        f.write("time_steps "+str(time_steps)+"\n")
        f.write("x_dims "+" ".join([str(x) for x in x_dims])+"\n")

def process_fold(filename, fold, name, num_classes, outputs, seed):
    data = load_hdf5(filename)
    index_one = False # Labels start from 0

    #
    # Train data
    #
    train_data = np.array(data[fold]["features_train"])
    train_labels = np.array(data[fold]["labels_train"])

    train_data, train_labels = one_hot(train_data, train_labels, num_classes, index_one)

    p = shuffle_together_calc(len(train_labels), seed=seed)
    train_data = train_data[p]
    train_labels = train_labels[p]

    train_filename = os.path.join(outputs, name+"_train_"+fold+".tfrecord")
    write_tfrecord(train_filename, train_data, train_labels)

    del p, train_data, train_labels

    #
    # Test data
    #
    test_data = np.array(data[fold]["features_test"])
    test_labels = np.array(data[fold]["labels_test"])

    test_data, test_labels = one_hot(test_data, test_labels, num_classes, index_one)

    p = shuffle_together_calc(len(test_labels), seed=seed+1)
    test_data = test_data[p]
    test_labels = test_labels[p]

    test_filename = os.path.join(outputs, name+"_test_"+fold+".tfrecord")
    write_tfrecord(test_filename, test_data, test_labels)

    del p, test_data, test_labels, data

def generate_config(feature_set, inputs="preprocessing/windows",
        outputs="datasets", fold=0):

    if not os.path.exists(outputs):
        os.makedirs(outputs)

    # Get number of classes -- should be the same for all datasets
    config = ALConfig()
    num_classes = len(config.labels)

    # Get list of all the datasets -- get the first one
    files = list(pathlib.Path(inputs).glob(feature_set+"_*.hdf5"))
    assert len(files) > 0, "found no files with feature set "+feature_set
    data = load_hdf5(str(files[0]))

    # Load dimensions from this one file (assume the rest are the same)
    train_data = np.array(data[str(fold)]["features_train"])

    # Information about dataset
    num_features = train_data.shape[2]
    time_steps = train_data.shape[1]
    x_dims = [time_steps, num_features]

    # Write out information about dimensions since tfrecord can only store
    # 1D arrays
    write_tfrecord_config(os.path.join(outputs, feature_set+".config"),
        num_features, num_classes, time_steps, x_dims)

def generate_tfrecords(inputs="preprocessing/windows",
        outputs="datasets", prefix="*"):
    # Get number of classes
    config = ALConfig()
    num_classes = len(config.labels)

    # Get list of all the datasets
    files = pathlib.Path(inputs).glob(prefix+"_*.hdf5")
    paths = [(x.stem, str(x)) for x in files]

    # Get all files and folds
    commands = []
    seed = 0

    for name, f in paths:
        data = load_hdf5(f)

        for fold in data.keys():
            commands.append((f, fold, name, num_classes, outputs, seed))
            seed += 2

    # Process them all
    run_job_pool(process_fold, commands)

if __name__ == "__main__":
    generate_config("al")
    generate_config("simple")
    generate_tfrecords()
    #generate_tfrecords(prefix="simple")
