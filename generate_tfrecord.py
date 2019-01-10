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

from load_data import one_hot, load_hdf5, ALConfig

def create_tf_example(x, y):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'x': tf.train.Feature(float_list=tf.train.FloatList(value=x.reshape(-1))),
        'y': tf.train.Feature(float_list=tf.train.FloatList(value=y)),
    }))
    return tf_example

def write_tfrecord(filename, x, y):
    """ Output to TF record file """
    assert len(x) == len(y)
    print("Saving", filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for i in range(len(x)):
            tf_example = create_tf_example(x[i], y[i])
            writer.write(tf_example.SerializeToString())

def write_tfrecord_config(filename, num_features, num_classes, time_steps, x_dims):
    print("Saving", filename)
    with open(filename, 'w') as f:
        f.write("num_features "+str(num_features)+"\n")
        f.write("num_classes "+str(num_classes)+"\n")
        f.write("time_steps "+str(time_steps)+"\n")
        f.write("x_dims "+" ".join([str(x) for x in x_dims])+"\n")

def generate_tfrecords(feature_set, inputs="preprocessing/windows",
        outputs="datasets"):
    # Get number of classes -- should be the same for all datasets
    config = ALConfig()
    num_classes = len(config.labels)
    num_features = None
    time_steps = None
    prev_x_dims = None

    # Get list of all the datasets
    files = pathlib.Path(inputs).glob(feature_set+"_*.hdf5")
    paths = [(x.stem, str(x)) for x in files]

    # For each file
    for name, f in paths:
        data = load_hdf5(f)

        # For each fold
        for fold in data.keys():
            train_data = np.array(data[fold]["features_train"])
            train_labels = np.array(data[fold]["labels_train"])
            test_data = np.array(data[fold]["features_test"])
            test_labels = np.array(data[fold]["labels_test"])

            # Information about dataset
            num_features = train_data.shape[2]
            time_steps = train_data.shape[1]
            x_dims = [time_steps, num_features]
            assert prev_x_dims is None or x_dims == prev_x_dims, \
                "time_steps and num_features must be the same for all X: " \
                +str(x_dims)+" vs. "+str(prev_x_dims)
            prev_x_dims = copy.deepcopy(x_dims)

            # One-hot encoding
            index_one = False # Labels start from 0
            train_data, train_labels = one_hot(train_data, train_labels, num_classes, index_one)
            test_data, test_labels = one_hot(train_data, train_labels, num_classes, index_one)

            # Write to .tfrecord file
            train_filename = os.path.join(outputs, name+"_train_"+fold+".tfrecord")
            test_filename = os.path.join(outputs, name+"_test_"+fold+".tfrecord")
            write_tfrecord(train_filename, train_data, train_labels)
            write_tfrecord(test_filename, test_data, test_labels)

    # Write out information about dimensions since tfrecord can only store
    # 1D arrays
    write_tfrecord_config(os.path.join(outputs, feature_set+".config"),
        num_features, num_classes, time_steps, x_dims)

if __name__ == "__main__":
    generate_tfrecords("simple")
    generate_tfrecords("al")
