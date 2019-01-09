"""
Load data

Functions to load the data into TensorFlow
"""
import os
import math
import h5py
import random
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf

# Not-so-pretty code to feed data to TensorFlow.
class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created.
    https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0"""
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iter_init_func = None

    def after_create_session(self, sess, coord):
        """Initialize the iterator after the session has been created."""
        self.iter_init_func(sess)

def _get_input_fn(features, labels, batch_size, evaluation=False, buffer_size=5000,
    eval_shuffle_seed=0):
    iter_init_hook = IteratorInitializerHook()

    def input_fn():
        # Input images using placeholders to reduce memory usage
        features_placeholder = tf.placeholder(features.dtype, features.shape)
        labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))

        if evaluation:
            dataset = dataset.shuffle(buffer_size, seed=eval_shuffle_seed).batch(batch_size)
        else:
            dataset = dataset.repeat().shuffle(buffer_size).batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        next_data_batch, next_label_batch = iterator.get_next()

        # Need to initialize iterator after creating a session in the estimator
        iter_init_hook.iter_init_func = lambda sess: sess.run(iterator.initializer,
                feed_dict={features_placeholder: features, labels_placeholder: labels})

        return next_data_batch, next_label_batch
    return input_fn, iter_init_hook

def one_hot(x, y, num_classes, index_one=False):
    """
    We want x to be floating point and of dimension [time_steps,num_features]
    where num_features is at least 1. If only a 1D array, then expand dimensions
    to make it [time_steps, 1].

    Also, we want y to be one-hot encoded. Though, note that for the UCR datasets
    (and my synthetic ones that I used the UCR dataset format for), it's indexed
    by 1 not 0, so we subtract one from the index. But, for most other datasets,
    it's 0-indexed.

    If np.squeeze(y) is already 2D (i.e. second dimension has more than 1 class),
    we'll skip one-hot encoding, assuming that it already is. Then we just convert
    to float32.
    """
    # Floating point
    x = x.astype(np.float32)

    # For if we only have one feature,
    # [batch_size, time_steps] --> [batch_size, time_steps, 1]
    if len(x.shape) < 3:
        x = np.expand_dims(x, axis=2)

    # One-hot encoded if not already 2D
    squeezed = np.squeeze(y)
    if len(squeezed.shape) < 2:
        if index_one:
            y = np.eye(num_classes, dtype=np.float32)[squeezed.astype(np.int32) - 1]
        else:
            y = np.eye(num_classes, dtype=np.float32)[squeezed.astype(np.int32)]
    else:
        y = y.astype(np.float32)
        assert squeezed.shape[1] == num_classes, "y.shape[1] != num_classes"

    return x, y

def tf_domain_labels(label, batch_size):
    """ Generate one-hot encoded labels for which domain data is from (using TensorFlow) """
    return tf.tile(tf.one_hot([0], depth=2), [batch_size,1])

def domain_labels(label, batch_size):
    """ Generate one-hot encoded labels for which domain data is from (using numpy) """
    return np.tile(np.eye(2)[label], [batch_size,1])

def shuffle_together(a, b, seed=None):
    """ Shuffle two lists in unison https://stackoverflow.com/a/13343383/2698494 """
    assert len(a) == len(b), "a and b must be the same length"
    rand = random.Random(seed)
    combined = list(zip(a, b))
    rand.shuffle(combined)
    return zip(*combined)

def shuffle_together_np(a, b, seed=None):
    """ Shuffle two numpy arrays together https://stackoverflow.com/a/4602224/2698494"""
    assert len(a) == len(b), "a and b must be the same length"
    rand = np.random.RandomState(seed)
    p = rand.permutation(len(a))
    return a[p], b[p]

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

# def load_hdf5_all(filename):
#     """
#     Load x,y data from hdf5 file -- all data, not cross validation folds
#     """
#     d = h5py.File(filename, "r")
#     features = np.array(d["features"])
#     labels = np.array(d["labels"])
#     return features, labels

def load_single_fold(filename, fold=None):
    """ Get one fold from .hdf5 dataset file """
    data = load_hdf5(filename)

    # Default to the last fold (if time-series data, this will be the one with
    # the most training data)
    if fold is None:
        fold = list(data.keys())[-1]

    train_data = np.array(data[fold]["features_train"])
    train_labels = np.array(data[fold]["labels_train"])
    test_data = np.array(data[fold]["features_test"])
    test_labels = np.array(data[fold]["labels_test"])

    return train_data, train_labels, \
        test_data, test_labels

def load_data_home(feature_set="simple", subdir="hh", A="half1", B="hh117"):
    """
    Load hh/half1.hdf5 as domain A and hh/half2.hdf5 as domain B, use the last
    fold for now -- TODO cross validation
    """
    train_data_a, train_labels_a, \
    test_data_a, test_labels_a = \
        load_single_fold(os.path.join("datasets", feature_set+"_"+A+".hdf5"))

    train_data_b, train_labels_b, \
    test_data_b, test_labels_b = \
        load_single_fold(os.path.join("datasets", feature_set+"_"+B+".hdf5"))

    return train_data_a, train_labels_a, \
        test_data_a, test_labels_a, \
        train_data_b, train_labels_b, \
        test_data_b, test_labels_b

class Config:
    """
    Load an al.config file to get the label and feature names

    Provides functions for converting from name -> int and int -> name:
        label_to_int(name)
        feature_to_int(name)
        int_to_label(index)
        int_to_feature(index)
    """
    def __init__(self, filename="preprocessing/al-features/al.config"):
        """ Gets the possible features and labels """
        self.features = None
        self.labels = None

        with open(filename, 'r') as f:
            for line in f:
                items = line.strip().split(' ')

                if len(items) > 0:
                    if items[0] == "sensors":
                        self.features = items[1:]
                    elif items[0] == "activities":
                        self.labels = items[1:]

        assert self.features is not None, "no \"sensors\" in al.config"
        assert self.labels is not None, "no \"activities\" in al.config"

    def label_to_int(self, label_name):
        """ e.g. Bathe to 0 """
        return self.labels.index(label_name)

    def feature_to_int(self, feature_name):
        """ e.g. Bathroom to 0 """
        return self.features.index(feature_name)

    def int_to_label(self, label_index):
        """ e.g. Bathe to 0 """
        return self.labels[label_index]

    def int_to_feature(self, feature_index):
        """ e.g. Bathroom to 0 """
        return self.features[feature_index]
