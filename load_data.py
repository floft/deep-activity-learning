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

def _get_input_fn(features, labels, batch_size,
        evaluation=False, buffer_size=200000, eval_shuffle_seed=0):
    """ Load data from numpy arrays (requires more memory but less disk space) """
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

def _get_tfrecord_input_fn(filenames, batch_size, x_dims, num_classes,
        evaluation=False, count=False, buffer_size=200000, eval_shuffle_seed=0):
    """ Load data from .tfrecord files (requires less memory but more disk space) """
    iter_init_hook = IteratorInitializerHook()

    # Create a description of the features
    # See: https://www.tensorflow.org/tutorials/load_data/tf-records
    feature_description = {
        'x': tf.FixedLenFeature(x_dims, tf.float32),
        'y': tf.FixedLenFeature([num_classes], tf.float32),
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        parsed = tf.parse_single_example(example_proto, feature_description)
        return parsed["x"], parsed["y"]

    def input_fn():
        dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
        dataset = dataset.map(_parse_function)

        if count: # only count, so no need to shuffle
            dataset = dataset.batch(batch_size)
        elif evaluation: # don't repeat since we want to evaluate entire set
            dataset = dataset.shuffle(buffer_size, seed=eval_shuffle_seed).batch(batch_size)
        else: # repeat, shuffle, and batch
            dataset = dataset.repeat().shuffle(buffer_size).batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        next_data_batch, next_label_batch = iterator.get_next()

        # Need to initialize iterator after creating a session in the estimator
        iter_init_hook.iter_init_func = lambda sess: sess.run(iterator.initializer)

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
    else:
        fold = str(fold) # even though it's 0,1,2,3... it's actually strings

    train_data = np.array(data[fold]["features_train"])
    train_labels = np.array(data[fold]["labels_train"])
    test_data = np.array(data[fold]["features_test"])
    test_labels = np.array(data[fold]["labels_test"])

    return train_data, train_labels, \
        test_data, test_labels

def load_data_home(feature_set="simple", dir_name="datasets", A="half1", B="hh117"):
    """
    Load hh/half1.hdf5 as domain A and hh/half2.hdf5 as domain B, use the last
    fold for now -- TODO cross validation
    """
    train_data_a, train_labels_a, \
    test_data_a, test_labels_a = \
        load_single_fold(os.path.join(dir_name, feature_set+"_"+A+".hdf5"))

    train_data_b, train_labels_b, \
    test_data_b, test_labels_b = \
        load_single_fold(os.path.join(dir_name, feature_set+"_"+B+".hdf5"))

    return train_data_a, train_labels_a, \
        test_data_a, test_labels_a, \
        train_data_b, train_labels_b, \
        test_data_b, test_labels_b

def load_data_home_da(fold, target, feature_set, dir_name="preprocessing/windows"):
    """
    Concatate all dataset files (of a particular fold) together except for the
    target dataset file into domain A and only the target dataset file as
    domain B.

    e.g. if 4 dataset files hh101, hh102, hh103, hh104 and target is hh101, then
        domain A: hh102, hh103, hh104
        domain B: hh101
    """
    # Get list of all the datasets
    files = pathlib.Path(dir_name).glob(feature_set+"_*.hdf5")
    paths = [(x.stem, str(x)) for x in files]
    target_file = None
    train_data_a = []
    train_labels_a = []
    test_data_a = []
    test_labels_a = []

    # Source(s) -- all except target
    for name, f in paths:
        # Skip target for now
        if name == feature_set+"_"+target:
            target_file = f
            continue

        train_data_partial, train_labels_partial, \
        test_data_partial, test_labels_partial = load_single_fold(f, fold)

        train_data_a.append(train_data_partial)
        train_labels_a.append(train_labels_partial)
        test_data_a.append(test_data_partial)
        test_labels_a.append(test_labels_partial)

    train_data_a = np.vstack(train_data_a)
    test_data_a = np.vstack(test_data_a)
    train_labels_a = np.hstack(train_labels_a)
    test_labels_a = np.hstack(test_labels_a)

    # Target
    assert target_file is not None, "Did not find target \""+target+"\""

    train_data_b, train_labels_b, \
    test_data_b, test_labels_b = load_single_fold(target_file, fold)

    return train_data_a, train_labels_a, \
        test_data_a, test_labels_a, \
        train_data_b, train_labels_b, \
        test_data_b, test_labels_b

def get_tfrecord_traintest_datasets(feature_set, target, fold, dataset, dir_name="datasets"):
    """ Get all the train sets or test sets (dataset="train" or dataset="test") """
    files = pathlib.Path(dir_name).glob(feature_set+"_*_"+dataset+"_"+str(fold)+".tfrecord")
    paths = [(x.stem, str(x)) for x in files]
    target_file = None
    tfrecords_a = []
    tfrecords_b = []

    # Source(s) -- all except target
    for name, f in paths:
        # Skip target for now
        if name == feature_set+"_"+target+"_"+dataset+"_"+str(fold):
            target_file = f
            continue

        tfrecords_a.append(f)

    # Target
    assert target_file is not None, "Did not find target \""+target+"\""
    tfrecords_b.append(target_file)

    return tfrecords_a, tfrecords_b

def get_tfrecord_datasets(feature_set, target, fold, dir_name="datasets"):
    """ Get all the train/test A/B .tfrecord files """
    tfrecords_train_a, tfrecords_train_b = \
        get_tfrecord_traintest_datasets(feature_set, target, fold, "train", dir_name)
    tfrecords_test_a, tfrecords_test_b = \
        get_tfrecord_traintest_datasets(feature_set, target, fold, "test", dir_name)

    return tfrecords_train_a, tfrecords_train_b, \
        tfrecords_test_a, tfrecords_test_b

def calc_class_weights(filenames, x_dims, num_classes, balance_pow=1.0,
        batch_size=1000000):
    # Since we're using a .tfrecord file, we need to load the data and sum
    # how many instances of each class there are in batches
    input_fn, input_hook = _get_tfrecord_input_fn(
        filenames, batch_size, x_dims, num_classes, count=True)
    _, next_labels_batch = input_fn()

    total = 0
    counts = np.zeros((num_classes,), dtype=np.int32)

    # Increment total by number of examples in each batch
    increment_total = tf.shape(next_labels_batch)[0]

    # Since the labels are one-hot encoded, we can add it to the counts
    # directly. For example, if we have 3 classes, our counts start out
    # as [0 0 0], and if we have an example of class 0, i.e. [1 0 0],
    # if we do [0 0 0]+[1 0 0]=[1 0 0], i.e. we've seen one instance of
    # class 0 now.
    # Note: first sum over examples, then we'll add this to "counts"
    increment_counts = tf.reduce_sum(tf.cast(next_labels_batch, tf.int32), axis=0)

    # Run on CPU since such a simple computation
    #config=tf.ConfigProto(device_count={'GPU': 0})
    with tf.train.SingularMonitoredSession(hooks=[input_hook]) as sess:
        # Continue till we've looked at all the data
        while True:
            try:
                inc_total, inc_counts = sess.run([increment_total, increment_counts])
            except tf.errors.OutOfRangeError:
                break

            total += inc_total
            counts += inc_counts

    # We only created a graph here to look through the data, but we don't want to
    # mess up the actual graph we'll use from this point on
    tf.reset_default_graph()

    class_weights = np.power(total/counts, balance_pow)
    print("Class weights:", class_weights)

    return class_weights

class ALConfig:
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

class TFRecordConfig:
    """
    Load an {simple,al}.config file to get the x dimensions for reshaping
    """
    def __init__(self, feature_set, dir_name="datasets"):
        """ Gets the possible features and labels """
        self.num_features = None
        self.num_classes = None
        self.time_steps = None
        self.x_dims = None

        with open(os.path.join(dir_name, feature_set+".config"), 'r') as f:
            for line in f:
                items = line.strip().split(' ')

                if len(items) > 0:
                    if items[0] == "x_dims":
                        self.x_dims = [int(x) for x in items[1:]]
                        assert len(items) == 3, "format: x_dims int int"
                    elif items[0] == "num_features":
                        self.num_features = int(items[1])
                        assert len(items) == 2, "format: num_features int"
                    elif items[0] == "num_classes":
                        self.num_classes = int(items[1])
                        assert len(items) == 2, "format: num_classes int"
                    elif items[0] == "time_steps":
                        self.time_steps = int(items[1])
                        assert len(items) == 2, "format: time_steps int"

        assert self.num_features is not None, "no \"num_features\" in .config"
        assert self.num_classes is not None, "no \"num_classes\" in .config"
        assert self.time_steps is not None, "no \"time_steps\" in .config"
        assert self.x_dims is not None, "no \"x_dims\" in .config"
