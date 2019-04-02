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

def tf_random_ones(x, mask, start, end):
    """
    Set from x[...,start:end] of each selected row (by the mask) to zero and then
    randomly pick one of them in start:end to set to 1.

    To test:
        import numpy as np
        import tensorflow as tf
        from load_data import tf_random_ones
        sensor_prob=0.5
        num_time_features=1
        num_sensors=2
        a = np.array([[5,1,2,6],[5,2,3,6],[5,3,4,6],[5,4,5,6],[5,5,6,6],
                    [5,6,7,6],[5,7,8,6],[5,8,9,6],[5,9,10,6],[5,10,11,6]])
        x = tf.convert_to_tensor(a, tf.float32)
        mask = tf.random.uniform([tf.shape(x)[0]]) < sensor_prob
        x_augmented = tf_random_ones(x, mask, num_time_features, num_time_features+num_sensors)

        sess = tf.Session()
        sess.run(x_augmented)
    """
    num_rows = tf.shape(x)[0]
    num_values = end - start

    # Zero the sensor values of those rows
    #
    # Copy x setting the rows in start:end to zero
    zeros = tf.zeros([num_rows, num_values], tf.float32)
    x_zeros = tf.concat([x[:, :start], zeros, x[:, end:]], axis=1)
    # Above we did it to all rows, so copy only the desired zeroed rows into x
    x = tf.where(mask, x_zeros, x)

    # Put 1's somewhere back in the sensors (may or may not be the same sensor)
    #
    # Get the indices (still using the above mask) of where to set to 1's
    mask_indices = tf.cast(tf.where(mask), dtype=tf.int32)
    # Randomly pick between start:end where to put the 1
    sensors = tf.random.uniform([tf.shape(input=mask_indices)[0]],
        minval=start, maxval=end, dtype=tf.int32)
    sensors = tf.expand_dims(sensors, axis=1)
    # To make this an index it needs to be, e.g. [[1,5],[2,6]], if we were
    # updating rows 1 and 2, where features 5 and 6 would be set to 1's
    set_to_one = tf.concat([mask_indices,sensors],axis=1)
    # Set those indices to ones
    ones = tf.ones([tf.shape(input=set_to_one)[0]], tf.float32)
    # Idea from:
    # https://github.com/tensorflow/tensorflow/issues/2358#issuecomment-274590896
    x = x + tf.scatter_nd(set_to_one, ones, tf.shape(input=x))

    return x

def perform_data_augmentation(x, zero_prob=0.05, time_prob=0.05,
        sensor_prob=0.05, value_prob=0.05):
    """
    Perform data augmentation for simple2 dataset (see simple2_features.py for
    the code that generates the features from the raw data)

    Note: this will not work well on AL features

    - Randomly set some rows zero -- This basically deletes the sensor event, but
      recall that the input "time window" isn't a fixed amount of time but rather
      a certain number of sensor events. Thus, this effectively (hopefully) makes
      the network learn to work even if an activity didn't have one sensor
      trigger, e.g. the person walked in a slightly different place and missed
      the motion detector.
    - Add noise to time values -- obviously people don't do things at exactly
      the same frequency, speed, etc. every time they perform an activity.
    - Add some probability of switching sensor or sensor value to one of the
      others (since one-hot encoded)

    Recall the simple2 features are (in this order):
    - time features
        - second (/60)
        - minute (/60)
        - hour (/12)
        - hour (/24)
        - second of day (/86400)
        - day of week (/7)
        - day of month (/31)
        - day of year (/366)
        - month of year (/12)
        - year
    - smart home features
        - one-hot encoding for which sensor it is, e.g. if three sensors
          <1,0,0> would be the vector for the first sensor
        - a vector of what values the sensor gave, in this case on/off/open/close,
            <1,0,0,0> - on
            <0,1,0,0> - off
            <0,0,1,0> - open
            <0,0,0,1> - close
    """
    assert len(x.shape) == 3, "augmentation data shape: (examples, times, features)"
    num_examples = tf.shape(input=x)[0]
    num_time_steps = tf.shape(input=x)[1]
    num_features = tf.shape(input=x)[2]
    num_time_features = 10
    num_values_features = 4
    num_sensors = num_features - num_time_features - num_values_features

    # Reshape into (examples*time steps, features) so we can process each row
    # individually
    x = tf.reshape(x, [num_examples*num_time_steps, num_features])
    num_rows = num_examples*num_time_steps

    # Zero rows
    mask = tf.random.uniform([num_rows]) < zero_prob
    zeros = tf.zeros([num_rows, num_features], tf.float32)
    x = tf.where(mask, zeros, x)

    # Time value noise
    # TODO

    # Switch sensors
    #
    # Randomly select rows
    mask = tf.random.uniform([num_rows]) < sensor_prob
    x = tf_random_ones(x, mask, num_time_features, num_time_features+num_sensors)

    # Switch sensor value
    mask = tf.random.uniform([num_rows]) < value_prob
    x = tf_random_ones(x, mask, num_time_features+num_sensors, num_features)

    # Return to the original shape
    x = tf.reshape(x, [num_examples, num_time_steps, num_features])

    return x

def load_tfrecords(filenames, batch_size, input_shape, num_classes, num_domains,
        evaluation=False, count=False, buffer_size=10000, eval_shuffle_seed=0,
        prefetch_buffer_size=1, data_augmentation=False):
    """ Load data from .tfrecord files (requires less memory but more disk space) """
    if len(filenames) == 0:
        return None

    # Create a description of the features
    # See: https://www.tensorflow.org/tutorials/load_data/tf-records
    feature_description = {
        'x': tf.io.FixedLenFeature(input_shape, tf.float32),
        'y': tf.io.FixedLenFeature([num_classes], tf.float32),
        'domain': tf.io.FixedLenFeature([num_domains], tf.float32),
    }

    def _parse_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        # parse_single_example is without a batch, parse_example is with batches
        parsed = tf.io.parse_example(serialized=example_proto, features=feature_description)
        x = parsed["x"]
        y = parsed["y"]
        domain = parsed["domain"]

        # Perform data augmentation
        if data_augmentation:
            x = perform_data_augmentation(x)

        return x, y, domain

    # Interleave the tfrecord files
    files = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP').prefetch(100),
        cycle_length=len(filenames), block_length=1)

    if count: # only count, so no need to shuffle
        pass
    elif evaluation: # don't repeat since we want to evaluate entire set
        dataset = dataset.shuffle(buffer_size, seed=eval_shuffle_seed)
    else: # repeat, shuffle, and batch
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size))

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_parse_function)

    # Prefetch for speed up
    # See: https://www.tensorflow.org/guide/performance/datasets
    dataset = dataset.prefetch(prefetch_buffer_size)

    return dataset

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

def tf_domain_labels(label, batch_size, num_domains=2):
    """ Generate one-hot encoded labels for which domain data is from (using TensorFlow) """
    return tf.tile(tf.one_hot([0], depth=num_domains), [batch_size,1])

def domain_labels(label, batch_size, num_domains=2):
    """ Generate one-hot encoded labels for which domain data is from (using numpy) """
    return np.tile(np.eye(num_domains)[label], [batch_size,1])

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

def shuffle_together_calc(length, seed=None):
    """ Generate indices of numpy array shuffling, then do x[p] """
    rand = np.random.RandomState(seed)
    p = rand.permutation(length)
    return p

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
    fold for now
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

def get_tfrecord_traintest_datasets(feature_set, target, fold, dataset, sample, dir_name="datasets"):
    """ Get all the train sets or test sets (dataset="train" or dataset="test") """
    if sample:
        feature_set = "sample_"+feature_set

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

    # Target -- require it unless it's blank, i.e. we don't want a target
    if target != "":
        assert target_file is not None, \
            "Did not find target "+target+" fold "+str(fold)
        tfrecords_b.append(target_file)

    return tfrecords_a, tfrecords_b

def get_tfrecord_datasets(feature_set, target, fold, sample=False, dir_name="datasets"):
    """ Get all the train/test A/B .tfrecord files """
    tfrecords_train_a, tfrecords_train_b = \
        get_tfrecord_traintest_datasets(feature_set, target, fold, "train", sample, dir_name)
    tfrecords_valid_a, tfrecords_valid_b = \
        get_tfrecord_traintest_datasets(feature_set, target, fold, "valid", sample, dir_name)
    tfrecords_test_a, tfrecords_test_b = \
        get_tfrecord_traintest_datasets(feature_set, target, fold, "test", sample, dir_name)

    return tfrecords_train_a, tfrecords_train_b, \
        tfrecords_valid_a, tfrecords_valid_b, \
        tfrecords_test_a, tfrecords_test_b

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
        self.num_domains = None
        self.time_steps = None
        self.input_shape = None

        with open(os.path.join(dir_name, feature_set+".config"), 'r') as f:
            for line in f:
                items = line.strip().split(' ')

                if len(items) > 0:
                    if items[0] == "x_dims":
                        self.input_shape = [int(x) for x in items[1:]]
                        assert len(items) == 3, "format: x_dims int int"
                    elif items[0] == "num_features":
                        self.num_features = int(items[1])
                        assert len(items) == 2, "format: num_features int"
                    elif items[0] == "num_classes":
                        self.num_classes = int(items[1])
                        assert len(items) == 2, "format: num_classes int"
                    elif items[0] == "num_domains":
                        self.num_domains = int(items[1])
                        assert len(items) == 2, "format: num_domains int"
                    elif items[0] == "time_steps":
                        self.time_steps = int(items[1])
                        assert len(items) == 2, "format: time_steps int"

        assert self.num_features is not None, "no \"num_features\" in .config"
        assert self.num_classes is not None, "no \"num_classes\" in .config"
        assert self.num_domains is not None, "no \"num_domains\" in .config"
        assert self.time_steps is not None, "no \"time_steps\" in .config"
        assert self.input_shape is not None, "no \"x_dims\" in .config"
