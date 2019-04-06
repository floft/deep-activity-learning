#!/usr/bin/env python3
"""
Deep activity learning
"""
import os
import time
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging
from tensorflow.python.framework import config as tfconfig

from load_data import load_tfrecords, domain_labels, get_tfrecord_datasets, \
    ALConfig, TFRecordConfig, calc_class_weights
from models import DomainAdaptationModel, make_task_loss, make_domain_loss, \
    compute_accuracy
from file_utils import last_modified_number, write_finished
from metrics import Metrics
from checkpoints import CheckpointManager

FLAGS = flags.FLAGS

flags.DEFINE_enum("model", "flat", ["flat"], "What model type to use")
flags.DEFINE_string("modeldir", "models", "Directory for saving model files")
flags.DEFINE_string("logdir", "logs", "Directory for saving log files")
flags.DEFINE_boolean("adapt", False, "Perform domain adaptation on the model")
flags.DEFINE_boolean("generalize", False, "Perform domain generalization on the model")
flags.DEFINE_boolean("balance", False, "On high class imbalances, weight the loss function by class")
flags.DEFINE_integer("fold", 2, "What fold to use from the dataset files")
flags.DEFINE_string("target", "", "What dataset to use as the target (default none, i.e. blank)")
flags.DEFINE_enum("features", "al", ["al", "simple", "simple2"], "What type of features to use")
flags.DEFINE_integer("steps", 100000, "Number of training steps to run")
flags.DEFINE_integer("batch", 1024, "Batch size to use for training")
flags.DEFINE_integer("eval_batch", 16384, "Batch size to use for evaluation")
flags.DEFINE_float("lr", 0.0001, "Learning rate for training")
flags.DEFINE_float("lr_mult", 1.0, "Multiplier for extra discriminator training learning rate")
flags.DEFINE_float("gpumem", 0.1, "Percentage of GPU memory to let TensorFlow use")
flags.DEFINE_integer("model_steps", 4000, "Save the model every so many steps")
flags.DEFINE_integer("log_train_steps", 500, "Log training information every so many steps")
flags.DEFINE_integer("log_val_steps", 4000, "Log validation information every so many steps (also saves model)")
flags.DEFINE_integer("max_examples", 0, "Max number of examples to evaluate for validation (default 0, i.e. all)")
flags.DEFINE_boolean("augment", False, "Perform data augmentation (for simple2 dataset)")
flags.DEFINE_boolean("sample", False, "Only use a small amount of data for training/testing")
flags.DEFINE_boolean("test", False, "Swap out validation data for real test data (debugging in tensorboard)")
flags.DEFINE_float("min_domain_accuracy", 0.5, "If generalize, min domain classifier accuracy")
flags.DEFINE_float("max_domain_iters", 10, "If generalize, max domain classifier training iterations")
flags.DEFINE_boolean("debug", False, "Start new log/model/images rather than continuing from previous run")
flags.DEFINE_integer("debugnum", -1, "Specify exact log/model/images number to use rather than incrementing from last. (Don't pass both this and --debug at the same time.)")

#flags.mark_flag_as_required("model")

def get_directory_names():
    """ Figure out the log and model directory names """
    prefix = FLAGS.target+"-fold"+str(FLAGS.fold)+"-"+FLAGS.model

    if FLAGS.adapt:
        prefix += "-da"
    elif FLAGS.generalize:
        prefix += "-dg"

    # Use the number specified on the command line (higher precidence than --debug)
    if FLAGS.debugnum >= 0:
        attempt = FLAGS.debugnum
        logging.info("Debugging attempt: %s", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(FLAGS.modeldir, prefix)
        log_dir = os.path.join(FLAGS.logdir, prefix)
    # Find last one, increment number
    elif FLAGS.debug:
        attempt = last_modified_number(FLAGS.logdir, prefix+"*")
        attempt = attempt+1 if attempt is not None else 1
        logging.info("Debugging attempt: %s", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(FLAGS.modeldir, prefix)
        log_dir = os.path.join(FLAGS.logdir, prefix)
    # If no debugging modes, use the model and log directory with only the "prefix"
    # (even though it's not actually a prefix in this case, it's the whole name)
    else:
        model_dir = os.path.join(FLAGS.modeldir, prefix)
        log_dir = os.path.join(FLAGS.logdir, prefix)

    return model_dir, log_dir

@tf.function
def train_step(data_a, data_b, model, opt, d_opt, global_step,
        source_domain, target_domain, task_loss, domain_loss):
    """ Compiled training step that we call many times """
    if data_a is not None:
        data_batch_a, labels_batch_a, domains_batch_a = data_a
    if data_b is not None:
        data_batch_b, labels_batch_b, domains_batch_b = data_b

    # Concatenate for adaptation - concatenate source labels with all-zero
    # labels for target since we can't use the target labels during
    # unsupervised domain adaptation
    if FLAGS.adapt:
        assert data_a is not None, "Adaptation requires both datasets A and B"
        assert data_b is not None, "Adaptation requires both datasets A and B"
        x = np.concatenate((data_batch_a, data_batch_b), axis=0)
        task_y_true = np.concatenate((labels_batch_a, np.zeros(labels_batch_b.shape)), axis=0)
        domain_y_true = np.concatenate((source_domain, target_domain), axis=0)
    elif FLAGS.generalize:
        x = data_batch_a
        task_y_true = labels_batch_a
        domain_y_true = domains_batch_a
    else:
        x = data_batch_a
        task_y_true = labels_batch_a
        domain_y_true = source_domain

    # GRL schedule from DANN paper
    grl_lambda = 2/(1+tf.exp(-10*(global_step/(FLAGS.steps+1))))-1

    # Run data through model and compute loss
    with tf.GradientTape() as tape, tf.GradientTape() as d_tape:
        task_y_pred, domain_y_pred = model(x, grl_lambda=grl_lambda, training=True)

        d_loss = domain_loss(domain_y_true, domain_y_pred)
        loss = task_loss(task_y_true, task_y_pred, training=True) + d_loss

    # Only update domain classifier during adaptation or generalization
    if FLAGS.adapt or FLAGS.generalize:
        trainable_vars = model.trainable_variables
    else:
        trainable_vars = model.trainable_variables_exclude_domain

    # Update model
    grad = tape.gradient(loss, trainable_vars)
    opt.apply_gradients(zip(grad, trainable_vars))

    # Update discriminator
    if FLAGS.adapt:
        d_grad = d_tape.gradient(d_loss, model.domain_classifier.trainable_variables)
        d_opt.apply_gradients(zip(d_grad, model.domain_classifier.trainable_variables))
    elif FLAGS.generalize:
        for _ in range(FLAGS.max_domain_iters):
            with tf.GradientTape() as d_tape:
                task_y_pred, domain_y_pred = model(x, grl_lambda=0.0, training=True)
                d_loss = domain_loss(domain_y_true, domain_y_pred)

            d_grad = d_tape.gradient(d_loss, model.domain_classifier.trainable_variables)
            d_opt.apply_gradients(zip(d_grad, model.domain_classifier.trainable_variables))

            # Break if high enough accuracy
            domain_acc = compute_accuracy(domain_y_true, domain_y_pred)
            if domain_acc > FLAGS.min_domain_accuracy:
                break

def train(num_classes, num_domains, input_shape,
        tfrecords_train_a, tfrecords_train_b,
        tfrecords_test_a, tfrecords_test_b,
        config, model_dir, log_dir):
    batch_size = FLAGS.batch
    eval_batch_size = FLAGS.eval_batch

    # For adaptation, we'll be concatenating together half source and half target
    # data, so to keep the batch_size about the same, we'll cut it in half
    if FLAGS.adapt:
        batch_size = batch_size // 2

    # Input training data
    train_data_a = load_tfrecords(tfrecords_train_a, batch_size, input_shape,
        num_classes, num_domains, data_augmentation=FLAGS.augment)
    train_data_b = load_tfrecords(tfrecords_train_b, batch_size, input_shape,
        num_classes, num_domains, data_augmentation=FLAGS.augment)
    # These iterate forever due to the .repeat() in the above
    train_data_a_iter = iter(train_data_a) if train_data_a is not None else None
    train_data_b_iter = iter(train_data_b) if train_data_b is not None else None

    # Load validation data
    eval_data_a = load_tfrecords(tfrecords_test_a, eval_batch_size, input_shape,
        num_classes, num_domains, evaluation=True, max_examples=FLAGS.max_examples)
    eval_data_b = load_tfrecords(tfrecords_test_b, eval_batch_size, input_shape,
        num_classes, num_domains, evaluation=True, max_examples=FLAGS.max_examples)

     # Loss functions
    class_weights = 1.0

    if FLAGS.balance:
        class_weights = calc_class_weights(tfrecords_train_a, input_shape,
            num_classes, num_domains)

    task_loss = make_task_loss(class_weights, FLAGS.adapt)
    domain_loss = make_domain_loss(FLAGS.adapt or FLAGS.generalize)

    # Above we needed to load with the right number of num_domains, but for
    # adaptation, we only want two: source and target. Default for any
    # non-generalization to also use two since the resulting network is smaller.
    if not FLAGS.generalize:
        num_domains = 2

    # Source domain will be [[1,0], [1,0], ...] and target domain [[0,1], [0,1], ...]
    source_domain = domain_labels(0, batch_size, num_domains)
    target_domain = domain_labels(1, batch_size, num_domains)

    # We need to know where we are in training for the GRL lambda schedule
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Build our model
    model = DomainAdaptationModel(num_classes, num_domains, FLAGS.model,
        FLAGS.generalize)

    # Optimizers
    opt = tf.keras.optimizers.Adam(FLAGS.lr)
    d_opt = tf.keras.optimizers.Adam(FLAGS.lr*FLAGS.lr_mult)

    # Checkpoints
    checkpoint = tf.train.Checkpoint(
        global_step=global_step, opt=opt, d_opt=d_opt, model=model)
    checkpoint_manager = CheckpointManager(checkpoint, model_dir, log_dir)
    checkpoint_manager.restore_latest()

    # Metrics
    have_target_domain = train_data_b is not None
    metrics = Metrics(log_dir, num_classes, num_domains, config,
        task_loss, domain_loss, FLAGS.generalize, have_target_domain)

    # Start training
    for i in range(int(global_step), FLAGS.steps+1):
        # Get data for this iteration
        data_a = next(train_data_a_iter) if train_data_a_iter is not None else None
        data_b = next(train_data_b_iter) if train_data_b_iter is not None else None

        t = time.time()
        train_step(data_a, data_b, model, opt, d_opt, global_step,
            source_domain, target_domain, task_loss, domain_loss)
        global_step.assign_add(1)
        t = time.time() - t

        if i%10 == 0:
            logging.info("step %d took %f seconds", int(global_step), t)

        # Metrics on training/validation data
        if i%FLAGS.log_train_steps == 0:
            metrics.train(model, data_a, data_b, global_step, t)

        validation_accuracy = None
        if i%FLAGS.log_val_steps == 0:
            validation_accuracy = metrics.test(model, eval_data_a, eval_data_b, global_step)

        # Checkpoints -- Save either if at the right model step or if we found
        # a new validation accuracy. If this is better than the previous best
        # model, we need to make a new checkpoint so we can restore from this
        # step with the best accuracy.
        if i%FLAGS.model_steps == 0 or validation_accuracy is not None:
            checkpoint_manager.save(int(global_step-1), validation_accuracy)

    # We're done -- used for hyperparameter tuning
    write_finished(log_dir)

def main(argv):
    # Allow running multiple at once
    # https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth
    # https://github.com/tensorflow/tensorflow/issues/25138
    # Note: GPU options must be set at program startup
    tfconfig.set_gpu_per_process_memory_fraction(FLAGS.gpumem)

    assert not (FLAGS.adapt and FLAGS.generalize), \
        "Currently cannot enable both adaptation and generalization at the same time"

    #
    # Load datasets and config files about dimensions of the data
    #
    al_config = ALConfig()
    tfrecord_config = TFRecordConfig(FLAGS.features)
    assert len(al_config.labels) == tfrecord_config.num_classes, \
        "al.config and tfrecord config disagree on the number of classes: " \
        +str(len(al_config.labels))+" vs. "+str(tfrecord_config.num_classes)

    tfrecords_train_a, tfrecords_train_b, \
    tfrecords_valid_a, tfrecords_valid_b, \
    tfrecords_test_a, tfrecords_test_b = \
        get_tfrecord_datasets(FLAGS.features, FLAGS.target, FLAGS.fold, FLAGS.sample)

    # If testing, then we'll test on the real test set and add the validation data
    # to the training set. Otherwise, if not testing, then the "test" set is the
    # validation data -- e.g. for hyperparameter tuning we evaluate on validation
    # data and pick the best model, then set --test and train on train/valid sets
    # and evaluate on the real test set.
    if FLAGS.test:
        tfrecords_train_a += tfrecords_valid_a
        tfrecords_train_b += tfrecords_valid_b
    else:
        tfrecords_test_a = tfrecords_valid_a
        tfrecords_test_b = tfrecords_valid_b

    #
    # Figure out the log and model directory filenames
    #
    model_dir, log_dir = get_directory_names()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #
    # Run training
    #
    train(tfrecord_config.num_classes,
            tfrecord_config.num_domains,
            tfrecord_config.input_shape,
            tfrecords_train_a, tfrecords_train_b,
            tfrecords_test_a, tfrecords_test_b,
            config=al_config,
            model_dir=model_dir,
            log_dir=log_dir)

if __name__ == "__main__":
    app.run(main)
