#!/usr/bin/env python3
"""
Deep activity learning

(code based on my TF implementation of VRADA: https://github.com/floft/vrada)
"""
import os
import time
import argparse
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from absl import app
from absl import flags
from absl import logging
from tensorflow.python.framework import config as tfconfig

# Make sure matplotlib is not interactive
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from plot import plot_embedding, plot_random_time_series, plot_real_time_series
from model import make_model
from load_data import load_tfrecords, domain_labels, get_tfrecord_datasets, ALConfig, TFRecordConfig
from eval_utils import get_files_to_keep, get_step_from_log, delete_models_except
from file_utils import last_modified_number, get_best_valid_accuracy, \
    write_best_valid_accuracy, write_finished

FLAGS = flags.FLAGS

flags.DEFINE_string("modeldir", "models", "Directory for saving model files")
flags.DEFINE_string("logdir", "logs", "Directory for saving log files")
flags.DEFINE_boolean("adapt", False, "Perform domain adaptation on the model")
flags.DEFINE_boolean("generalize", False, "Perform domain generalization on the model")
flags.DEFINE_integer("fold", 0, "What fold to use from the dataset files")
flags.DEFINE_string("target", "", "What dataset to use as the target (default none, i.e. blank)")
flags.DEFINE_enum("features", "al", ["al", "simple", "simple2"], "What type of features to use")
flags.DEFINE_integer("steps", 100000, "Number of training steps to run")
flags.DEFINE_integer("batch", 1024, "Batch size to use")
flags.DEFINE_float("lr", 0.0001, "Learning rate for training")
flags.DEFINE_float("lr_mult", 1.0, "Multiplier for extra discriminator training learning rate")
flags.DEFINE_float("gpu_mem", 0.4, "Percentage of GPU memory to let TensorFlow use")
flags.DEFINE_integer("model_steps", 4000, "Save the model every so many steps")
flags.DEFINE_integer("log_steps", 500, "Log training losses and accuracy every so many steps")
flags.DEFINE_integer("log_steps_val", 4000, "Log validation accuracy and AUC every so many steps")
flags.DEFINE_integer("log_steps_extra", 100000, "Log weights, plots, etc. every so many steps")
flags.DEFINE_integer("max_examples", 0, "Max number of examples to evaluate for validation (default 0, i.e. all)")
flags.DEFINE_integer("max_plot_examples", 1000, "Max number of examples to use for plotting")
flags.DEFINE_boolean("augment", False, "Perform data augmentation (for simple2 dataset)")
flags.DEFINE_boolean("sample", False, "Only use a small amount of data for training/testing")
flags.DEFINE_boolean("test", False, "Swap out validation data for real test data (debugging in tensorboard)")
flags.DEFINE_boolean("bidirectional", False, "Use a bidirectional RNN (when selected method includes an RNN)")
flags.DEFINE_boolean("feature_extractor", True, "Use a feature extractor before task classifier/domain predictor")
flags.DEFINE_float("min_domain_accuracy", 0.5, "If generalize, min domain classifier accuracy")
flags.DEFINE_float("max_domain_iters", 10, "If generalize, max domain classifier training iterations")
flags.DEFINE_boolean("debug", False, "Start new log/model/images rather than continuing from previous run")
flags.DEFINE_integer("debug_num", -1, "Specify exact log/model/images number to use rather than incrementing from last. (Don't pass both this and --debug at the same time.)")

def update_metrics_on_val(sess,
    eval_input_hook_a, eval_input_hook_b,
    next_data_batch_test_a, next_labels_batch_test_a,
    next_data_batch_test_b, next_labels_batch_test_b,
    next_domains_batch_test_a, next_domains_batch_test_b,
    x, y, domain, keep_prob, training, num_domains, generalization,
    batch_size, update_metrics_a, update_metrics_b,
    max_examples):
    """
    Calculate metrics over all the evaluation data, but batched to make sure
    we don't run out of memory

    Note: if max_examples is less than the total number of validation examples,
    it'll take the first max_examples. If it's zero, it'll evaluate all.
    """
    # Keep track of how many samples we've evaluated so far
    examples = 0

    # Reinitialize the evaluation batch initializers so we start at
    # the beginning of the evaluation data again
    eval_input_hook_a.iter_init_func(sess)
    eval_input_hook_b.iter_init_func(sess)

    # We'll break if we either evaluate all of the evaluation data and thus
    # get the out of range TensorFlow exception or of we have now evaluated
    # at least the number we wanted to.
    #
    # Evaluate all if max_examples == 0
    #
    # Domain A
    if next_data_batch_test_a is not None and \
        next_labels_batch_test_a is not None and \
        next_domains_batch_test_a is not None:
        while max_examples == 0 or examples < max_examples:
            try:
                # Get next evaluation batch
                eval_data_a, eval_labels_a, eval_domains_a = sess.run([
                    next_data_batch_test_a, next_labels_batch_test_a,
                    next_domains_batch_test_a,
                ])

                # Make sure we don't go over the desired number of examples
                # But, only if we don't want to evaluate all examples (i.e. if
                # max_examples == 0)
                if max_examples != 0:
                    diff = max_examples - examples

                    if examples + eval_data_a.shape[0] > max_examples:
                        eval_data_a = eval_data_a[:diff]
                        eval_labels_a = eval_labels_a[:diff]

                examples += eval_data_a.shape[0]

                if generalization:
                    source_domain = eval_domains_a
                else:
                    # Match the number of examples we have
                    source_domain = domain_labels(0, eval_data_a.shape[0], num_domains)

                # Log summaries run on the evaluation/validation data
                sess.run(update_metrics_a, feed_dict={
                    x: eval_data_a, y: eval_labels_a, domain: source_domain,
                    keep_prob: 1.0, training: False
                })
            except tf.errors.OutOfRangeError:
                break

    # Domain B -- separate since very likely a different size
    examples = 0
    if next_data_batch_test_b is not None and \
        next_labels_batch_test_b is not None and \
        next_domains_batch_test_b is not None:
        while max_examples == 0 or examples < max_examples:
            try:
                # Get next evaluation batch
                eval_data_b, eval_labels_b, eval_domains_b = sess.run([
                    next_data_batch_test_b, next_labels_batch_test_b,
                    next_domains_batch_test_b,
                ])

                # Make sure we don't go over the desired number of examples
                # But, only if we don't want to evaluate all examples (i.e. if
                # max_examples == 0)
                if max_examples != 0:
                    diff = max_examples - examples

                    if examples + eval_data_b.shape[0] > max_examples:
                        eval_data_b = eval_data_b[:diff]
                        eval_labels_b = eval_labels_b[:diff]

                examples += eval_data_b.shape[0]

                if generalization:
                    target_domain = eval_domains_b
                else:
                    # Match the number of examples we have
                    target_domain = domain_labels(1, eval_data_b.shape[0], num_domains)

                # Log summaries run on the evaluation/validation data
                sess.run(update_metrics_b, feed_dict={
                    x: eval_data_b, y: eval_labels_b, domain: target_domain,
                    keep_prob: 1.0, training: False
                })
            except tf.errors.OutOfRangeError:
                break

def evaluation_plots(sess,
    eval_input_hook_a, eval_input_hook_b,
    next_data_batch_test_a, next_labels_batch_test_a,
    next_data_batch_test_b, next_labels_batch_test_b,
    feature_extractor, x, keep_prob, training, adaptation,
    extra_model_outputs, num_features, first_step, config,
    max_plot_examples=100,
    tsne_filename=None,
    pca_filename=None,
    recon_a_filename=None,
    recon_b_filename=None,
    real_a_filename=None,
    real_b_filename=None):
    """
    Run the first batch of evaluation data through the feature extractor, then
    generate and return the PCA and t-SNE plots. Optionally, save these to a file
    as well.
    """
    # Skip if we don't have all domain A and B data
    if next_data_batch_test_a is None or next_labels_batch_test_a is None or \
        next_data_batch_test_b is None or next_labels_batch_test_b is None:
        return None

    # Get the first batch of evaluation data to use for these plots
    eval_input_hook_a.iter_init_func(sess)
    eval_input_hook_b.iter_init_func(sess)
    eval_data_a, eval_labels_a, eval_data_b, eval_labels_b = sess.run([
        next_data_batch_test_a, next_labels_batch_test_a,
        next_data_batch_test_b, next_labels_batch_test_b,
    ])

    # Limit the number of plots
    if eval_data_a.shape[0] > max_plot_examples:
        eval_data_a = eval_data_a[:max_plot_examples]
        eval_labels_a = eval_labels_a[:max_plot_examples]

    if eval_data_b.shape[0] > max_plot_examples:
        eval_data_b = eval_data_b[:max_plot_examples]
        eval_labels_b = eval_labels_b[:max_plot_examples]

    # Match the number of plots we have
    source_domain = domain_labels(0, eval_data_a.shape[0])
    target_domain = domain_labels(1, eval_data_b.shape[0])

    combined_x = np.concatenate((eval_data_a, eval_data_b), axis=0)
    combined_labels = np.concatenate((eval_labels_a, eval_labels_b), axis=0)
    combined_domain = np.concatenate((source_domain, target_domain), axis=0)

    embedding = sess.run(feature_extractor, feed_dict={
        x: combined_x, keep_prob: 1.0, training: False
    })

    tsne = TSNE(n_components=2, init='pca', n_iter=3000).fit_transform(embedding)
    pca = PCA(n_components=2).fit_transform(embedding)

    if adaptation:
        title = "Domain Adaptation"
    else:
        title = "No Adaptation"

    tsne_plot = plot_embedding(
        tsne, combined_labels.argmax(1), combined_domain.argmax(1),
        title=title + " - t-SNE", filename=tsne_filename)
    pca_plot = plot_embedding(pca, combined_labels.argmax(1), combined_domain.argmax(1),
        title=title + " - PCA", filename=pca_filename)

    plots = []
    if tsne_plot is not None:
        plots.append(('tsne', tsne_plot))
    if pca_plot is not None:
        plots.append(('pca', pca_plot))

    # Output time-series "reconstructions" from our generator (if VRNN and we
    # only have a single-dimensional x, e.g. in the "trivial" datasets)
    if extra_model_outputs is not None:
        # We'll get the decoder's mu and sigma from the evaluation/validation set since
        # it's much larger than the training batches
        mu_a, sigma_a = sess.run(extra_model_outputs, feed_dict={
            x: eval_data_a, keep_prob: 1.0, training: False
        })

        mu_b, sigma_b = sess.run(extra_model_outputs, feed_dict={
            x: eval_data_b, keep_prob: 1.0, training: False
        })

        for i in range(num_features):
            feature_name = config.int_to_feature(i)

            recon_a_plot = plot_random_time_series(
                mu_a[:,:,i], sigma_a[:,:,i],
                title='VRNN Reconstruction (source domain, feature '+str(i)+')',
                filename=recon_a_filename)

            recon_b_plot = plot_random_time_series(
                mu_b[:,:,i], sigma_b[:,:,i],
                title='VRNN Reconstruction (target domain, feature '+str(i)+')',
                filename=recon_b_filename)

            plots.append(('feature_'+feature_name+'_reconstruction_a', recon_a_plot))
            plots.append(('feature_'+feature_name+'_reconstruction_b', recon_b_plot))

            # Real data -- but only plot once, since this doesn't change for the
            # evaluation data
            if first_step:
                real_a_plot = plot_real_time_series(
                    eval_data_a[:,:,i],
                    title='Real Data (source domain, feature '+str(i)+')',
                    filename=real_a_filename)

                real_b_plot = plot_real_time_series(
                    eval_data_b[:,:,i],
                    title='Real Data (target domain, feature '+str(i)+')',
                    filename=real_b_filename)

                plots.append(('feature_'+feature_name+'_real_a', real_a_plot))
                plots.append(('feature_'+feature_name+'_real_b', real_b_plot))

    return plots

def create_reset_metric(metric, scope='reset_metrics', **metric_args):
    """
    Metric that can be reset
    https://github.com/tensorflow/tensorflow/issues/4814#issuecomment-314801758
    """
    with tf.compat.v1.variable_scope(scope) as scope:
        metric_op, update_op = metric(**metric_args)
        variables = tf.contrib.framework.get_variables(
            scope, collection=tf.compat.v1.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.compat.v1.variables_initializer(variables)

    return metric_op, update_op, reset_op

def metric_summaries(domain,
    task_labels, task_predictions_raw,
    domain_labels, domain_predictions_raw,
    num_classes, multi_class, config,
    datasets=("training", "validation")):
    """
    Generate the summaries for a particular domain (e.g. "source" or "target")
    and for each dataset (e.g. both "training" and "validation") given
    task and domain labels and predictions and the number of classes.

    This generates both overall and per-class metrics. The computation of overall
    and per-class accuracy differs for multi-class vs. single-class predictions.
    """
    summs = [[] for d in datasets]
    summs_values = {}

    with tf.compat.v1.variable_scope("metrics_%s" % domain):
        # Depending on if multi-class, what we consider a positive class is different
        # TODO this actually is probably actually wrong for multi-class
        if multi_class:
            # If multi-class, then each output is a sigmoid independent of the others,
            # so for each class check >0.5 for predicting a "yes" for that class.
            per_class_predictions = tf.cast(
                tf.greater(task_predictions_raw, 0.5), tf.float32)

            # Since multi-class, our overall accuracy from predictions will need to
            # compare the predictions for each class
            acc_labels = task_labels
            acc_predictions = per_class_predictions
        else:
            # If only predicting a single class (using softmax), then look for the
            # max value
            # e.g. [0.2 0.2 0.4 0.2] -> [0 0 1 0]
            per_class_predictions = tf.one_hot(
                tf.argmax(input=task_predictions_raw, axis=-1), num_classes)

            # For overall accuracy if not multi-class, we want to look at *just*
            # the argmax; otherwise, if there's a bunch of classes we'll get very
            # high accuracies due to all the matching zeros.
            acc_labels = tf.argmax(input=task_labels, axis=-1,
                output_type=tf.int32)
            acc_predictions = tf.argmax(input=task_predictions_raw, axis=-1,
                output_type=tf.int32)

        # Domain classification accuracy is always binary
        domain_acc_labels = tf.argmax(input=domain_labels, axis=-1,
            output_type=tf.int32)
        domain_acc_predictions = tf.argmax(input=domain_predictions_raw, axis=-1,
            output_type=tf.int32)

        # Overall metrics
        task_acc, update_task_acc, reset_task_acc = create_reset_metric(
            tf.compat.v1.metrics.accuracy, "task_acc",
            labels=acc_labels, predictions=acc_predictions)
        task_auc, update_task_auc, reset_task_auc = create_reset_metric(
            tf.compat.v1.metrics.auc, "task_auc",
            labels=task_labels, predictions=task_predictions_raw)
        domain_acc, update_domain_acc, reset_domain_acc = create_reset_metric(
            tf.compat.v1.metrics.accuracy, "domain_acc",
            labels=domain_acc_labels, predictions=domain_acc_predictions)

    reset_metrics = [reset_task_acc, reset_task_auc, reset_domain_acc]
    update_metrics = [update_task_acc, update_task_auc, update_domain_acc]

    for j, dataset in enumerate(datasets):
        summs[j] += [
            tf.compat.v1.summary.scalar("auc_task/%s/%s" % (domain, dataset), task_auc),
            tf.compat.v1.summary.scalar("accuracy_task/%s/%s" % (domain, dataset), task_acc),
            tf.compat.v1.summary.scalar("accuracy_domain/%s/%s" % (domain, dataset), domain_acc),
        ]

        summs_values["auc_task/%s/%s" % (domain, dataset)] = task_auc
        summs_values["accuracy_task/%s/%s" % (domain, dataset)] = task_acc
        summs_values["accuracy_domain/%s/%s" % (domain, dataset)] = domain_acc

    # Per-class metrics
    for i in range(num_classes):
        class_name = config.int_to_label(i)

        with tf.compat.v1.variable_scope("metrics_%s/class_%s" % (domain,class_name)):
            # Get ith column (all groundtruth/predictions for ith class)
            class_y = tf.slice(
                task_labels, [0,i], [tf.shape(input=task_labels)[0], 1])
            class_predictions = tf.slice(
                per_class_predictions, [0,i], [tf.shape(input=task_labels)[0], 1])

            if multi_class:
                # Note: using the above directly works for multi-class since any
                # example could be any number of classes. This does not work for
                # single-prediction since then every prediction has n-1 zeros
                # for n classes. Thus, class-accuracies are always high.
                acc_class_y = class_y
                acc_class_predictions = class_predictions
            else:
                # For single-class prediction, we want to first isolate which
                # examples in the batch were supposed to be class X. Then, of
                # those, calculate accuracy = correct / total.
                rows_of_class_y = tf.where(tf.equal(class_y, 1)) # i.e. have 1
                acc_class_y = tf.gather(class_y, rows_of_class_y)
                acc_class_predictions = tf.gather(class_predictions, rows_of_class_y)

        for j, dataset in enumerate(datasets):
            with tf.compat.v1.variable_scope("metrics_%s/class_%s/%s" % (domain,class_name,dataset)):
                acc, update_acc, reset_acc = create_reset_metric(
                    tf.compat.v1.metrics.accuracy, "acc_%d" % j,
                    labels=acc_class_y, predictions=acc_class_predictions)
                tp, update_TP, reset_TP = create_reset_metric(
                    tf.compat.v1.metrics.true_positives, "TP_%d" % j,
                    labels=class_y, predictions=class_predictions)
                fp, update_FP, reset_FP = create_reset_metric(
                    tf.compat.v1.metrics.false_positives, "FP_%d" % j,
                    labels=class_y, predictions=class_predictions)
                tn, update_TN, reset_TN = create_reset_metric(
                    tf.compat.v1.metrics.true_negatives, "TN_%d" % j,
                    labels=class_y, predictions=class_predictions)
                fn, update_FN, reset_FN = create_reset_metric(
                    tf.compat.v1.metrics.false_negatives, "FN_%d" % j,
                    labels=class_y, predictions=class_predictions)

            reset_metrics += [reset_acc, reset_TP, reset_FP, reset_TN, reset_FN]
            update_metrics += [update_acc, update_TP, update_FP, update_TN, update_FN]

            summs[j] += [
                tf.compat.v1.summary.scalar("accuracy_task_class_%s/%s/%s" % (class_name,domain,dataset), acc),
                tf.compat.v1.summary.scalar("rates_class_%s/TP/%s/%s" % (class_name,domain,dataset), tp),
                tf.compat.v1.summary.scalar("rates_class_%s/FP/%s/%s" % (class_name,domain,dataset), fp),
                tf.compat.v1.summary.scalar("rates_class_%s/TN/%s/%s" % (class_name,domain,dataset), tn),
                tf.compat.v1.summary.scalar("rates_class_%s/FN/%s/%s" % (class_name,domain,dataset), fn),
            ]

            summs_values["accuracy_task_class_%s/%s/%s" % (class_name,domain,dataset)] = acc
            summs_values["rates_class_%s/TP/%s/%s" % (class_name,domain,dataset)] = tp
            summs_values["rates_class_%s/FP/%s/%s" % (class_name,domain,dataset)] = fp
            summs_values["rates_class_%s/TN/%s/%s" % (class_name,domain,dataset)] = tn
            summs_values["rates_class_%s/FN/%s/%s" % (class_name,domain,dataset)] = fn

    return reset_metrics, update_metrics, summs, summs_values

def opt_with_summ(optimizer, loss, var_list=None):
    """
    Run the optimizer, but also create summaries for the gradients (possibly
    useful for debugging)
    """
    summaries = []

    # Calculate and perform update
    grads = optimizer.compute_gradients(loss, var_list=var_list)
    update_step = optimizer.apply_gradients(grads)

    # Generate summaries for each gradient
    # grads = [(grad1, var1), ...]
    for grad, var in grads:
        # Skip those whose gradient is not computed (i.e. not in var list above)
        if grad is not None:
            summaries.append(tf.compat.v1.summary.histogram("{}-grad".format(var.name), grad))

    return update_step, summaries

class RemoveOldCheckpoints:
    """
    Remove checkpoints that are not the best or the last one so we don't waste
    tons of disk space
    """
    def __init__(self, log_dir, model_dir):
        self.log_dir = log_dir
        self.model_dir = model_dir

    def before_save(self,):
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

@tf.function
def train_step(train_data_a_iter, train_data_b_iter, model):
    """ Compiled training step that we call many times """

    # Get data for this iteration
    data_batch_a, labels_batch_a, domains_batch_a = next(train_data_a_iter)
    data_batch_b, labels_batch_b, domains_batch_b = next(train_data_b_iter)

    with tf.GradientTape() as tape_all, tf.GradientTape() as tape_discrim:

        if FLAGS.adaptation:
            # Concatenate for adaptation - concatenate source labels with all-zero
            # labels for target since we can't use the target labels during
            # unsupervised domain adaptation
            combined_x = np.concatenate((data_batch_a, data_batch_b), axis=0)
            combined_labels = np.concatenate((labels_batch_a, np.zeros(labels_batch_b.shape)), axis=0)
            combined_domain = np.concatenate((source_domain, target_domain), axis=0)

            # Train everything in one step and domain more next. This seemed
            # to work better for me than just nondomain then domain, though
            # it seems likely the results would be similar.
            sess.run(train_all, feed_dict={
                x: combined_x, y: combined_labels, domain: combined_domain,
                grl_lambda: grl_lambda_value,
                keep_prob: FLAGS.dropout, lr: lr_value, training: True
            })

            y_pred, domain_pred = model(combined_x, grl_lambda_value)

            # Update domain more
            #
            # Depending on the num_steps, your learning rate, etc. it may be
            # beneficial to have a different learning rate here -- hence the
            # lr_multiplier option. This may also depend on your dataset though.
            sess.run(train_domain, feed_dict={
                x: combined_x, y: combined_labels, domain: combined_domain,
                grl_lambda: 0.0,
                keep_prob: FLAGS.dropout, lr: FLAGS.lr_mult*lr_value, training: True
            })

        elif FLAGS.generalization:
            sess.run(train_all, feed_dict={
                x: data_batch_a, y: labels_batch_a, domain: domains_batch_a,
                grl_lambda: grl_lambda_value,
                keep_prob: FLAGS.dropout, lr: lr_value, training: True
            })

            # Update discriminator till it's accurate enough
            for j in range(FLAGS.max_domain_iters):
                feed_dict = {
                    x: data_batch_a, y: labels_batch_a, domain: domains_batch_a,
                    grl_lambda: 0.0,
                    keep_prob: FLAGS.dropout, lr: FLAGS.lr_mult*lr_value, training: True
                }
                sess.run(train_domain, feed_dict=feed_dict)

                # Break if high enough accuracy
                domain_acc = sess.run(domain_accuracy, feed_dict=feed_dict)

                if domain_acc > FLAGS.min_domain_accuracy:
                    break

            # For debugging, print occasionally
            if i%1000 == 0:
                print("Iteration", i, "Domain iters", j, "domain acc", domain_acc)
        else:
            # Train task classifier on source domain to be correct
            sess.run(train_notdomain, feed_dict={
                x: data_batch_a, y: labels_batch_a,
                keep_prob: FLAGS.dropout, lr: lr_value, training: True
            })

def train(
        num_features, num_classes, num_domains, x_dims,
        tfrecords_train_a, tfrecords_train_b,
        tfrecords_test_a, tfrecords_test_b,
        config,
        model_dir, log_dir,
        multi_class=False):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # For adaptation, we'll be concatenating together half source and half target
    # data, so to keep the batch_size about the same, we'll cut it in half
    if FLAGS.adaptation:
        batch_size = batch_size // 2

    # Input training data
    train_data_a_iter = iter(load_tfrecords(tfrecords_train_a, batch_size, x_dims,
        num_classes, num_domains, data_augmentation=FLAGS.augment))
    train_data_b_iter = iter(load_tfrecords(tfrecords_train_b, batch_size, x_dims,
        num_classes, num_domains, data_augmentation=FLAGS.augment))

    # Load all the test data in one batch
    eval_data_a = load_tfrecords(tfrecords_test_a, batch_size, x_dims,
        num_classes, num_domains, evaluation=True)
    eval_data_b = load_tfrecords(tfrecords_test_b, batch_size, x_dims,
        num_classes, num_domains, evaluation=True)

    # Above we needed to load with the right number of num_domains, but for
    # adaptation, we only want two: source and target. Default for any
    # non-generalization to also use two since the resulting network is smaller.
    if not FLAGS.generalization:
        num_domains = 2

    # Variables
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Source domain will be [[1,0], [1,0], ...] and target domain [[0,1], [0,1], ...]
    source_domain = domain_labels(0, batch_size, num_domains)
    target_domain = domain_labels(1, batch_size, num_domains)

    model = make_model(x_dims, num_classes, num_domains, grl_lambda, FLAGS.lr, training)

    # If the discriminator has lower than a certain accuracy on classifying which
    # domain a feature output was from, then we'll train that and skip training
    # the rest of the model. Thus, we need a quick way to calculate this for
    # the training batch. For evaluation I'll use tf.metrics but for this during
    # training I'll compute manually for simplicity.
    with tf.compat.v1.variable_scope("domain_accuracy"):
        domain_accuracy = tf.reduce_mean(input_tensor=tf.cast(tf.equal(
                tf.argmax(input=domain, axis=-1),
                tf.argmax(input=domain_classifier, axis=-1)),
            tf.float32))

    # Keep track of state and summaries
    writer = tf.compat.v1.summary.FileWriter(log_dir)

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)
    checkpoint.restore(tf.train.latest_checkpoint(model_dir))

    remove_old_checkpoints = RemoveOldCheckpoints(log_dir, model_dir)

    # Allow running multiple at once
    # https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth
    # https://github.com/tensorflow/tensorflow/issues/25138
    tfconfig.set_gpu_per_process_memory_fraction(FLAGS.gpu_mem)

    # Start training
    for i in range(global_step, FLAGS.steps+1):
        t = time.time()

        if i == 0:
            writer.add_graph(sess.graph)

        # GRL schedule and learning rate schedule from DANN paper
        grl_lambda_value = 2/(1+np.exp(-10*(i/(FLAGS.steps+1))))-1

        # training step
        train_step(train_data_a_iter, train_data_b_iter, model)

        t = time.time() - t

        if i%FLAGS.log_save_steps == 0:
            # Log the step time
            summ = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(tag="step_time/model_train", simple_value=t)
            ])
            writer.add_summary(summ, step)

            t = time.time()

            if FLAGS.generalization:
                domains_a = domains_batch_a

                if next_data_batch_b is not None:
                    domains_b = domains_batch_b
            else:
                domains_a = source_domain

                if next_data_batch_b is not None:
                    domains_b = target_domain

            # Log summaries run on the training data
            #
            # Reset metrics, update metrics, then generate summaries
            feed_dict = {
                x: data_batch_a, y: labels_batch_a, domain: domains_a,
                keep_prob: 1.0, training: False
            }
            sess.run(reset_metrics)
            sess.run(update_metrics_a, feed_dict=feed_dict)
            summ = sess.run(training_summaries_a, feed_dict=feed_dict)
            writer.add_summary(summ, step)

            if next_data_batch_b is not None:
                feed_dict = {
                    x: data_batch_b, y: labels_batch_b, domain: domains_b,
                    keep_prob: 1.0, training: False
                }
                sess.run(update_metrics_b, feed_dict=feed_dict)
                summ = sess.run(training_summaries_b, feed_dict=feed_dict)
                writer.add_summary(summ, step)

            t = time.time() - t

            # Log the time to update metrics
            summ = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(tag="step_time/metrics/training", simple_value=t)
            ])
            writer.add_summary(summ, step)

        # Log validation accuracy/AUC less frequently
        if i%FLAGS.log_validation_accuracy_steps == 0:
            t = time.time()

            # Evaluation accuracy, AUC, rates, etc.
            sess.run(reset_metrics)
            update_metrics_on_val(sess,
                eval_input_hook_a, eval_input_hook_b,
                next_data_batch_test_a, next_labels_batch_test_a,
                next_data_batch_test_b, next_labels_batch_test_b,
                next_domains_batch_test_a, next_domains_batch_test_b,
                x, y, domain, keep_prob, training, num_domains, FLAGS.generalization,
                batch_size, update_metrics_a, update_metrics_b,
                FLAGS.max_examples)

            # Add the summaries about rates that were updated above in the
            # evaluation function (via update_metrics_* lists)
            summs_a, summs_b = sess.run([
                validation_summaries_a, validation_summaries_b])
            writer.add_summary(summs_a, step)

            if next_data_batch_b is not None:
                writer.add_summary(summs_b, step)

            t = time.time() - t

            # Log the time to update metrics
            summ = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(tag="step_time/metrics/validation", simple_value=t)
            ])
            writer.add_summary(summ, step)

        # Larger stuff like weights and t-SNE plots occasionally
        if i%FLAGS.log_extra_save_steps == 0:
            t = time.time()

            if FLAGS.generalization:
                domains_a = domains_batch_a
            else:
                domains_a = source_domain

            # Training weights
            summ = sess.run(training_summaries_extra_a, feed_dict={
                x: data_batch_a, y: labels_batch_a, domain: domains_a,
                keep_prob: 1.0, training: False
            })
            writer.add_summary(summ, step)

            t = time.time() - t

            # Log the time to update metrics
            summ = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(tag="step_time/extra", simple_value=t)
            ])
            writer.add_summary(summ, step)

            t = time.time()

            # t-SNE, PCA, and VRNN reconstruction plots
            first_step = i==0 # only plot real ones once
            plots = evaluation_plots(sess,
                eval_input_hook_a, eval_input_hook_b,
                next_data_batch_test_a, next_labels_batch_test_a,
                next_data_batch_test_b, next_labels_batch_test_b,
                feature_extractor, x, keep_prob, training, FLAGS.adaptation,
                extra_model_outputs, num_features, first_step, config,
                FLAGS.max_plot_examples)

            if plots is not None:
                for name, buf in plots:
                    # See: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
                    plot = tf.compat.v1.Summary.Image(encoded_image_string=buf)
                    summ = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(
                        tag=name, image=plot)])
                    writer.add_summary(summ, step)

            t = time.time() - t

            # Log the time to update metrics
            summ = tf.compat.v1.Summary(value=[
                tf.compat.v1.Summary.Value(tag="step_time/plots", simple_value=t)
            ])
            writer.add_summary(summ, step)

        # Checkpoints
        if i%FLAGS.model_steps == 0:
            remove_old_checkpoints.before_save()
            checkpoint.save(file_prefix=model_dir)
            remove_old_checkpoints.after_save()

    # We're done -- used for hyperparameter tuning
    write_finished(log_dir)

def main(argv):
    # Load datasets
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

    prefix = FLAGS.target+"-fold"+str(FLAGS.fold)+"-"+FLAGS.model

    assert not (FLAGS.adaptation and FLAGS.generalization), \
        "Currently cannot enable both adaptation and generalization at the same time"

    if FLAGS.adaptation:
        prefix += "-da"
    elif FLAGS.generalization:
        prefix += "-dg"

    # Use the number specified on the command line (higher precidence than --debug)
    if FLAGS.debug_num >= 0:
        attempt = FLAGS.debug_num
        print("Debugging attempt:", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(FLAGS.modeldir, prefix)
        log_dir = os.path.join(FLAGS.logdir, prefix)
    # Find last one, increment number
    elif FLAGS.debug:
        attempt = last_modified_number(FLAGS.logdir, prefix+"*")
        attempt = attempt+1 if attempt is not None else 1
        print("Debugging attempt:", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(FLAGS.modeldir, prefix)
        log_dir = os.path.join(FLAGS.logdir, prefix)
    # If no debugging modes, use the model and log directory with only the "prefix"
    # (even though it's not actually a prefix in this case, it's the whole name)
    else:
        model_dir = os.path.join(FLAGS.modeldir, prefix)
        log_dir = os.path.join(FLAGS.logdir, prefix)

    train(tfrecord_config.num_features,
            tfrecord_config.num_classes,
            tfrecord_config.num_domains,
            tfrecord_config.x_dims,
            tfrecords_train_a, tfrecords_train_b,
            tfrecords_test_a, tfrecords_test_b,
            config=al_config,
            model_dir=model_dir,
            log_dir=log_dir,
            multi_class=False)

if __name__ == "__main__":
    app.run(main)
