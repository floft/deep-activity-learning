#!/usr/bin/env python3
"""
Deep activity learning

(code based on my TF implementation of VRADA: https://github.com/floft/vrada)
"""
import os
import re
import time
import argparse
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Make sure matplotlib is not interactive
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from plot import plot_embedding, plot_random_time_series, plot_real_time_series
from model import build_lstm, build_vrnn, build_cnn, build_tcn, build_flat, \
    build_resnet, build_attention
from load_data import IteratorInitializerHook, _get_tfrecord_input_fn, \
    domain_labels, get_tfrecord_datasets, ALConfig, TFRecordConfig, \
    calc_class_weights

def update_metrics_on_val(sess,
    eval_input_hook_a, eval_input_hook_b,
    next_data_batch_test_a, next_labels_batch_test_a,
    next_data_batch_test_b, next_labels_batch_test_b,
    x, y, domain, keep_prob, training,
    batch_size, update_metrics_a, update_metrics_b,
    max_examples):
    """
    Calculate metrics over all the evaluation data, but batched to make sure
    we don't run out of memory

    Note: if max_examples is less than the total number of validation examples,
    it'll take the first max_examples.
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
    while examples < max_examples:
        try:
            # Get next evaluation batch
            eval_data_a, eval_labels_a, eval_data_b, eval_labels_b = sess.run([
                next_data_batch_test_a, next_labels_batch_test_a,
                next_data_batch_test_b, next_labels_batch_test_b,
            ])

            # Make sure we don't go over the desired number of examples
            diff = max_examples - examples

            if examples + eval_data_a.shape[0] > max_examples:
                eval_data_a = eval_data_a[:diff]
                eval_labels_a = eval_labels_a[:diff]

            if examples + eval_data_b.shape[0] > max_examples:
                eval_data_b = eval_data_b[:diff]
                eval_labels_b = eval_labels_b[:diff]

            examples += eval_data_a.shape[0]

            # Match the number of examples we have
            source_domain = domain_labels(0, eval_data_a.shape[0])
            target_domain = domain_labels(1, eval_data_b.shape[0])

            # Log summaries run on the evaluation/validation data
            sess.run(update_metrics_a, feed_dict={
                x: eval_data_a, y: eval_labels_a, domain: source_domain,
                keep_prob: 1.0, training: False
            })
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
    with tf.variable_scope(scope) as scope:
        metric_op, update_op = metric(**metric_args)
        variables = tf.contrib.framework.get_variables(
            scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(variables)

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

    with tf.variable_scope("metrics_%s" % domain):
        # Depending on if multi-class, what we consider a positive class is different
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
                tf.argmax(task_predictions_raw, axis=-1), num_classes)

            # For overall accuracy if not multi-class, we want to look at *just*
            # the argmax; otherwise, if there's a bunch of classes we'll get very
            # high accuracies due to all the matching zeros.
            acc_labels = tf.argmax(task_labels, axis=-1,
                output_type=tf.int32)
            acc_predictions = tf.argmax(task_predictions_raw, axis=-1,
                output_type=tf.int32)

        # Domain classification accuracy is always binary
        domain_acc_labels = tf.argmax(domain_labels, axis=-1,
            output_type=tf.int32)
        domain_acc_predictions = tf.argmax(domain_predictions_raw, axis=-1,
            output_type=tf.int32)

        # Overall metrics
        task_acc, update_task_acc, reset_task_acc = create_reset_metric(
            tf.metrics.accuracy, "task_acc",
            labels=acc_labels, predictions=acc_predictions)
        task_auc, update_task_auc, reset_task_auc = create_reset_metric(
            tf.metrics.auc, "task_auc",
            labels=task_labels, predictions=task_predictions_raw)
        domain_acc, update_domain_acc, reset_domain_acc = create_reset_metric(
            tf.metrics.accuracy, "domain_acc",
            labels=domain_acc_labels, predictions=domain_acc_predictions)

    reset_metrics = [reset_task_acc, reset_task_auc, reset_domain_acc]
    update_metrics = [update_task_acc, update_task_auc, update_domain_acc]

    for j, dataset in enumerate(datasets):
        summs[j] += [
            tf.summary.scalar("auc_task/%s/%s" % (domain, dataset), task_auc),
            tf.summary.scalar("accuracy_task/%s/%s" % (domain, dataset), task_acc),
            tf.summary.scalar("accuracy_domain/%s/%s" % (domain, dataset), domain_acc),
        ]

    # Per-class metrics
    for i in range(num_classes):
        class_name = config.int_to_label(i)

        with tf.variable_scope("metrics_%s/class_%s" % (domain,class_name)):
            # Get ith column (all groundtruth/predictions for ith class)
            class_y = tf.slice(
                task_labels, [0,i], [tf.shape(task_labels)[0], 1])
            class_predictions = tf.slice(
                per_class_predictions, [0,i], [tf.shape(task_labels)[0], 1])

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
            with tf.variable_scope("metrics_%s/class_%s/%s" % (domain,class_name,dataset)):
                acc, update_acc, reset_acc = create_reset_metric(
                    tf.metrics.accuracy, "acc_%d" % j,
                    labels=acc_class_y, predictions=acc_class_predictions)
                tp, update_TP, reset_TP = create_reset_metric(
                    tf.metrics.true_positives, "TP_%d" % j,
                    labels=class_y, predictions=class_predictions)
                fp, update_FP, reset_FP = create_reset_metric(
                    tf.metrics.false_positives, "FP_%d" % j,
                    labels=class_y, predictions=class_predictions)
                tn, update_TN, reset_TN = create_reset_metric(
                    tf.metrics.true_negatives, "TN_%d" % j,
                    labels=class_y, predictions=class_predictions)
                fn, update_FN, reset_FN = create_reset_metric(
                    tf.metrics.false_negatives, "FN_%d" % j,
                    labels=class_y, predictions=class_predictions)

            reset_metrics += [reset_acc, reset_TP, reset_FP, reset_TN, reset_FN]
            update_metrics += [update_acc, update_TP, update_FP, update_TN, update_FN]

            summs[j] += [
                tf.summary.scalar("accuracy_task_class_%s/%s/%s" % (class_name,domain,dataset), acc),
                tf.summary.scalar("rates_class_%s/TP/%s/%s" % (class_name,domain,dataset), tp),
                tf.summary.scalar("rates_class_%s/FP/%s/%s" % (class_name,domain,dataset), fp),
                tf.summary.scalar("rates_class_%s/TN/%s/%s" % (class_name,domain,dataset), tn),
                tf.summary.scalar("rates_class_%s/FN/%s/%s" % (class_name,domain,dataset), fn),
            ]

    return reset_metrics, update_metrics, summs

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
            summaries.append(tf.summary.histogram("{}-grad".format(var.name), grad))

    return update_step, summaries

def train(
        num_features, num_classes, x_dims,
        tfrecords_train_a, tfrecords_train_b,
        tfrecords_test_a, tfrecords_test_b,
        config,
        model_func=build_lstm,
        batch_size=2048,
        num_steps=100000,
        learning_rate=0.0003,
        lr_multiplier=1,
        dropout_keep_prob=0.8,
        units=100,
        use_feature_extractor=True,
        model_dir="models",
        log_dir="logs",
        model_save_steps=1000,
        log_save_steps=100,
        log_validation_accuracy_steps=2000,
        log_extra_save_steps=2000,
        adaptation=True,
        multi_class=False,
        bidirectional=False,
        class_weights=1.0,
        plot_gradients=False,
        min_domain_accuracy=0.60,
        max_examples=5000,
        max_plot_examples=100):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # For adaptation, we'll be concatenating together half source and half target
    # data, so to keep the batch_size about the same, we'll cut it in half
    if adaptation:
        batch_size = batch_size // 2

    # Input training data
    with tf.variable_scope("training_data_a"):
        input_fn_a, input_hook_a = _get_tfrecord_input_fn(
            tfrecords_train_a, batch_size, x_dims, num_classes)
        next_data_batch_a, next_labels_batch_a = input_fn_a()
    with tf.variable_scope("training_data_b"):
        input_fn_b, input_hook_b = _get_tfrecord_input_fn(
            tfrecords_train_b, batch_size, x_dims, num_classes)
        next_data_batch_b, next_labels_batch_b = input_fn_b()

    # Load all the test data in one batch (we'll assume test set is small for now)
    with tf.variable_scope("evaluation_data_a"):
        eval_input_fn_a, eval_input_hook_a = _get_tfrecord_input_fn(
            tfrecords_test_a, batch_size, x_dims, num_classes, evaluation=True)
        next_data_batch_test_a, next_labels_batch_test_a = eval_input_fn_a()
    with tf.variable_scope("evaluation_data_b"):
        eval_input_fn_b, eval_input_hook_b = _get_tfrecord_input_fn(
            tfrecords_test_b, batch_size, x_dims, num_classes, evaluation=True)
        next_data_batch_test_b, next_labels_batch_test_b = eval_input_fn_b()

    # Inputs
    keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob') # for dropout
    x = tf.placeholder(tf.float32, [None]+x_dims, name='x') # input data
    domain = tf.placeholder(tf.float32, [None, 2], name='domain') # which domain
    y = tf.placeholder(tf.float32, [None, num_classes], name='y') # class 1, 2, etc. one-hot
    training = tf.placeholder(tf.bool, name='training') # whether we're training (batch norm)
    grl_lambda = tf.placeholder_with_default(1.0, shape=(), name='grl_lambda') # gradient multiplier for GRL
    lr = tf.placeholder(tf.float32, (), name='learning_rate')

    # Source domain will be [[1,0], [1,0], ...] and target domain [[0,1], [0,1], ...]
    source_domain = domain_labels(0, batch_size)
    target_domain = domain_labels(1, batch_size)

    # Model, loss, feature extractor output -- e.g. using build_lstm or build_vrnn
    #
    # Optionally also returns additional summaries to log, e.g. loss components
    task_classifier, domain_classifier, total_loss, \
    feature_extractor, model_summaries, extra_model_outputs = \
        model_func(x, y, domain, grl_lambda, keep_prob, training,
            num_classes, num_features, adaptation, units, multi_class,
            bidirectional, class_weights, x_dims, use_feature_extractor)

    # Get variables of model - needed if we train in two steps
    variables = tf.trainable_variables()
    model_vars = [v for v in variables if '_model' in v.name]
    feature_extractor_vars = [v for v in variables if 'feature_extractor' in v.name]
    task_classifier_vars = [v for v in variables if 'task_classifier' in v.name]
    domain_classifier_vars = [v for v in variables if 'domain_classifier' in v.name]

    # If the discriminator has lower than a certain accuracy on classifying which
    # domain a feature output was from, then we'll train that and skip training
    # the rest of the model. Thus, we need a quick way to calculate this for
    # the training batch. For evaluation I'll use tf.metrics but for this during
    # training I'll compute manually for simplicity.
    with tf.variable_scope("domain_accuracy"):
        domain_accuracy = tf.reduce_mean(tf.cast(tf.equal(
                tf.argmax(domain, axis=-1),
                tf.argmax(domain_classifier, axis=-1)),
            tf.float32))

    # Optimizer - update ops for batch norm layers
    with tf.variable_scope("optimizer"), \
        tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(lr)

        if plot_gradients:
            train_all, grad_summs = opt_with_summ(optimizer, total_loss)
        else:
            train_all = optimizer.minimize(total_loss)

        train_notdomain = optimizer.minimize(total_loss,
            var_list=model_vars+feature_extractor_vars+task_classifier_vars)

        if adaptation:
            train_domain = optimizer.minimize(total_loss,
                var_list=domain_classifier_vars)

    # Summaries - training and evaluation for both domains A and B
    #
    # Most will be using tf.metrics... as updatable/resetable. Thus, we'll first
    # run reset_metrics followed by the update_metrics_{a,b} (optionally over
    # multiple batches, e.g. if for the entire validation dataset). Then we
    # run and log the summaries.
    train_a_summs = [tf.summary.scalar("loss/total_loss", total_loss)]

    reset_a, update_metrics_a, summs = metric_summaries(
        "source", y, task_classifier, domain, domain_classifier,
        num_classes, multi_class, config)
    train_a_summs += summs[0]
    val_a_summs = summs[1]

    reset_b, update_metrics_b, summs = metric_summaries(
        "target", y, task_classifier, domain, domain_classifier,
        num_classes, multi_class, config)
    train_b_summs = summs[0]
    val_b_summs = summs[1]

    reset_metrics = reset_a + reset_b

    # If we want to plot gradients, include both the training summaries and
    # gradient summaries
    if plot_gradients:
        training_summaries_a = tf.summary.merge(train_a_summs+grad_summs)
    else:
        training_summaries_a = tf.summary.merge(train_a_summs)

    training_summaries_extra_a = tf.summary.merge(model_summaries)
    training_summaries_b = tf.summary.merge(train_b_summs)
    validation_summaries_a = tf.summary.merge(val_a_summs)
    validation_summaries_b = tf.summary.merge(val_b_summs)

    # Allow restoring global_step from past run
    global_step = tf.Variable(0, name="global_step", trainable=False)
    inc_global_step = tf.assign_add(global_step, 1, name='incr_global_step')

    # Keep track of state and summaries
    saver = tf.train.Saver(max_to_keep=num_steps)
    saver_hook = tf.train.CheckpointSaverHook(model_dir,
        save_steps=model_save_steps, saver=saver)
    writer = tf.summary.FileWriter(log_dir)

    # Allow running two at once
    # https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2

    # Start training
    with tf.train.SingularMonitoredSession(checkpoint_dir=model_dir, hooks=[
                input_hook_a, input_hook_b,
                saver_hook
            ], config=config) as sess:

        for i in range(sess.run(global_step),num_steps+1):
            if i == 0:
                writer.add_graph(sess.graph)

            # GRL schedule and learning rate schedule from DANN paper
            grl_lambda_value = 2/(1+np.exp(-10*(i/(num_steps+1))))-1
            #lr_value = 0.001/(1.0+10*i/(num_steps+1))**0.75
            lr_value = learning_rate

            t = time.time()
            step = sess.run(inc_global_step)

            # Get data for this iteration
            data_batch_a, labels_batch_a, data_batch_b, labels_batch_b = sess.run([
                next_data_batch_a, next_labels_batch_a,
                next_data_batch_b, next_labels_batch_b,
            ])

            if adaptation:
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
                    keep_prob: dropout_keep_prob, lr: lr_value, training: True
                })

                # Update domain more
                #
                # Depending on the num_steps, your learning rate, etc. it may be
                # beneficial to have a different learning rate here -- hence the
                # lr_multiplier option. This may also depend on your dataset though.
                sess.run(train_domain, feed_dict={
                    x: combined_x, y: combined_labels, domain: combined_domain,
                    grl_lambda: 0.0,
                    keep_prob: dropout_keep_prob, lr: lr_multiplier*lr_value, training: True
                })

                # Train only the domain predictor / discriminator
                # feed_dict={
                #     x: combined_x, domain: combined_domain,
                #     keep_prob: dropout_keep_prob, lr: lr_multiplier*lr_value, training: True
                # }
                # sess.run(train_domain, feed_dict=feed_dict)
                # domain_acc = sess.run(domain_accuracy, feed_dict=feed_dict)

                # print("Iteration", i, "domain acc", domain_acc)

                # # Don't update the rest of the model if the discriminator is
                # # still really bad -- otherwise its gradient may not be helpful
                # # for adaptation
                # if domain_acc > min_domain_accuracy:
                #     print("    Training rest of model")
                #     # Train everything except the domain predictor / discriminator but
                #     # flip labels rather than using GRL (i.e. set lambda=-1)
                #     sess.run(train_notdomain, feed_dict={
                #         x: combined_x, y: combined_labels, domain: combined_domain_flip,
                #         keep_prob: dropout_keep_prob, lr: lr_value, training: True
                #     })
            else:
                # Train task classifier on source domain to be correct
                sess.run(train_notdomain, feed_dict={
                    x: data_batch_a, y: labels_batch_a,
                    keep_prob: dropout_keep_prob, lr: lr_value, training: True
                })

            t = time.time() - t

            if i%log_save_steps == 0:
                # Log the step time
                summ = tf.Summary(value=[
                    tf.Summary.Value(tag="step_time/model_train", simple_value=t)
                ])
                writer.add_summary(summ, step)

                t = time.time()

                # Log summaries run on the training data
                #
                # Reset metrics, update metrics, then generate summaries
                feed_dict = {
                    x: data_batch_a, y: labels_batch_a, domain: source_domain,
                    keep_prob: 1.0, training: False
                }
                sess.run(reset_metrics)
                sess.run(update_metrics_a, feed_dict=feed_dict)
                summ = sess.run(training_summaries_a, feed_dict=feed_dict)
                writer.add_summary(summ, step)

                feed_dict = {
                    x: data_batch_b, y: labels_batch_b, domain: target_domain,
                    keep_prob: 1.0, training: False
                }
                sess.run(update_metrics_b, feed_dict=feed_dict)
                summ = sess.run(training_summaries_b, feed_dict=feed_dict)
                writer.add_summary(summ, step)

                t = time.time() - t

                # Log the time to update metrics
                summ = tf.Summary(value=[
                    tf.Summary.Value(tag="step_time/metrics/training", simple_value=t)
                ])
                writer.add_summary(summ, step)

            # Log validation accuracy/AUC less frequently
            if i%log_validation_accuracy_steps == 0:
                t = time.time()

                # Evaluation accuracy, AUC, rates, etc.
                sess.run(reset_metrics)
                update_metrics_on_val(sess,
                    eval_input_hook_a, eval_input_hook_b,
                    next_data_batch_test_a, next_labels_batch_test_a,
                    next_data_batch_test_b, next_labels_batch_test_b,
                    x, y, domain, keep_prob, training,
                    batch_size, update_metrics_a, update_metrics_b,
                    max_examples)

                # Add the summaries about rates that were updated above in the
                # evaluation function (via update_metrics_* lists)
                summs_a, summs_b = sess.run([
                    validation_summaries_a, validation_summaries_b])
                writer.add_summary(summs_a, step)
                writer.add_summary(summs_b, step)

                t = time.time() - t

                # Log the time to update metrics
                summ = tf.Summary(value=[
                    tf.Summary.Value(tag="step_time/metrics/validation", simple_value=t)
                ])
                writer.add_summary(summ, step)

            # Larger stuff like weights and t-SNE plots occasionally
            if i%log_extra_save_steps == 0:
                t = time.time()

                # Training weights
                summ = sess.run(training_summaries_extra_a, feed_dict={
                    x: data_batch_a, y: labels_batch_a, domain: source_domain,
                    keep_prob: 1.0, training: False
                })
                writer.add_summary(summ, step)

                t = time.time() - t

                # Log the time to update metrics
                summ = tf.Summary(value=[
                    tf.Summary.Value(tag="step_time/extra", simple_value=t)
                ])
                writer.add_summary(summ, step)

                t = time.time()

                # t-SNE, PCA, and VRNN reconstruction plots
                first_step = i==0 # only plot real ones once
                plots = evaluation_plots(sess,
                    eval_input_hook_a, eval_input_hook_b,
                    next_data_batch_test_a, next_labels_batch_test_a,
                    next_data_batch_test_b, next_labels_batch_test_b,
                    feature_extractor, x, keep_prob, training, adaptation,
                    extra_model_outputs, num_features, first_step, config,
                    max_plot_examples)

                for name, buf in plots:
                    # See: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
                    plot = tf.Summary.Image(encoded_image_string=buf)
                    summ = tf.Summary(value=[tf.Summary.Value(
                        tag=name, image=plot)])
                    writer.add_summary(summ, step)

                t = time.time() - t

                # Log the time to update metrics
                summ = tf.Summary(value=[
                    tf.Summary.Value(tag="step_time/plots", simple_value=t)
                ])
                writer.add_summary(summ, step)

                # Make sure we write to disk before too long so we can monitor live in
                # TensorBoard. If it's too delayed we won't be able to detect problems
                # for a long time.
                writer.flush()

        writer.flush()

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', default="models", type=str,
        help="Directory for saving model files")
    parser.add_argument('--logdir', default="logs", type=str,
        help="Directory for saving log files")
    parser.add_argument('--da', dest='adaptation', action='store_true',
        help="Perform domain adaptation on the model")
    parser.add_argument('--lstm', dest='lstm', action='store_true',
        help="Use LSTM model")
    parser.add_argument('--vrnn', dest='vrnn', action='store_true',
        help="Use VRNN model")
    parser.add_argument('--cnn', dest='cnn', action='store_true',
        help="Use CNN model")
    parser.add_argument('--resnet', dest='resnet', action='store_true',
        help="Use resnet model")
    parser.add_argument('--attention', dest='attention', action='store_true',
        help="Use attention model")
    parser.add_argument('--tcn', dest='tcn', action='store_true',
        help="Use TCN model")
    parser.add_argument('--flat', dest='flat', action='store_true',
        help="Use flat model, i.e. only flatten input fed to feature extractor")
    parser.add_argument('--fold', default=0, type=int,
        help="What fold to use from the dataset files (default fold 0)")
    parser.add_argument('--target', default="hh101", type=str,
        help="What dataset to use as the target (default \"hh101\")")
    parser.add_argument('--features', default="simple", type=str,
        help="Whether to use \"al\" or \"simple\" features (default \"simple\")")
    parser.add_argument('--units', default=100, type=int,
        help="Number of LSTM hidden units and VRNN latent variable size (default 100)")
    parser.add_argument('--steps', default=100000, type=int,
        help="Number of training steps to run (default 100000)")
    parser.add_argument('--batch', default=2048, type=int,
        help="Batch size to use (default 128, decrease if you run out of memory)")
    parser.add_argument('--lr', default=0.0003, type=float,
        help="Learning rate for training (default 0.0003)")
    parser.add_argument('--lr-mult', default=1.0, type=float,
        help="Multiplier for extra discriminator training learning rate (default 1.0)")
    parser.add_argument('--dropout', default=0.8, type=float,
        help="Keep probability for dropout (default 0.8)")
    parser.add_argument('--model-steps', default=4000, type=int,
        help="Save the model every so many steps (default 4000)")
    parser.add_argument('--log-steps', default=500, type=int,
        help="Log training losses and accuracy every so many steps (default 500)")
    parser.add_argument('--log-steps-val', default=4000, type=int,
        help="Log validation accuracy and AUC every so many steps (default 4000)")
    parser.add_argument('--log-steps-extra', default=50000, type=int,
        help="Log weights, plots, etc. every so many steps (default 50000)")
    parser.add_argument('--max-examples', default=5000, type=int,
        help="Max number of examples to evaluate for validation (default 5000)")
    parser.add_argument('--max-plot-examples', default=100, type=int,
        help="Max number of examples to use for plotting (default 100)")
    parser.add_argument('--balance', dest='balance', action='store_true',
        help="On high class imbalances weight the loss function (default)")
    parser.add_argument('--no-balance', dest='balance', action='store_false',
        help="Do not weight loss function with high class imbalances")
    parser.add_argument('--sample', dest='sample', action='store_true',
        help="Only use a small amount of data for training/testing")
    parser.add_argument('--balance-pow', default=1.0, type=float,
        help="For increased balancing, raise weights to a specified power (default 1.0)")
    parser.add_argument('--bidirectional', dest='bidirectional', action='store_true',
        help="Use a bidirectional RNN (when selected method includes an RNN)")
    parser.add_argument('--feature-extractor', dest='feature_extractor', action='store_true',
        help="Use a feature extractor before task classifier/domain predictor (default)")
    parser.add_argument('--no-feature-extractor', dest='feature_extractor', action='store_false',
        help="Do not use a feature extractor")
    parser.add_argument('--debug', dest='debug', action='store_true',
        help="Start new log/model/images rather than continuing from previous run")
    parser.add_argument('--debug-num', default=-1, type=int,
        help="Specify exact log/model/images number to use rather than incrementing from last. " \
            +"(Don't pass both this and --debug at the same time.)")
    parser.set_defaults(
        lstm=False, vrnn=False, cnn=False, tcn=False, flat=False,
        resnet=False, attention=False,
        adaptation=False, balance=True, sample=False,
        bidirectional=False, feature_extractor=True, debug=False)
    args = parser.parse_args()

    # Load datasets
    al_config = ALConfig()
    tfrecord_config = TFRecordConfig(args.features)
    assert len(al_config.labels) == tfrecord_config.num_classes, \
        "al.config and tfrecord config disagree on the number of classes: " \
        +str(len(al_config.labels))+" vs. "+str(tfrecord_config.num_classes)

    tfrecords_train_a, tfrecords_train_b, \
    tfrecords_test_a, tfrecords_test_b = \
        get_tfrecord_datasets(args.features, args.target, args.fold, args.sample)

    # Calculate class imbalance using labeled training data (i.e. from domain A)
    if args.balance:
        class_weights = calc_class_weights(tfrecords_train_a,
            tfrecord_config.x_dims, tfrecord_config.num_classes,
            args.balance_pow)
    else:
        class_weights = 1.0

    # Train with desired method
    assert args.lstm + args.vrnn + args.cnn + args.tcn + args.flat + args.resnet == 1, \
        "Must specify exactly one method to run"

    prefix = args.target+"-"

    if args.lstm:
        prefix += "lstm"
        model_func = build_lstm
    elif args.vrnn:
        prefix += "vrnn"
        model_func = build_vrnn
    elif args.cnn:
        prefix += "cnn"
        model_func = build_cnn
    elif args.resnet:
        prefix += "resnet"
        model_func = build_resnet
    elif args.attention:
        prefix += "attention"
        model_func = build_attention
    elif args.tcn:
        prefix += "tcn"
        model_func = build_tcn
    elif args.flat:
        prefix += "flat"
        model_func = build_flat

    if args.adaptation:
        prefix += "-da"

    # Use the number specified on the command line (higher precidence than --debug)
    if args.debug_num >= 0:
        attempt = args.debug_num
        print("Debugging attempt:", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(args.modeldir, prefix)
        log_dir = os.path.join(args.logdir, prefix)
    # Find last one, increment number
    elif args.debug:
        attempt = last_modified_number(args.logdir, prefix+"*")
        attempt = attempt+1 if attempt is not None else 1
        print("Debugging attempt:", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(args.modeldir, prefix)
        log_dir = os.path.join(args.logdir, prefix)
    # If no debugging modes, use the model and log directory as-is
    else:
        model_dir = args.modeldir
        log_dir = args.logdir

    train(tfrecord_config.num_features,
            tfrecord_config.num_classes,
            tfrecord_config.x_dims,
            tfrecords_train_a, tfrecords_train_b,
            tfrecords_test_a, tfrecords_test_b,
            al_config,
            model_func=model_func,
            model_dir=model_dir,
            log_dir=log_dir,
            adaptation=args.adaptation,
            num_steps=args.steps,
            learning_rate=args.lr,
            lr_multiplier=args.lr_mult,
            units=args.units,
            use_feature_extractor=args.feature_extractor,
            batch_size=args.batch,
            dropout_keep_prob=args.dropout,
            model_save_steps=args.model_steps,
            log_save_steps=args.log_steps,
            log_validation_accuracy_steps=args.log_steps_val,
            log_extra_save_steps=args.log_steps_extra,
            multi_class=False,
            bidirectional=args.bidirectional,
            class_weights=class_weights,
            max_examples=args.max_examples,
            max_plot_examples=args.max_plot_examples)
