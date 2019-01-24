#!/usr/bin/env python3
"""
Deep activity learning -- evaluation

This takes a model trained by dal.py and evaluates it on both:
    train/valid - combined this is the same as the AL train set
    test - this is the same as the AL test set

It'll output the {source,target}-{train,test} accuracies for comparison with AL.
"""
import os
import pathlib
import argparse
import numpy as np
import tensorflow as tf

from model import build_lstm, build_vrnn, build_cnn, build_tcn, build_flat, \
    build_resnet, build_attention
from load_data import _get_tfrecord_input_fn, get_tfrecord_datasets, \
    ALConfig, TFRecordConfig
from dal import last_modified_number, update_metrics_on_val, metric_summaries
from pool import run_job_pool

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

    task_accuracy = []
    for e in tf.train.summary_iterator(logfile):
        for v in e.summary.value:
            if v.tag == tag:
                task_accuracy.append((e.step, v.simple_value))

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

def process_model(model_dir, log_dir, model, features, target, fold, al_config,
        tfrecord_config, gpu_mem, bidirectional, batch_size, units,
        feature_extractor, adaptation, last,
        multi_class=False, class_weights=1.0, max_examples=0):
    num_features = tfrecord_config.num_features
    num_classes = tfrecord_config.num_classes
    x_dims = tfrecord_config.x_dims

    tfrecords_train_a, tfrecords_train_b, \
    tfrecords_valid_a, tfrecords_valid_b, \
    tfrecords_test_a, tfrecords_test_b = \
        get_tfrecord_datasets(features, target, fold, False)

    # For training we split the train set into train/valid but now for comparison
    # with AL (that didn't make that split), we want to re-combine them so the
    # "train" set accuracies between AL and DAL are on the same data.
    tfrecords_train_a += tfrecords_valid_a
    tfrecords_train_b += tfrecords_valid_b

    # Get the step at which we have the highest accuracy on the validation set
    # Unless, we want the last one, then just get that one.
    max_accuracy_step, max_accuracy = get_step_from_log(log_dir, last)

    # Get checkpoint file
    ckpt, max_accuracy_step = get_checkpoint(model_dir, max_accuracy_step)

    print(target+","+str(fold)+","+model+","+str(max_accuracy_step)+","+str(max_accuracy))

    # Load train dataset
    with tf.variable_scope("training_data_a"):
        input_fn_a, input_hook_a = _get_tfrecord_input_fn(
            tfrecords_train_a, batch_size, x_dims, num_classes, evaluation=True)
        next_data_batch_a, next_labels_batch_a = input_fn_a()
    with tf.variable_scope("training_data_b"):
        input_fn_b, input_hook_b = _get_tfrecord_input_fn(
            tfrecords_train_b, batch_size, x_dims, num_classes, evaluation=True)
        next_data_batch_b, next_labels_batch_b = input_fn_b()

    # Load test dataset
    with tf.variable_scope("evaluation_data_a"):
        eval_input_fn_a, eval_input_hook_a = _get_tfrecord_input_fn(
            tfrecords_test_a, batch_size, x_dims, num_classes, evaluation=True)
        next_data_batch_test_a, next_labels_batch_test_a = eval_input_fn_a()
    with tf.variable_scope("evaluation_data_b"):
        eval_input_fn_b, eval_input_hook_b = _get_tfrecord_input_fn(
            tfrecords_test_b, batch_size, x_dims, num_classes, evaluation=True)
        next_data_batch_test_b, next_labels_batch_test_b = eval_input_fn_b()

    # Only build graph in this process if we're the first run, i.e. if the graph
    # isn't already built. Alternatively we could reset the graph with
    # tf.reset_default_graph() and then recreate it.
    already_built = len(tf.trainable_variables()) > 0

    # We have no access to the process itself when running multiple jobs and the
    # variables are out of scope by the time we get here a second time. Thus,
    # make them global (per process) and then we can refer to the ones we
    # previously created (when already_built=True the graph already exists
    # in this process)
    global sess, keep_prob, x, domain, y, training, reset_metrics
    global update_metrics_a, update_metrics_b
    global source_summs_values, target_summs_values

    if not already_built:
        if model == "lstm":
            model_func = build_lstm
        elif model == "vrnn":
            model_func = build_vrnn
        elif model == "cnn":
            model_func = build_cnn
        elif model == "resnet":
            model_func = build_resnet
        elif model == "attention":
            model_func = build_attention
        elif model == "tcn":
            model_func = build_tcn
        elif model == "flat":
            model_func = build_flat
        else:
            raise NotImplementedError

        # Inputs
        keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob') # for dropout
        x = tf.placeholder(tf.float32, [None]+x_dims, name='x') # input data
        domain = tf.placeholder(tf.float32, [None, 2], name='domain') # which domain
        y = tf.placeholder(tf.float32, [None, num_classes], name='y') # class 1, 2, etc. one-hot
        training = tf.placeholder(tf.bool, name='training') # whether we're training (batch norm)
        grl_lambda = tf.placeholder_with_default(1.0, shape=(), name='grl_lambda') # gradient multiplier for GRL

        # Model, loss, feature extractor output -- e.g. using build_lstm or build_vrnn
        task_classifier, domain_classifier, _, \
        _, _, _ = \
            model_func(x, y, domain, grl_lambda, keep_prob, training,
                num_classes, num_features, adaptation, units, multi_class,
                bidirectional, class_weights, x_dims, feature_extractor)

        # Summaries - training and evaluation for both domains A and B
        reset_a, update_metrics_a, _, source_summs_values = metric_summaries(
            "source", y, task_classifier, domain, domain_classifier,
            num_classes, multi_class, al_config)

        reset_b, update_metrics_b, _, target_summs_values = metric_summaries(
            "target", y, task_classifier, domain, domain_classifier,
            num_classes, multi_class, al_config)

        reset_metrics = reset_a + reset_b

        # Allow running multiple at once
        # https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_mem

        # Don't keep recreating the session
        sess = tf.Session(config=config)

    # Load checkpoint
    saver = tf.train.Saver()
    saver.restore(sess, ckpt)

    # Train data
    sess.run(reset_metrics)
    update_metrics_on_val(sess,
        input_hook_a, input_hook_b,
        next_data_batch_a, next_labels_batch_a,
        next_data_batch_b, next_labels_batch_b,
        x, y, domain, keep_prob, training,
        batch_size, update_metrics_a, update_metrics_b,
        max_examples)

    s_train, t_train = sess.run([
        source_summs_values['accuracy_task/source/training'],
        target_summs_values['accuracy_task/target/training'],
    ])

    # Test data
    sess.run(reset_metrics)
    update_metrics_on_val(sess,
        eval_input_hook_a, eval_input_hook_b,
        next_data_batch_test_a, next_labels_batch_test_a,
        next_data_batch_test_b, next_labels_batch_test_b,
        x, y, domain, keep_prob, training,
        batch_size, update_metrics_a, update_metrics_b,
        max_examples)

    s_test, t_test = sess.run([
        source_summs_values['accuracy_task/source/validation'],
        target_summs_values['accuracy_task/target/validation'],
    ])

    # Note: ideally we'd do sess.close() but since we don't know when we're done
    # with the process (many times calling this per process) we don't know when
    # to close it.

    return target, fold, s_train, t_train, s_test, t_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', default="models", type=str,
        help="Directory for saving model files")
    parser.add_argument('--logdir', default="logs", type=str,
        help="Directory for saving log files")
    parser.add_argument('--features', default="simple", type=str,
        help="Whether to use \"al\" or \"simple\" features (default \"simple\")")
    parser.add_argument('--jobs', default=4, type=int,
        help="Number of TensorFlow jobs to run at once (default 4)")
    parser.add_argument('--last', dest='last', action='store_true',
        help="Use last model rather than one with best validation set performance")
    parser.add_argument('--units', default=100, type=int,
        help="Number of LSTM hidden units and VRNN latent variable size (default 100)")
    parser.add_argument('--batch', default=8192, type=int,
        help="Batch size to use (default 1024, decrease if you run out of memory)")
    parser.add_argument('--gpu-mem', default=0.7, type=float,
        help="Percentage of GPU memory to let TensorFlow use (default 0.7, divided among jobs)")
    parser.add_argument('--bidirectional', dest='bidirectional', action='store_true',
        help="Use a bidirectional RNN (when selected method includes an RNN)")
    parser.add_argument('--feature-extractor', dest='feature_extractor', action='store_true',
        help="Use a feature extractor before task classifier/domain predictor (default)")
    parser.add_argument('--no-feature-extractor', dest='feature_extractor', action='store_false',
        help="Do not use a feature extractor")
    parser.set_defaults(bidirectional=False, feature_extractor=True, debug=False, last=False)
    args = parser.parse_args()

    # Load datasets
    al_config = ALConfig()
    tfrecord_config = TFRecordConfig(args.features)
    assert len(al_config.labels) == tfrecord_config.num_classes, \
        "al.config and tfrecord config disagree on the number of classes: " \
        +str(len(al_config.labels))+" vs. "+str(tfrecord_config.num_classes)

    files = pathlib.Path(args.logdir).glob("*-*-*")
    models_to_evaluate = []

    for log_dir in files:
        items = str(log_dir.stem).split("-")
        assert len(items) == 3 or len(items) == 4, \
            "name should be target-model-fold or target-model-da-fold"

        if len(items) == 3:
            target, model, fold = items
            adaptation = False
        else:
            target, model, _, fold = items
            adaptation = True

        model_dir = os.path.join(args.modeldir, log_dir.stem)
        assert os.path.exists(model_dir), "Model does not exist "+str(model_dir)

        models_to_evaluate.append((str(log_dir), model_dir, target, model, fold, adaptation))

    gpu_mem = args.gpu_mem / args.jobs

    # Run in parallel
    commands = []
    last_model = None

    for log_dir, model_dir, target, model, fold, adaptation in models_to_evaluate:
        assert last_model is None or last_model == model, \
            "all must use same model but one was "+last_model+" and another "+model
        commands.append((model_dir, log_dir, model, args.features, target, fold,
            al_config, tfrecord_config, gpu_mem, args.bidirectional, args.batch,
            args.units, args.feature_extractor, adaptation, args.last))

    print("Target,Fold,Model,Best Step,Accuracy at Step")
    results = run_job_pool(process_model, commands, cores=args.jobs)
    print()

    # Process results
    source_train = []
    source_test = []
    target_train = []
    target_test = []

    print("Target,Fold,Train A,Test A,Train B,Test B")
    for target, fold, s_train, t_train, s_test, t_test in results:
        print(target+","+fold+","+ \
            str(s_train)+","+str(s_test)+","+ \
            str(t_train)+","+str(t_test))

        source_train.append(s_train)
        source_test.append(s_test)
        target_train.append(t_train)
        target_test.append(t_test)

    source_train = np.array(source_train)
    source_test = np.array(source_test)
    target_train = np.array(target_train)
    target_test = np.array(target_test)

    print()
    print()
    print("Dataset,Avg,Std")
    print("Train A,"+str(source_train.mean())+","+str(source_train.std()))
    print("Test A,"+str(source_test.mean())+","+str(source_test.std()))
    print("Train B,"+str(target_train.mean())+","+str(target_train.std()))
    print("Test B,"+str(target_test.mean())+","+str(target_test.std()))

    print()
    print()
    print("Averages over", len(source_train), "runs (each home is 3-fold CV)")
    print("Train A \t Avg:", source_train.mean(), "\t Std:", source_train.std())
    print("Test A  \t Avg:", source_test.mean(), "\t Std:", source_test.std())
    print("Train B \t Avg:", target_train.mean(), "\t Std:", target_train.std())
    print("Test B  \t Avg:", target_test.mean(), "\t Std:", target_test.std())
