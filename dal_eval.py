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

def run_eval(
        ckpt,
        num_features, num_classes, x_dims,
        tfrecords_train_a, tfrecords_train_b,
        tfrecords_test_a, tfrecords_test_b,
        config,
        model_func=build_lstm,
        batch_size=2048,
        units=100,
        use_feature_extractor=True,
        model_dir="models",
        log_dir="logs",
        adaptation=True,
        multi_class=False,
        bidirectional=False,
        class_weights=1.0,
        gpu_memory=0.8,
        max_examples=0): # 0 = all examples
    # Input training data
    with tf.variable_scope("training_data_a"):
        input_fn_a, input_hook_a = _get_tfrecord_input_fn(
            tfrecords_train_a, batch_size, x_dims, num_classes, evaluation=True)
        next_data_batch_a, next_labels_batch_a = input_fn_a()
    with tf.variable_scope("training_data_b"):
        input_fn_b, input_hook_b = _get_tfrecord_input_fn(
            tfrecords_train_b, batch_size, x_dims, num_classes, evaluation=True)
        next_data_batch_b, next_labels_batch_b = input_fn_b()

    # Load all the test data in one batch
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

    # Model, loss, feature extractor output -- e.g. using build_lstm or build_vrnn
    task_classifier, domain_classifier, _, \
    _, _, _ = \
        model_func(x, y, domain, grl_lambda, keep_prob, training,
            num_classes, num_features, adaptation, units, multi_class,
            bidirectional, class_weights, x_dims, use_feature_extractor)

    # Summaries - training and evaluation for both domains A and B
    reset_a, update_metrics_a, _, source_summs_values = metric_summaries(
        "source", y, task_classifier, domain, domain_classifier,
        num_classes, multi_class, config)

    reset_b, update_metrics_b, _, target_summs_values = metric_summaries(
        "target", y, task_classifier, domain, domain_classifier,
        num_classes, multi_class, config)

    reset_metrics = reset_a + reset_b

    # Keep track of state and summaries
    saver = tf.train.Saver()

    # Allow running two at once
    # https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_memory

    # Start training
    with tf.Session(config=config) as sess:
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

    return s_train, t_train, s_test, t_test

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

def process_model(model_dir, log_dir, features, target, fold, al_config,
        gpu_mem, bidirectional, batch, units, feature_extractor, adaptation,
        last):
    print("Processing target", target, "fold", fold, "using model", model)

    # In case a previous run still has the model in memory, reset graph.
    # This isn't a problem since each process only runs one model at a time.
    tf.reset_default_graph()

    tfrecords_train_a, tfrecords_train_b, \
    tfrecords_valid_a, tfrecords_valid_b, \
    tfrecords_test_a, tfrecords_test_b = \
        get_tfrecord_datasets(features, target, fold, False)

    # For training we split the train set into train/valid but now for comparison
    # with AL (that didn't make that split), we want to re-combine them so the
    # "train" set accuracies between AL and DAL are on the same data.
    tfrecords_train_a += tfrecords_valid_a
    tfrecords_train_b += tfrecords_valid_b

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

    # Open the log file generated during training, find the step with the
    # highest validation accuracy
    logfile = last_modified(log_dir, "*.tfevents*")

    task_accuracy = []
    for e in tf.train.summary_iterator(logfile):
        for v in e.summary.value:
            if v.tag == 'accuracy_task/source/validation':
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

    # Load corresponding checkpoint -- if it doesn't exist, then it probably
    # saved the iter-1 as a checkpoint. For example, most evaluations are at
    # iterations like 84001 but the checkpoint was at iteration 84000.
    ckpt = os.path.join(model_dir, "model.ckpt-"+str(max_accuracy_step))

    if not os.path.exists(ckpt+".index"):
        max_accuracy_step -= 1
        ckpt = os.path.join(model_dir, "model.ckpt-"+str(max_accuracy_step))

        assert os.path.exists(ckpt+".index"), \
            "could not find model checkpoint "+ckpt

    print("Loading checkpoint", max_accuracy_step, "with accuracy", max_accuracy)

    s_train, t_train, s_test, t_test = run_eval(ckpt,
            tfrecord_config.num_features,
            tfrecord_config.num_classes,
            tfrecord_config.x_dims,
            tfrecords_train_a, tfrecords_train_b,
            tfrecords_test_a, tfrecords_test_b,
            al_config,
            model_func=model_func,
            model_dir=model_dir,
            adaptation=adaptation,
            units=units,
            use_feature_extractor=feature_extractor,
            batch_size=batch,
            multi_class=False,
            bidirectional=bidirectional,
            class_weights=1.0,
            gpu_memory=gpu_mem)

    return target, fold, s_train, t_train, s_test, t_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', default="models", type=str,
        help="Directory for saving model files")
    parser.add_argument('--logdir', default="logs", type=str,
        help="Directory for saving log files")
    parser.add_argument('--features', default="simple", type=str,
        help="Whether to use \"al\" or \"simple\" features (default \"simple\")")
    parser.add_argument('--jobs', default=8, type=int,
        help="Number of TensorFlow jobs to run at once (default 8)")
    parser.add_argument('--last', dest='last', action='store_true',
        help="Use last model rather than one with best validation set performance")
    parser.add_argument('--units', default=100, type=int,
        help="Number of LSTM hidden units and VRNN latent variable size (default 100)")
    parser.add_argument('--batch', default=1024, type=int,
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

    print("Models found in", args.logdir)

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

        print("  target", target, "fold", fold, "using model", model)
        models_to_evaluate.append((str(log_dir), model_dir, target, model, fold, adaptation))

    gpu_mem = args.gpu_mem / args.jobs

    # Run in parallel
    commands = []

    for log_dir, model_dir, target, model, fold, adaptation in models_to_evaluate:
        commands.append((model_dir, log_dir, args.features, target, fold, al_config,
            gpu_mem, args.bidirectional, args.batch, args.units,
            args.feature_extractor, adaptation, args.last))

    results = run_job_pool(process_model, commands, cores=args.jobs)

    # Process results
    source_train = []
    source_test = []
    target_train = []
    target_test = []

    for target, fold, s_train, t_train, s_test, t_test in results:
        print("Target", target, "fold", fold)
        print(s_train)
        print(t_train)
        print(s_test)
        print(t_test)
        print()

        source_train.append(s_train)
        source_test.append(s_test)
        target_train.append(t_train)
        target_test.append(t_test)

    source_train = np.array(source_train)
    source_test = np.array(source_test)
    target_train = np.array(target_train)
    target_test = np.array(target_test)

    print()
    print("Averages over", len(source_train), "runs (each home is 3-fold CV)")
    print("Train A \t Avg:", source_train.mean(), "\t Std:", source_train.std())
    print("Test A  \t Avg:", source_test.mean(), "\t Std:", source_test.std())
    print("Train B \t Avg:", target_train.mean(), "\t Std:", target_train.std())
    print("Test B  \t Avg:", target_test.mean(), "\t Std:", target_test.std())
