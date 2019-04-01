#!/usr/bin/env python3
"""
Deep activity learning -- evaluation

This takes a model trained by dal.py and evaluates it on both:
    train/valid - combined this is the same as the AL train set
    test - this is the same as the AL test set

It'll output the {source,target}-{train,test} accuracies for comparison with AL.
"""
import os
import sys
import pathlib
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf

from model import build_lstm, build_vrnn, build_cnn, build_tcn, build_flat, \
    build_resnet, build_attention
from load_data import _get_tfrecord_input_fn, get_tfrecord_datasets, \
    ALConfig, TFRecordConfig
from dal import last_modified_number, update_metrics_on_val, metric_summaries
from pool import run_job_pool
from eval_utils import get_step_from_log, get_checkpoint

def get_gpus():
    """
    Get the list of GPU ID's that SLURM is giving us
    """
    return [int(x) for x in os.getenv("SLURM_JOB_GPUS", "").split(",")]

def get_pool_id():
    """
    Get unique ID for this process in the job pool. It'll range from
    1 to max_jobs. See: https://stackoverflow.com/a/10192611/2698494

    Will return a number in [0,max_jobs)
    """
    current = multiprocessing.current_process()
    return current._identity[0]-1

def process_model(model_dir, log_dir, model, features, target, fold, al_config,
        tfrecord_config, gpu_mem, bidirectional, batch_size, units, layers,
        feature_extractor, adaptation, generalization, last, multi_gpu=False,
        multi_class=False, class_weights=1.0, max_examples=0):
    # Get what GPU to run this on
    if multi_gpu:
        # Get all GPUs SLURM gave to us and what process in the pool this is
        available_gpus = get_gpus()
        pool_id = get_pool_id()

        # Pick which one based on pool id
        gpu = available_gpus[pool_id]

        # Only let TensorFlow see this GPU. I tried tf.device, but somehow
        # each process still put some stuff into memory on every GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        # By default (if only using one GPU) the first one
        gpu = 0

    # Get data
    num_features = tfrecord_config.num_features
    num_classes = tfrecord_config.num_classes
    num_domains = tfrecord_config.num_domains
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

    # If no log file.... give up
    if max_accuracy_step is None:
        return target, fold, None, None, None, None

    # Get checkpoint file
    ckpt, max_accuracy_step = get_checkpoint(model_dir, max_accuracy_step)

    print(target+","+str(fold)+","+model+","+str(max_accuracy_step)+","+str(max_accuracy))

    # Load train dataset
    with tf.compat.v1.variable_scope("training_data_a"):
        input_fn_a, input_hook_a = _get_tfrecord_input_fn(
            tfrecords_train_a, batch_size, x_dims, num_classes, num_domains,
            evaluation=True)
        next_data_batch_a, next_labels_batch_a, next_domains_batch_a = input_fn_a()
    with tf.compat.v1.variable_scope("training_data_b"):
        input_fn_b, input_hook_b = _get_tfrecord_input_fn(
            tfrecords_train_b, batch_size, x_dims, num_classes, num_domains,
            evaluation=True)
        next_data_batch_b, next_labels_batch_b, next_domains_batch_b = input_fn_b()

    # Load test dataset
    with tf.compat.v1.variable_scope("evaluation_data_a"):
        eval_input_fn_a, eval_input_hook_a = _get_tfrecord_input_fn(
            tfrecords_test_a, batch_size, x_dims, num_classes, num_domains,
            evaluation=True)
        next_data_batch_test_a, next_labels_batch_test_a, \
            next_domains_batch_test_a = eval_input_fn_a()
    with tf.compat.v1.variable_scope("evaluation_data_b"):
        eval_input_fn_b, eval_input_hook_b = _get_tfrecord_input_fn(
            tfrecords_test_b, batch_size, x_dims, num_classes, num_domains,
            evaluation=True)
        next_data_batch_test_b, next_labels_batch_test_b, \
            next_domains_batch_test_b = eval_input_fn_b()

    # Above we needed to load with the right number of num_domains, but for
    # adaptation, we only want two: source and target
    if not generalization:
        num_domains = 2

    # Only build graph in this process if we're the first run, i.e. if the graph
    # isn't already built. Alternatively we could reset the graph with
    # tf.reset_default_graph() and then recreate it.
    already_built = len(tf.compat.v1.trainable_variables()) > 0

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
            raise NotImplementedError()

        # Inputs
        keep_prob = tf.compat.v1.placeholder_with_default(1.0, shape=(), name='keep_prob') # for dropout
        x = tf.compat.v1.placeholder(tf.float32, [None]+x_dims, name='x') # input data
        domain = tf.compat.v1.placeholder(tf.float32, [None, num_domains], name='domain') # which domain
        y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name='y') # class 1, 2, etc. one-hot
        training = tf.compat.v1.placeholder(tf.bool, name='training') # whether we're training (batch norm)
        grl_lambda = tf.compat.v1.placeholder_with_default(1.0, shape=(), name='grl_lambda') # gradient multiplier for GRL

        # Model, loss, feature extractor output -- e.g. using build_lstm or build_vrnn
        task_classifier, domain_classifier, _, \
        _, _, _ = \
            model_func(x, y, domain, grl_lambda, keep_prob, training,
                num_classes, num_domains, num_features, adaptation, generalization,
                units, layers, multi_class,
                bidirectional, class_weights, x_dims, feature_extractor)

        # Summaries - training and evaluation for both domains A and B
        reset_a, update_metrics_a, _, source_summs_values = metric_summaries(
            "source", y, task_classifier, domain, domain_classifier,
            num_classes, multi_class, al_config)

        reset_b, update_metrics_b, _, target_summs_values = metric_summaries(
            "target", y, task_classifier, domain, domain_classifier,
            num_classes, multi_class, al_config)

        reset_metrics = reset_a + reset_b

        config = tf.compat.v1.ConfigProto()
        # https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth
        config.gpu_options.per_process_gpu_memory_fraction = gpu_mem

        # Don't keep recreating the session
        sess = tf.compat.v1.Session(config=config)

    # Load checkpoint
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess, ckpt)

    # Train data
    sess.run(reset_metrics)
    update_metrics_on_val(sess,
        input_hook_a, input_hook_b,
        next_data_batch_a, next_labels_batch_a,
        next_data_batch_b, next_labels_batch_b,
        next_domains_batch_a, next_domains_batch_b,
        x, y, domain, keep_prob, training, num_domains, generalization,
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
        next_domains_batch_test_a, next_domains_batch_test_b,
        x, y, domain, keep_prob, training, num_domains, generalization,
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
    parser.add_argument('--match', default="*-*-*", type=str,
        help="String matching to determine which logs/models to process (default *-*-*)")
    parser.add_argument('--features', default="al", type=str,
        help="Whether to use \"al\", \"simple\", or \"simple2\" features (default \"al\")")
    parser.add_argument('--jobs', default=4, type=int,
        help="Number of TensorFlow jobs to run at once (default 4)")
    parser.add_argument('--gpus', default=1, type=int,
        help="Split jobs between GPUs -- overrides jobs (default 1, run multiple jobs on first GPU)")
    parser.add_argument('--last', dest='last', action='store_true',
        help="Use last model rather than one with best validation set performance")
    parser.add_argument('--units', default=50, type=int,
        help="Number of LSTM hidden units and VRNN latent variable size (default 50)")
    parser.add_argument('--layers', default=5, type=int,
        help="Number of layers for the feature extractor (default 5)")
    parser.add_argument('--batch', default=16384, type=int,
        help="Batch size to use (default 16384, decrease if you run out of memory)")
    parser.add_argument('--gpu-mem', default=0.8, type=float,
        help="Percentage of GPU memory to let TensorFlow use (default 0.8, divided among jobs)")
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

    files = pathlib.Path(args.logdir).glob(args.match)
    models_to_evaluate = []

    for log_dir in files:
        items = str(log_dir.stem).split("-")
        assert len(items) >= 3 or len(items) <= 5, \
            "name should be one of target-foldX-model{-d{a,g}{-num,},-num,}"

        adaptation = False
        generalization = False

        if len(items) == 3:
            target, fold, model = items
        elif len(items) == 4 or len(items) == 5:
            target, fold, model, keyword = items[:4]

            if keyword == "da":
                adaptation = True
            elif keyword == "dg":
                generalization = True
            else:
                pass # probably a debug number, which we don't care about

        # Fold name is foldX, get just the X and convert to int
        fold = int(fold.replace("fold",""))

        model_dir = os.path.join(args.modeldir, log_dir.stem)
        assert os.path.exists(model_dir), "Model does not exist "+str(model_dir)

        models_to_evaluate.append((str(log_dir), model_dir, target, model, fold,
            adaptation, generalization))

    # If single GPU, then split memory between jobs. But, if multiple GPUs,
    # each GPU has its own memory, so don't divide it up.
    #
    # If multiple GPUs, the jobs are split by GPU not by the "jobs" argument, so
    # ignore it and just set jobs to the GPU count.
    if args.gpus == 1:
        gpu_mem = args.gpu_mem / args.jobs
        jobs = args.jobs
        multi_gpu = False
    else:
        gpu_mem = args.gpu_mem
        jobs = args.gpus
        multi_gpu = True

    # Run in parallel
    commands = []
    last_model = None

    for log_dir, model_dir, target, model, fold, adaptation, generalization in models_to_evaluate:
        assert last_model is None or last_model == model, \
            "all must use same model but one was "+last_model+" and another "+model
        commands.append((model_dir, log_dir, model, args.features, target, fold,
            al_config, tfrecord_config, gpu_mem, args.bidirectional, args.batch,
            args.units, args.layers, args.feature_extractor, adaptation,
            generalization, args.last, multi_gpu))

    print("Target,Fold,Model,Best Step,Accuracy at Step")
    results = run_job_pool(process_model, commands, cores=jobs)

    # Process results
    source_train = []
    source_test = []
    target_train = []
    target_test = []

    print("Target,Fold,Train A,Test A,Train B,Test B")
    for target, fold, s_train, t_train, s_test, t_test in results:
        if s_train is not None and t_train is not None \
            and s_test is not None and t_test is not None:
            print(target+","+str(fold)+","+ \
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

    if len(source_train) > 0 and len(source_test) > 0 \
            and len(target_train) > 0 and len(target_test) > 0:
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
    else:
        print("No data.")
