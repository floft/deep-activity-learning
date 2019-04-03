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
import multiprocessing
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from tensorflow.python.framework import config as tfconfig

from pool import run_job_pool
from models import DomainAdaptationModel
from metrics import Metrics
from checkpoints import CheckpointManager
from load_data import load_tfrecords, get_tfrecord_datasets, ALConfig, TFRecordConfig

FLAGS = flags.FLAGS

flags.DEFINE_string("modeldir", "models", "Directory for saving model files")
flags.DEFINE_string("logdir", "logs", "Directory for saving log files")
flags.DEFINE_enum("features", "al", ["al", "simple", "simple2"], "What type of features to use")
flags.DEFINE_integer("batch", 1024, "Batch size to use for evaluation")
flags.DEFINE_float("gpumem", 0.8, "Percentage of GPU memory to let TensorFlow use (divided among jobs)")
flags.DEFINE_string("match", "*-*-*", "String matching to determine which logs/models to process")
flags.DEFINE_integer("jobs", 4, "Number of TensorFlow jobs to run at once")
flags.DEFINE_integer("gpus", 1, "Split jobs between GPUs -- overrides jobs (1 == run multiple jobs on first GPU)")
flags.DEFINE_boolean("last", False, "Use last model rather than one with best validation set performance")

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

def get_models_to_evaluate():
    """
    Returns the models to evaluate based on what is in logdir and modeldir
    specified as command line arguments. The matching pattern is specified by
    the match argument.

    Returns: [(log_dir, model_dir, target, model_name, fold, adaptation,
        generalization), ...]
    """
    files = pathlib.Path(FLAGS.logdir).glob(FLAGS.match)
    models_to_evaluate = []

    for log_dir in files:
        items = str(log_dir.stem).split("-")
        assert len(items) >= 3 or len(items) <= 5, \
            "name should be one of target-foldX-model{-d{a,g}{-num,},-num,}"

        adaptation = False
        generalization = False

        if len(items) == 3:
            target, fold, model_name = items
        elif len(items) == 4 or len(items) == 5:
            target, fold, model_name, keyword = items[:4]

            if keyword == "da":
                adaptation = True
            elif keyword == "dg":
                generalization = True
            else:
                pass # probably a debug number, which we don't care about

        # Fold name is foldX, get just the X and convert to int
        fold = int(fold.replace("fold",""))

        model_dir = os.path.join(FLAGS.modeldir, log_dir.stem)
        assert os.path.exists(model_dir), "Model does not exist "+str(model_dir)

        models_to_evaluate.append((str(log_dir), model_dir, target, model_name,
            fold, adaptation, generalization))

    return models_to_evaluate

def print_results(results):
    """ Print out the accuracies on {Train,Test}{A,B} on each target-fold pair
    followed by the averages and standard deviations of these. """
    source_train = []
    source_test = []
    target_train = []
    target_test = []

    print("Target,Fold,Train A,Test A,Train B,Test B")
    for target, fold, s_train, t_train, s_test, t_test in results:
        if s_train is not None and s_test is not None:
            # If we don't have a target domain, just output zero
            if t_train is None:
                t_train = 0
            if t_test is None:
                t_test = 0

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

def process_model(log_dir, model_dir, target, model_name, fold, adaptation,
        generalization, al_config, tfrecord_config, multi_gpu):
    """ Evaluate a model on the train/test data and compute the results """
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
    num_classes = tfrecord_config.num_classes
    num_domains = tfrecord_config.num_domains
    input_shape = tfrecord_config.input_shape

    tfrecords_train_a, tfrecords_train_b, \
    tfrecords_valid_a, tfrecords_valid_b, \
    tfrecords_test_a, tfrecords_test_b = \
        get_tfrecord_datasets(FLAGS.features, target, fold, False)

    # For training we split the train set into train/valid but now for comparison
    # with AL (that didn't make that split), we want to re-combine them so the
    # "train" set accuracies between AL and DAL are on the same data.
    tfrecords_train_a += tfrecords_valid_a
    tfrecords_train_b += tfrecords_valid_b

    # Load datasets
    train_data_a = load_tfrecords(tfrecords_train_a, FLAGS.batch, input_shape,
        num_classes, num_domains, evaluation=True)
    train_data_b = load_tfrecords(tfrecords_train_b, FLAGS.batch, input_shape,
        num_classes, num_domains, evaluation=True)
    eval_data_a = load_tfrecords(tfrecords_test_a, FLAGS.batch, input_shape,
        num_classes, num_domains, evaluation=True)
    eval_data_b = load_tfrecords(tfrecords_test_b, FLAGS.batch, input_shape,
        num_classes, num_domains, evaluation=True)

    # Above we needed to load with the right number of num_domains, but for
    # adaptation, we only want two: source and target
    if not generalization:
        num_domains = 2

    # Build our model
    model = DomainAdaptationModel(num_classes, num_domains, model_name,
        generalization)

    # Load model from checkpoint
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = CheckpointManager(checkpoint, model_dir, log_dir)

    if FLAGS.last:
        checkpoint_manager.restore_latest()
        max_accuracy_step = checkpoint_manager.latest_step()
        max_accuracy = 0 # We don't really care...
    else:
        checkpoint_manager.restore_best()
        max_accuracy_step = checkpoint_manager.best_step()
        max_accuracy = checkpoint_manager.best_validation

    # Print which step we're loading the model for
    print(target+","+str(fold)+","+model_name+","+str(max_accuracy_step)+","+str(max_accuracy))

    # If not found, give up
    if not checkpoint_manager.found:
        return target, fold, None, None, None, None

    # Metrics
    have_target_domain = train_data_b is not None
    metrics = Metrics(log_dir, num_classes, num_domains, al_config,
        None, None, generalization, have_target_domain)

    # Evaluate on both datasets
    metrics.train(model, train_data_a, train_data_b, evaluation=True)
    metrics.test(model, eval_data_a, eval_data_b, evaluation=True)

    # Get results
    results = metrics.results()
    s_train = results["accuracy_task/source/training"]
    s_test = results["accuracy_task/source/validation"]

    if train_data_b is not None:
        t_train = results["accuracy_task/target/training"]
        t_test = results["accuracy_task/target/validation"]
    else:
        t_train = None
        t_test = None

    return target, fold, s_train, t_train, s_test, t_test

def main(argv):
    # Load datasets
    al_config = ALConfig()
    tfrecord_config = TFRecordConfig(FLAGS.features)
    assert len(al_config.labels) == tfrecord_config.num_classes, \
        "al.config and tfrecord config disagree on the number of classes: " \
        +str(len(al_config.labels))+" vs. "+str(tfrecord_config.num_classes)

    # If single GPU, then split memory between jobs. But, if multiple GPUs,
    # each GPU has its own memory, so don't divide it up.
    #
    # If multiple GPUs, the jobs are split by GPU not by the "jobs" argument, so
    # ignore it and just set jobs to the GPU count.
    if FLAGS.gpus == 1:
        jobs = FLAGS.jobs
        gpumem = FLAGS.gpumem / jobs
        multi_gpu = False
    else:
        jobs = FLAGS.gpus
        gpumem = FLAGS.gpumem
        multi_gpu = True

    # TODO does this carry over when the process is forked?
    tfconfig.set_gpu_per_process_memory_fraction(gpumem)

    # Find models in the model/log directories
    models_to_evaluate = get_models_to_evaluate()

    # Run in parallel
    commands = []

    for model_params in models_to_evaluate:
        commands.append((*model_params, al_config, tfrecord_config, multi_gpu))

    # Also prints which models we load
    print("Target,Fold,Model,Best Step,Accuracy at Step")
    results = run_job_pool(process_model, commands, cores=jobs)

    # Print results, averages, etc.
    print_results(results)

if __name__ == "__main__":
    app.run(main)
