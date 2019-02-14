#!/usr/bin/env python3
"""
Hyperparameter tuning using Nevergrad (Facebook gradient-free optimization library)

Since Nevergrad performs minimization, we will chose to minimize:
    f(x) = 1 - max accuracy on validation data with hyperparameters x

To install (see https://github.com/facebookresearch/nevergrad):
    pip install --user nevergrad
"""
import os
import sys
import time
import random
import subprocess
import numpy as np
import nevergrad.optimization as optimization
from nevergrad import instrumentation as inst

from eval_utils import get_finished, get_best_valid_accuracy, get_last_int
from pickle_data import load_pickle, save_pickle
from hyperparameter_tuning_commands import output_command

class Network:
    """
    Run the neural net and return 1 - highest validation accuracy
    Note: this assumes that rsync is running in the background, see Sync()
    """
    def __init__(self, directory, instrum, params):
        args, _ = instrum.data_to_arguments(params, deterministic=True)
        self.directory = directory
        self.instrum = instrum
        self.params = params
        self.args = args
        self.exiting = False
        self.log_dir = None
        self.jobs = [] # slurm job IDs

    def ssh_command(self, command):
        """ Get the command for running something over SSH """
        return [
            "ssh", "kamiak",
            "cd "+self.directory+"; "+command
        ]

    def start(self):
        """ Start the training by submitting to the queue """
        # Figure out what neural network command to run
        name, train_command, _ = output_command(*self.args)

        # Run the command on Kamiak via ssh
        cmd = self.ssh_command(train_command)

        # Get the response to know what job number these are
        output = subprocess.check_output(cmd).decode("utf-8")

        for line in output:
            self.jobs.append(get_last_int(line))

        # Where the log files will be stored (synced with rsync, see Sync())
        self.log_dir = "kamiak-logs-" + name

    def run(self):
        """ Wait for the results """
        self.start()

        while not self.exiting:
            if self.finished():
                return self.result()
            else:
                time.sleep(5)

        return None

    def finished(self):
        """ Check if the file saying we're done exists """
        if self.log_dir is not None:
            return get_finished(self.log_dir)

        return False

    def result(self):
        """ Check the result in the best accuracy file """
        if self.log_dir is not None:
            best_accuracy = get_best_valid_accuracy(self.log_dir)

            if best_accuracy is not None:
                return 1.0 - best_accuracy

        return None

    def stop(self):
        """ Stop run() and cancel the slurm jobs """
        # Exit the run() function
        self.exiting = True

        # Exit the jobs
        cmd = self.ssh_command("scancel "+" ".join([str(j) for j in self.jobs]))
        output = subprocess.check_output(cmd)
        print(output, file=sys.stderr)

class Sync:
    """ Run rsync to sync the best_valid_accuracy.txt files """
    def __init__(self, directory):
        self.directory = directory
        self.p = None
        self.exiting = False

    def run(self):
        """ Continuously run rsync """
        while not self.exiting:
            # Run rsync
            self.start()

            if self.p is not None:
                self.p.wait()

            # Wait and then we'll run it again
            time.sleep(30)

    def start(self):
        """ Start rsync process """
        if self.p is None:
            cmd = [
                "rsync", "-Pahuv",
                "--include", "best_valid_accuracy.txt", # accuracy files
                "--include", "finished.txt", # is it done
                "--exclude", "*", # only included files
                self.directory, # from
                "./", # to
            ]
            print("Executing", cmd, file=sys.stderr)
            self.p = subprocess.Popen(cmd)

    def stop(self):
        """ Stop run() and the rsync process """
        self.exiting = True

        if self.p is not None:
            self.p.terminate()
            self.p.wait()
            self.p = None

def make_repeatable():
    """ Set random seeds for making this more repeatable """
    random.seed(1234)
    np.random.seed(1234)

def get_summary(instrum, params=None):
    """ Get human-readable arguments to run with given parameters"""
    # If None, then do default params
    if params is None:
        dim = instrum.dimension
        params = [0] * dim

    return instrum.get_summary(params)

def make_instrumentation(debug=False):
    """ Create the possibilities for all of our hyperparameters """
    # Possible argument values
    batch = inst.var.OrderedDiscrete([2**i for i in range(7,13)]) # 128 to 4096 by powers of 2
    lr = inst.var.OrderedDiscrete([10.0**(-i) for i in range(3,6)]) # 0.001 to 0.00001 by powers of 10
    balance = inst.var.OrderedDiscrete([True, False]) # boolean
    units = inst.var.OrderedDiscrete([i*10 for i in range(1,21)]) # 10 to 200 by 10's
    layers = inst.var.OrderedDiscrete(list(range(1,13))) # 1 to 12
    dropout = inst.var.OrderedDiscrete([5*i/100 for i in range(10,21)]) # 0.5 to 1.0 by 0.05's

    # Our "function" (neural net training with output of max validation accuracy)
    # is a function of the above hyperparameters
    instrum = inst.Instrumentation(batch, lr, balance, units, layers, dropout)

    if debug:
        # Make sure defaults are reasonable and in the middle
        print("Default values")
        print(get_summary(instrum))

    return instrum

def hyperparameter_tuning(tool="PortfolioDiscreteOnePlusOne", budget=40,
        num_workers=1, directory="/data/vcea/garrett.wilson/dal",
        pickle_file="nevergrad_optim.pickle", print_result=True):
    """
    Run hyperparameter tuning
    See: https://github.com/facebookresearch/nevergrad/blob/master/docs/machinelearning.md
    """
    instrum = make_instrumentation()

    # Optimization -- load if it exists, otherwise create it
    data = load_pickle(pickle_file)

    if data is not None:
        optim, _ = data

        # Update options if it changed
        optim.budget = budget
        optim.num_workers = num_workers
    else:
        optim = optimization.registry[tool](dimension=instrum.dimension,
            budget=budget, num_workers=num_workers)

    s = Sync(directory)

    try:
        s.run()
    except KeyboardInterrupt:
        s.stop()

    for _ in range(budget):
        params = optim.ask()

        try:
            n = Network(directory, instrum, params)
            result = n.run()
        except KeyboardInterrupt:
            n.stop()
            break

        if result is not None:
            optim.tell(params, result)

    # Save it
    recommendation = optim.provide_recommendation()
    save_pickle(pickle_file, (optim, recommendation), overwrite=True)

    # Get best recommended parameters
    if print_result:
        print("Recommendation")
        print(get_summary(instrum, recommendation))
        args, _ = instrum.data_to_arguments(recommendation, deterministic=True)
        _, train, _ = output_command(*args)
        print("Command:", train)

    return args

if __name__ == "__main__":
    make_repeatable()
    hyperparameter_tuning()
