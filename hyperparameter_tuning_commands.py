#!/usr/bin/env python3
"""
Generate a bunch of hyperparameter values that we'll then queue to run
"""
import math
import numpy as np

def generate_values(N):
    batch = 2**np.random.randint(7, 13, N) # 128 to 4096 by powers of 2
    #lr = np.around(10**(-np.random.uniform(2,5,size=N)), decimals=6)
    lr = 10.0**(-np.random.randint(3,6,N)) # 0.001 to 0.00001 by powers of 10
    balance = np.random.uniform(size=N) < 0.5 # True or False
    units = np.random.randint(1,21,size=N)*10 # 10 to 200 by 10's
    layers = np.random.randint(1,13,size=N) # 1 to 12
    dropout = 5*np.random.randint(10,21,N)/100 # 0.5 to 1.0 by 0.05's

    # Make commands
    trains = []
    evals = []

    for i in range(N):
        train, evaluate = output_command(batch[i], lr[i], balance[i], units[i],
            layers[i], dropout[i])
        trains.append(train)
        evals.append(evaluate)

    return trains, evals

def output_command(batch, lr, balance, units, layers, dropout):
    if balance:
        balance = "--balance"
        balance_short = "b"
    else:
        balance = "--no-balance"
        balance_short = "nb"

    name = "cv-" \
        + "b"+str(int(math.log2(batch))) + "-" \
        + "l"+str(int(-math.log10(lr))) + "-" \
        + balance_short + "-" \
        + "u"+str(units) + "-" \
        + "l"+str(layers) + "-" \
        + "d"+str(int(dropout*100))

    args = "--dataset=al.zip " \
        + "--features=al " \
        + "--units="+str(units) + " " \
        + "--layers="+str(layers)

    train_args = args + " " \
        + "--flat " \
        + "--batch="+str(batch) + " " \
        + "--lr=%.5f "%lr \
        + balance + " " \
        + "--dropout=%.2f"%dropout
    eval_args = args

    train = "./kamiak_queue_all.sh " + name + " " + train_args
    evaluate = "sbatch kamiak_eval.srun " + name + " " + eval_args

    return train, evaluate

if __name__ == "__main__":
    # Make repeatable
    np.random.seed(0)

    # Generate -- repeat in sets of 10 so we can get more in sets of 10.
    # Otherwise, if we for instance start with 40 and change to 50, then
    # all of the values change. If we change this to 4 then 5 though,
    # the first 40 stay the same and we just get another 10 at the end.
    trains = []
    evals = []
    for i in range(4):
        train, evaluate = generate_values(10)
        trains += train
        evals += evaluate

    # Output
    for t in trains:
        print(t)

    for e in evals:
        print(e)
