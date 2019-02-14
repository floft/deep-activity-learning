"""
Functions for pickling and unpickling data
"""
import os
import sys
import pickle

def save_pickle(filename, data, overwrite=False, debug=True):
    """
    Pickle some data so we don't have to rerun everything

    Usage:
        savePickle("data.pickle", (data1, data2))
    """
    if not overwrite and os.path.exists(filename):
        if debug:
            print("Skipping,", filename, "exists", file=sys.stderr)
    else:
        if debug:
            print("Pickling", filename, file=sys.stderr)

        if os.path.exists(filename):
            os.rename(filename, filename+".bak")

        try:
            with open(filename, 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        except Exception as error:
            print('Error saving', filename, ':', error, file=sys.stderr)

def load_pickle(filename, debug=True):
    """
    Load some data from a pickle file

    Usage:
        if os.path.exists("data.pickle"):
            data1, data2 = loadPickle("data.pickle")
    """
    if os.path.exists(filename):
        if debug:
            print("Loading", filename, file=sys.stderr)

        with open(filename, 'rb') as f:
            data = pickle.load(f)

        return data

    return None
