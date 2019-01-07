#!/usr/bin/env python3
"""
Create simple feature vectors for a smart home data .txt file writing the results
as a .hdf5 file
"""
import sys
import h5py
import numpy as np

from pytz import timezone
from datetime import datetime

def load_config(filename):
    """ Gets the possible features and labels """
    config = { "features": None, "labels": None }

    with open(filename, 'r') as f:
        for line in f:
            items = line.strip().split(' ')

            if len(items) > 0:
                if items[0] == "sensors":
                    config["features"] = items[1:]
                elif items[0] == "activities":
                    config["labels"] = items[1:]

    return config

def label_to_int(config, label_name):
    """ e.g. Bathe to 0 """
    return config["labels"].index(label_name)

def feature_to_int(config, feature_name):
    """ e.g. Bathroom to 0 """
    return config["features"].index(feature_name)

def parse_datetime(dt, timezone=timezone("UTC")):
    """
    Load the date/time from a couple possible formats into a Python datetime
    object and set the desired timezone

    If it doesn't fit a known format, this will throw a ValueError.
    """
    # Date time -- format documentation http://strftime.org/
    # Some apparently don't have the decimal and microseconds
    try:
        dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    finally:
        dt = dt.replace(tzinfo=timezone)

    return dt

def load_al_file(filename, timezone=timezone("UTC")):
    """
    Load data in the activity learning format, where each line resembles:
        2012-01-01 13:55:55.999999 OutsideDoor OFF Other_Activity

    Returns:
        - [(datetime, sensor_name str, sensor_value str, activity_label str), ...]

    Note: probably could have written this with Pandas read_csv instead
    """
    data = []

    with open(filename) as f:
        for i, line in enumerate(f):
            parts = line.strip().split(" ")
            assert len(parts) == 5, "Parts of a line "+str(i)+": "+len(parts)+" != 5"

            dt = parse_datetime(parts[0]+" "+parts[1], timezone)
            sensor_name = parts[2]
            sensor_value = parts[3]
            activity_label = parts[4]

            if dt is not None:
                data.append((dt, sensor_name, sensor_value, activity_label))

    return data

def compute_simple_features(config, data, filename=None):
    """
    Compute simple features from smart home data

    Time features:
        - second (/60)
        - minute (/60)
        - hour (/12)
        - hour (/24)
        - second of day (/86400)
        - day of week (/7)
        - day of month (/31)
        - day of year (/366)
        - month of year (/12)
        - year

    Smart home features:
        - 0 if no information about a feature at this time step,
          1 if a sensor turns off,
          2 if a sensor turns on,
          or if the sensor has a value (e.g. bat. or temp.), its given value

          Note: we could alternatively keep a history of whether other sensors
          are on/off and have the feature vector 0's or 1's accordingly and the
          last value of each bat. or temp. sensor, but unless the person is
          moving about rapidly (or multi-person home, which are being excluded)
          then probably only one or maybe two sensors are triggering at a time
          anyway. Thus, it probably doesn't matter all that much.
    """
    features = []
    labels = []

    num_time_features = 10 # see above
    num_sensor_features = len(config["features"])
    num_features = num_time_features + num_sensor_features

    for i, (dt, sensor_name, sensor_value, activity_label) in enumerate(data):
        # Skip these, not even sure what it is and very unlikely to correlate
        # well with activites in the home
        if sensor_name == "ZigbeeNetSecCounter":
            continue

        f = np.zeros((num_features,), dtype=np.float32)
        sensor = feature_to_int(config, sensor_name)
        label = label_to_int(config, activity_label)

        # Time features
        f[0] = dt.second
        f[1] = dt.minute
        f[2] = dt.hour % 12 # 12-hour
        f[3] = dt.hour # 24-hour
        f[4] = (dt - dt.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() # since midnight
        f[5] = dt.weekday()
        f[6] = dt.day # day of month
        f[7] = (dt - dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)).days + 1 # day of year
        f[8] = dt.month
        f[9] = dt.year

        # Sensor features
        if sensor_value in ["OFF", "OF", "OFF\"", "CLOSE"]:
            sensor_value = 1
        elif sensor_value in ["ON", "OPEN", "OPEN\""]:
            sensor_value = 2
        else:
            try:
                sensor_value = float(sensor_value)
            except ValueError:
                print("Warning: not parsing "+sensor_value+" as float on line "+str(i+1)+" of file "+str(filename))
                sensor_value = None

        # If we parsed it without an error
        if sensor_value is not None:
            f[num_time_features + sensor] = sensor_value

            # Output
            features.append(f)
            labels.append(label)

    return features, labels

def write_features(data, labels, filename):
    """ Write out the feature vector and labels to a .hdf5 file """
    d = h5py.File(filename, "w")
    d.create_dataset("features", data=np.array(data), compression="gzip")
    d.create_dataset("labels", data=np.array(labels), compression="gzip")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 simple_features.py al.config input.txt output.hdf5")
        sys.exit(1)
    else:
        config_file = sys.argv[1]
        input_file = sys.argv[2]
        output_file = sys.argv[3]

        # Load files
        config = load_config(config_file)
        data = load_al_file(input_file, timezone=timezone("US/Pacific"))

        # Compute simple features
        features, labels = compute_simple_features(config, data, filename=input_file)

        # Save to disk
        write_features(features, labels, output_file)
