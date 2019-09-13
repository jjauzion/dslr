from pathlib import Path
import numpy as np
import argparse
import math
import pandas as pd


def read_from_csv(file, header=True):
    """
    create a dataframe from a csv file
    :param file: csv file path to open
    :param header: if True (default), the first line is read as a aheader to get column names
    :return name, dataframe. Name is retrun as a list, and dataframe as a np.array
    """
    with Path(file).open(mode='r', encoding='utf-8') as fp:
        name = fp.readline().split(',') if header else []
        df = np.genfromtxt(fp, delimiter=',', dtype="float64")
    return name, df


def mean_2(df, axis=0):
    _sum = np.nansum(df, axis=axis)
    count = np.count_nonzero(~np.isnan(df), axis=0)
    return _sum / count


def count1D(vector):
    count = 0
    for val in vector:
        if not np.isnan(val):
            count += 1
    return count


def count_val(df, axis=0):
    return np.apply_along_axis(count1D, axis=0, arr=df)


def mean1D(vector):
    _sum = 0
    nb_nan = 0
    for val in vector:
        if not np.isnan(val):
            _sum += val
        else:
            nb_nan += 1
    count = len(vector) - nb_nan
    return _sum / count if count > 0 else np.nan


def mean(df, axis=0):
    return np.apply_along_axis(mean1D, axis=0, arr=df)


def min1D(vector):
    min_val = vector[0]
    for val in vector:
        if val < min_val:
            min_val = val
    return min_val


def _min(df, axis=0):
    return np.apply_along_axis(min1D, axis=0, arr=df)


def max1D(vector):
    max_val = vector[0]
    for val in vector:
        if val > max_val:
            max_val = val
    return max_val


def _max(df, axis=0):
    return np.apply_along_axis(max1D, axis=0, arr=df)


def std1D(vector):
    _mean = mean1D(vector)
    _sum = 0
    nb_nan = 0
    for val in vector:
        if not np.isnan(val):
            _sum += (val - _mean) ** 2
        else:
            nb_nan += 1
    count = len(vector) - nb_nan
    return math.sqrt(_sum / (count - 1)) if count > 0 else np.nan


def std(df, axis=0):
    return np.apply_along_axis(std1D, axis=0, arr=df)


def percentile1D(vector, centile):
    if 0 < centile > 100:
        raise ValueError("centile shall be between 0 and 100. Got '{}'".format(centile))
    sorted_vect = np.sort(vector)
    print("len={}; cent={}.".format(len(vector), centile))
    index = (len(vector) + 1) * centile / 100 - 1
    print(index)
    decimal = index - math.floor(index)
    if decimal != 0:
        a = sorted_vect[math.floor(index)]
        b = sorted_vect[math.floor(index) + 1]
        print("dec={}; a={}; b={}.".format(decimal,a,b))
        return a + (b - a) * decimal
    else:
        return sorted_vect[int(index)]


def percentile(df, centile, axis=0):
    return np.apply_along_axis(percentile1D, 0, df, centile)


def describe_df(df, name):
    stats = count_val(df)
    stats = np.vstack((stats, mean(df)))
    stats = np.vstack((stats, std(df)))
    stats = np.vstack((stats, _min(df)))
    stats = np.vstack((stats, percentile(df, 25)))
    stats = np.vstack((stats, percentile(df, 50)))
    stats = np.vstack((stats, percentile(df, 75)))
    stats = np.vstack((stats, _max(df)))
    stats_df = pd.DataFrame(stats, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], columns=name)
    stats_df.dropna(axis='columns', inplace=True)
    pd.options.display.width = 0
    print(stats_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="file to describe, shall be csv format")
    args = parser.parse_args()

    try:
        name, df = read_from_csv(args.file)
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError) as err:
        print("Could not read file '{}' because : {}".format(Path(args.file), err))
    print(df)
    describe_df(df, name)
    proof = pd.read_csv(args.file)
    print(proof.describe())
