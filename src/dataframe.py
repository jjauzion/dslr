from pathlib import Path
import numpy as np
import pandas as pd
import pickle

from . import toolbox as tb
from . import preprocessing


class DataFrame:

    def __init__(self, import_scale_and_label=None):
        self.data = None
        self.original_data = None
        self.header = None
        self.scaler = None
        self.labelizer = None
        if import_scale_and_label is not None:
            self.load_scale_and_label(import_scale_and_label)

    def read_from_csv(self, file, header=True, converts=None):
        """
        create a dataframe from a csv file
        :param file: csv file path to open
        :param header: if True (default), the first line is read as a header to get column names
        :param converts: {column: [classes]} -> will convert the value in column: each value of 'classes' will be a numeric value
        """
        with Path(file).open(mode='r', encoding='utf-8') as fp:
            self.header = np.array(fp.readline().split(',')) if header else []
            if converts is not None:
                converters = {}
                self.labelizer = {}
                for column in converts:
                    self.labelizer[column] = preprocessing.LabelEncoder(converts[column])
                    converters[column] = self.labelizer[column].transform
                self.data = np.genfromtxt(fp, delimiter=',', dtype="float64", converters=converters)
            else:
                self.data = np.genfromtxt(fp, delimiter=',', dtype="float64")
        self.original_data = np.copy(self.data)

    def scale(self, scale_type="minmax", import_scale=None, first_col=0):
        """

        :param scale_type: minmax (default) or meannorm
        :param import_scale: path to the scale file to be imported. If None, a new scaler will be created.
        :param first_col: nb of column at the beginning of the df that shall not be scaled
        :return:
        """
        if scale_type == "minmax":
            self.scaler = preprocessing.MinMaxScaler()
        elif scale_type == "meannorm":
            self.scaler = preprocessing.MeanNormScaler()
        else:
            raise ValueError("scale type unknown. Got '{}'".format(scale_type))
        self.scaler.fit_transform(self.data[:, first_col:], inplace=True)

    def count(self, axis=0):
        return np.apply_along_axis(tb.count_vector, axis=axis, arr=self.data)

    def mean(self, axis=0):
        return np.apply_along_axis(tb.mean_vector, axis=axis, arr=self.data)

    def min(self, axis=0):
        return np.apply_along_axis(tb.min_vector, axis=axis, arr=self.data)

    def max(self, axis=0):
        return np.apply_along_axis(tb.max_vector, axis=axis, arr=self.data)

    def std(self, axis=0):
        return np.apply_along_axis(tb.std_vector, axis=axis, arr=self.data)

    def percentile(self, centile, axis=0):
        return np.apply_along_axis(tb.percentile_vector, axis, self.data, centile)

    def describe(self):
        stats = np.array([
            self.count(), self.mean(), self.std(), self.min(),
            self.percentile(25), self.percentile(50), self.percentile(75), self.max()
        ])
        stats_df = pd.DataFrame(stats, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], columns=self.header)
        stats_df.dropna(axis='columns', inplace=True)
        pd.options.display.width = 0
        print(stats_df)

    def drop_column(self, col_index):
        self.data = np.delete(self.data, col_index, axis=1)
        self.header = np.delete(self.header, col_index)

    def drop_nan_column(self):
        mask = ~np.all(np.isnan(self.data), axis=0)
        self.data = self.data[:, mask]
        self.header = self.header[mask]

    def drop_nan_row(self):
        self.data = self.data[~np.any(np.isnan(self.data), axis=1)]

    def save_scale_and_label(self, file):
        df_tool = {
            "scaler": self.scaler,
            "labelizer": self.labelizer
        }
        with Path(file).open(mode='wb') as fp:
            pickle.dump(df_tool, fp)

    def load_scale_and_label(self, file):
        with Path(file).open(mode='rb') as fp:
            df_tool = pickle.load(fp)
        self.labelizer = df_tool["labelizer"]
        self.scaler = df_tool["scaler"]
