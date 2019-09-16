from pathlib import Path
import numpy as np
import pandas as pd

from . import toolbox as tb
from . import scaler


class DataFrame:

    def __init__(self):
        self.data = None
        self.original_data = None
        self.header = None
        self.scaler = None

    def read_from_csv(self, file, header=True):
        """
        create a dataframe from a csv file
        :param file: csv file path to open
        :param header: if True (default), the first line is read as a aheader to get column names
        """
        with Path(file).open(mode='r', encoding='utf-8') as fp:
            self.header = np.array(fp.readline().split(',')) if header else []
            self.data = np.genfromtxt(fp, delimiter=',', dtype="float64")
        self.original_data = np.copy(self.data)

    def scale(self, scale_type="minmax"):
        if scale_type == "minmax":
            self.scaler = scaler.MinMaxScaler()
        elif scale_type == "meannorm":
            self.scaler == scaler.MeanNormScaler()
        else:
            raise ValueError("scale type unknown. Got '{}'".format(scale_type))
        self.scaler.fit_transform(self.data, inplace=True)

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
