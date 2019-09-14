import numpy as np


class IdentityScale:
    """
    No scale, return data as is.
    """
    name = "IdentityScale"

    def __repr__(self):
        return "{}".format(self.name)

    def fit(self, data):
        return True

    def transform(self, data):
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


class MeanNormScaler:

    name = "MeanNormScale"

    def __init__(self):
        self.mean = None
        self.scale = None

    def __repr__(self):
        return "{} ; mean = {} ; std = {}".format(self.name, self.mean, self.scale)

    def fit(self, data):
        """
        fit scaler to data per column (parameters are column, experience are lines)
        :param data: array with parameters in columns
        """
        self.mean = np.average(data, axis=0)
        self.scale = np.std(data, axis=0)

    def transform(self, data, inplace=False):
        if self.mean is None or self.scale is None:
            raise RuntimeError("Scaler must be fitted to the data before transform.")
        normalized_data = np.zeros(data.shape)
        for i, col in enumerate(data.T):
            normalized_data[:, i] = (col - self.mean[i]) / self.scale[i]
        if inplace:
            data[:] = normalized_data[:]
        return normalized_data

    def fit_transform(self, data, inplace=False):
        self.fit(data)
        return self.transform(data, inplace=inplace)


class MinMaxScaler():

    name = "MeanNormScale"

    def __init__(self):
        self.min = None
        self.max = None
        self.range = None

    def __repr__(self):
        return "{} ; min = {} ; max = {}".format(self.name, self.min, self.max)

    def fit(self, data):
        """
        fit scaler to data per column (parameters are column, experience are lines)
        :param data: array with parameters in columns
        """
        self.min = np.nanmin(data, axis=0)
        self.max = np.nanmax(data, axis=0)
        self.range = self.max - self.min

    def transform(self, data, inplace=False):
        if self.min is None or self.max is None:
            raise RuntimeError("Scaler must be fitted to the data before transform.")
        normalized_data = np.zeros(data.shape)
        for i, col in enumerate(data.T):
            normalized_data[:, i] = (col - self.min[i]) / self.range[i]
        if inplace:
            data[:] = normalized_data[:]
        return normalized_data

    def fit_transform(self, data, inplace=False):
        self.fit(data)
        return self.transform(data,inplace=inplace)
