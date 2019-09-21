import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pickle


class LogReg:

    def __init__(self, nb_itertion=1000, learning_rate=0.1, nb_class=1, regularization="l1", model_name=None):
        self.nb_iter = nb_itertion
        self.learning_rate = learning_rate
        self.nb_class = nb_class
        self.regularization = regularization
        self.name = model_name
        self.confusion_matrix = np.zeros((nb_class, nb_class), dtype=int)
        self.precision = [-1]
        self.recall = [-1]
        self.f1score = [-1]
        self.accuracy = -1
        self.weight = None
        self.cost_history = np.zeros((nb_itertion, nb_class))

    def describe(self):
        print("Weights :\n{}".format(self.weight))
        print("\nPerformance:")
        self.print_accuracy()

    def print_accuracy(self, class_name=None):
        """

        :param class_name: list containing the name of each class in order
        """
        class_name = class_name if class_name is not None else [str(elm) for elm in range(self.nb_class)]
        col_padding = [15] + [max(7, len(elm)) for elm in class_name]
        line = [
            "".ljust(col_padding[0], " "),
            "Precision".ljust(col_padding[0], " "),
            "Recall".ljust(col_padding[0], " "),
            "F1score".ljust(col_padding[0], " ")
        ]
        for i in range(self.nb_class):
            line[0] += class_name[i].ljust(col_padding[i + 1], " ")
            line[1] += "{}%".format(str(round(self.precision[i] * 100, 2))).ljust(col_padding[i + 1], " ")
            line[2] += "{}%".format(str(round(self.recall[i] * 100, 2))).ljust(col_padding[i + 1], " ")
            line[3] += "{}%".format(str(round(self.f1score[i] * 100, 2))).ljust(col_padding[i + 1], " ")
        print("\n".join(line))
        print("{title:<{width1}}{val:<{width2}}%".format(
            title="Accuracy", width1=col_padding[0], val=round(self.accuracy * 100, 2), width2=col_padding[1] - 2))

    def plot_training(self, class_name=None):
        fig = plt.figure("Training convergence")
        for i in range(self.nb_class):
            plt.plot(self.cost_history[:, i])
        plt.legend(list(range(self.nb_class)))
        plt.title("Cost history")
        plt.xlabel("nb of iterations")
        plt.ylabel("Cost")
        plt.show()

    def load_model(self, file):
        with Path(file).open(mode='rb') as fd:
            try:
                model = pickle.load(fd)
            except (pickle.UnpicklingError, EOFError) as err:
                print("Can't load model from '{}' because : {}".format(file, err))
                return False
        if not isinstance(model, dict):
            print("Given file '{}' is not a valid model".format(file))
            return False
        for key in model.keys():
            if key not in self.__dict__.keys():
                print("Given file '{}' is not a valid model".format(file))
                return False
        self.__dict__.update(model)
        return True

    @staticmethod
    def _sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def _to_class_id(Y_pred):
        """

        :param Y_pred: m by nb_class matrix, with m nb of sample
        :return: m by 1 matrix -> predicted class number
        """
        return Y_pred.argmax(axis=1)

    def _compute_hypothesis(self, X):
        """

        :param weight: n by nb_class matrix, with n the number of parameter
        :param X: m by n matrix, with n the number of parameter and m nb of sample
        :return: m by nb_class matrix, with m nb of sample
        """
        return self._sigmoid(np.matmul(X, self.weight))

    def _compute_cost(self, X, Y, H, regul=0):
        """

        self.weight : n by nb_class matrix, with n the number of parameter
        :param X: m by n matrix, with n the number of parameter and m nb of sample
        :param Y: m by nb_class matrix, with m nb of sample
        :param H: m by nb_class matrix, with m nb of sample. Matrix of the computed hypothesis Y with the current weight
        :param regul: value of the regularization term
        :return:
        """
        cost = -1 / X.shape[0] * (np.matmul(Y.T, np.log(H)) + np.matmul((1 - Y).T, np.log(1 - H)))
        regul_tmp = regul / (2 * X.shape[0]) * (np.matmul(self.weight.T, self.weight))
        return np.diagonal(cost + regul_tmp)

    def _update_weight(self, X, Y, H, regul=0):
        """

        self.weight : n by nb_class matrix, with n the number of parameter
        :param X: m by n matrix, with n the number of parameter and m nb of sample
        :param Y: m by nb_class matrix, with m nb of sample
        :param H: m by nb_class matrix, with m nb of sample. Matrix of the computed hypothesis Y with the current weight
        :param regul: value of the regularization term
        :return: n by nb_class matrix
        """
        m = X.shape[0]
        return self.weight - self.learning_rate * (np.matmul(X.T, H - Y) / m + regul * self.weight / m)

    def _compute_accuracy(self, y, y_pred):
        for i in range(y.shape[0]):
            self.confusion_matrix[int(y_pred[i]), int(y[i])] += 1
        total_predicted = np.sum(self.confusion_matrix, axis=1)
        total_true = np.sum(self.confusion_matrix, axis=0)
        true_positive = np.diagonal(self.confusion_matrix)
        self.precision = true_positive / total_true
        self.recall = true_positive / total_predicted
        self.f1score = 2 * self.precision * self.recall / (self.precision + self.recall)
        self.accuracy = np.count_nonzero(np.equal(y, y_pred)) / y.shape[0]

    @staticmethod
    def _get_multi_class_y(y, nb_class):
        def is_class(val, class_nb):
            return 1 if val == class_nb else 0
        Y = np.ones((y.shape[0], nb_class), dtype="float64")
        for i in range(nb_class):
            Y[:, i] = [is_class(val, i) for val in y]
        return Y

    def fit(self, X, y, verbose=1):
        """

        :param X: matrix of shape (n_samples, n_feature)
        :param y: vector of shape (n_samples)
        :param verbose:
        :return:
        """
        X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        Y = self._get_multi_class_y(y, self.nb_class) if self.nb_class > 1 else y.reshape(-1, 1)
        self.weight = np.random.random((X.shape[1], self.nb_class))
        for i in range(self.nb_iter):
            H = self._compute_hypothesis(X)
            self.weight = self._update_weight(X, Y, H)
            self.cost_history[i, :] = self._compute_cost(X, Y, H)
        Y_pred = self._compute_hypothesis(X)
        y_pred = self._to_class_id(Y_pred)
        self._compute_accuracy(y, y_pred)
        if verbose > 0:
            print("Training completed!")
            self.print_accuracy()
        if verbose > 1:
            self.plot_training()
        return y, Y, y_pred, Y_pred

    def predict(self, x, verbose=1):
        """
        Make prediction based on x
        :param x: List or 1 by n numpy array with n = nb of parameter
        :return:
        """
        if not isinstance(x, np.ndarray):
            if not isinstance(x, list):
                raise TypeError("x shall be a list or a np array. Got {}".format(x))
            x_pred = np.array([x])
        else:
            x_pred = x
        x_pred = np.insert(x_pred, 0, np.ones(x_pred.shape[0]), axis=1)
        if self.weight is None:
            self.weight = np.zeros((x_pred.shape[1], 1))
        prediction = np.matmul(x_pred, self.weight)
        if verbose == 2:
            self.plot_prediction(x[0], prediction)
        return prediction

    def plot_train_set(self):
        print(self.X_original)
        plt.scatter(self.X_original, self.y)
        plt.show()

    def save_model(self, file):
        with Path(file).open(mode='wb') as fd:
            pickle.dump(self.__dict__, fd)
        print("Model save to '{}'".format(Path(file)))
