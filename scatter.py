import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from src import dataframe


def scatter_all(df):
    fig = plt.figure("scatter")
    plt.subplot()
    nb_col = df.data.shape[1]
    for x in range(nb_col):
        for y in range(x + 1, nb_col):
            plt.scatter(x=df.data[:, x], y=df.data[:, y])
            plt.xlabel(df.header[x])
            plt.ylabel(df.header[y])
            plt.subplot(nb_col, nb_col, y + x * nb_col)
    plt.show()


def scatter(df, x_column, y_column):
    plt.scatter(x=df.data[:, x_column], y=df.data[:, y_column])
    plt.xlabel(df.header[x_column])
    plt.ylabel(df.header[y_column])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/dataset_train.csv", help="file to describe, shall be csv format")
    parser.add_argument("-a", "--all", action="store_true", help="Scatter plot of all column in the dataset")
    args = parser.parse_args()
    df = dataframe.DataFrame()
    try:
        df.read_from_csv(args.file)
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError) as err:
        print("Could not read file '{}' because : {}".format(Path(args.file), err))
        exit(0)
    df.drop_nan_column()
    df.drop_column(0)
    df.scale()
    if args.all:
        scatter_all(df)
    else:
        scatter(df, 1, 3)
