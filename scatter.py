import argparse
import pandas as pd
import matplotlib.pyplot as plt

from src import dataframe


def scatter(df):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="file to describe, shall be csv format")
    args = parser.parse_args()
    df = dataframe.DataFrame()
    try:
        # pddf = pd.read_csv(args.file)
        df.read_from_csv(args.file)
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError) as err:
        print("Could not read file '{}' because : {}".format(Path(args.file), err))
        exit(0)
    df.drop_nan_column()
    df.drop_column(0)
    df.scale()
    scatter(df)
