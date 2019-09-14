import matplotlib.pylab as plt
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src import dataframe


def histo(df, course):
    plt.hist(df[df["Hogwarts House"] == "Ravenclaw"][course].dropna(), color='r', alpha=0.5)
    plt.hist(df[df["Hogwarts House"] == "Slytherin"][course].dropna(), color='g', alpha=0.5)
    plt.hist(df[df["Hogwarts House"] == "Gryffindor"][course].dropna(), color='y', alpha=0.5)
    plt.hist(df[df["Hogwarts House"] == "Hufflepuff"][course].dropna(), color='b', alpha=0.5)
    plt.legend(("Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="file to describe, shall be csv format")
    args = parser.parse_args()
    df = dataframe.DataFrame()
    try:
        pddf = pd.read_csv(args.file)
        df.read_from_csv(args.file)
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError) as err:
        print("Could not read file '{}' because : {}".format(Path(args.file), err))
        exit(0)
    df.drop_nan_column()
    df.scale()
    most_homogeneous = df.header[np.nanargmin(df.std())]
    print("Course with lowest standard deviation is : '{}'".format(most_homogeneous))
    histo(pddf, most_homogeneous)
