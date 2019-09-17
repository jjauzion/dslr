import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import dataframe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="file to describe, shall be csv format")
    args = parser.parse_args()
    df = dataframe.DataFrame()
    try:
        df = pd.read_csv(args.file)
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError) as err:
        print("Could not read file '{}' because : {}".format(Path(args.file), err))
        exit(0)
    pd.options.display.width = 0
    g = sns.pairplot(df.sample(100), hue="Hogwarts House", dropna=True)
    plt.show()
