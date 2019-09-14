from pathlib import Path
import numpy as np
import argparse
import math
import pandas as pd

from src import dataframe

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="file to describe, shall be csv format")
args = parser.parse_args()
# try:
    # name, df = read_from_csv(args.file)
# except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError) as err:
    # print("Could not read file '{}' because : {}".format(Path(args.file), err))
# print(name)
# describe_df(df, name)
df = dataframe.DataFrame()
try:
    df.read_from_csv(args.file, header=True)
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
df.describe()
