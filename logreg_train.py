from pathlib import Path
import argparse

from src import dataframe

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="data/dataset_train.csv", help="file to describe, shall be csv format")
args = parser.parse_args()
df = dataframe.DataFrame()
try:
    classes = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    df.read_from_csv(args.file, header=True, converts={1: classes, 5: ["Left", "Right"]})
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
