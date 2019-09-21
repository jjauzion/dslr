from pathlib import Path
import argparse

from src import dataframe
from src import processing

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="data/dataset_train.csv", help="file to describe, shall be csv format")
args = parser.parse_args()
df = dataframe.DataFrame()
try:
    classes = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    df.read_from_csv(args.file, header=True, converts={1: classes, 5: ["Left", "Right"]})
    # df.read_from_csv(args.file, header=True, converts={1: classes})
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
df.drop_column(0)
df.drop_nan_column()
df.drop_nan_row()
df.scale(scale_type="minmax", first_col=1)
model = processing.LogReg(nb_itertion=1000, learning_rate=0.1, nb_class=4, regularization_rate=0.3)
y, Y, y_pred, Y_pred = model.fit(df.data[:, 1:], df.data[:, 0], verbose=2)
model.save_model("with_regul.pkl")
