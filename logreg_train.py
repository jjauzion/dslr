from pathlib import Path
import argparse

from src import dataframe
from src import processing

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="data/dataset_train.csv", help="file to describe, shall be csv format")
parser.add_argument("--name", type=str, default="model", help="name of the model")
parser.add_argument("--save_dir", type=str, default="model", help="directory where to save model and df")
args = parser.parse_args()
df = dataframe.DataFrame()
try:
    classes = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    df.read_from_csv(args.file, header=True, converts={1: classes, 5: ["Left", "Right"]})
    # df.read_from_csv(args.file, header=True, converts={1: classes})
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
if not Path(args.save_dir).is_dir():
    print("Directory '{}' does not exist".format(args.save_dir))
    exit(0)
df.drop_column(0)
df.drop_nan_column()
df.drop_nan_row()
df.scale(scale_type="minmax", first_col=1)
model = processing.LogReg(nb_itertion=1000, learning_rate=0.1, nb_class=4, regularization_rate=0.3, model_name=args.name)
y_pred = model.fit(df.data[:, 1:], df.data[:, 0], verbose=2)
model_file = Path(args.save_dir) / "{}.pkl".format(args.name)
try:
    model.save_model(model_file)
except PermissionError as err:
    print("Can't save model to '{}' because : {}".format(model_file, err))
print("Model saved to '{}'".format(model_file))
df_tool_file = Path(args.save_dir) / "{}_df_tool.pkl".format(args.name)
try:
    df.save_scale_and_label(df_tool_file)
except PermissionError as err:
    print("Can't save dataframe scaler and label to '{}' because : {}".format(model_file, err))
print("Dataframe scaler and label saved to '{}'".format(df_tool_file))
