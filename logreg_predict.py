from pathlib import Path
import argparse

from src import dataframe
from src import processing

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="input file")
parser.add_argument("--model", type=str, default="model/model.pkl", help="file to the model to be used for prediction")
parser.add_argument("--df_tool", default="model/model_df_tool.pkl", help="import scaler and labelizer from a file")
parser.add_argument("-o", "--output", default="houses.csv", help="output file")
parser.add_argument("-v", "--verbosity", type=int, default=1, choices=[0, 1, 2], help="verbosity level: 0->silent, 1->simple print, 2->detailed print")
args = parser.parse_args()

model = processing.LogReg()
try:
    df = dataframe.DataFrame(import_scale_and_label=args.df_tool)
    df.read_from_csv(args.file, header=True)
    model.load_model(args.model)
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file because : {}".format(err))
    exit(0)
df.drop_column([1])  # drop hogwart house column
df.drop_nan_column()
df.drop_nan_row()
df.scale(first_col=2)
y_pred, _ = model.predict(df.data[:, 1:], verbose=args.verbosity)
res = df.labelizer[1].inverse_transform(y_pred)
try:
    with Path(args.output).open(mode='w', encoding='utf-8') as fp:
        fp.write("Index,Hogwarts House\n")
        for index, stud in enumerate(res):
            fp.write("{},{}\n".format(int(df.data[index, 0]), stud))
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
if args.verbosity >= 1:
    print("Output written to '{}'".format(args.output))
