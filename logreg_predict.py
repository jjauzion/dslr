from pathlib import Path
import argparse
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from src import dataframe
from src import processing


def impute_nan(df, method):
    if method == "0":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
    elif method == "mean":
        imputer = SimpleImputer()
    elif method == "adv":
        imputer = IterativeImputer()
    else:
        raise ValueError("'{}' is not a valid impute method")
    imputer.fit(df)
    return imputer.transform(df)


parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="input file")
parser.add_argument("--model", type=str, default="model/model.pkl", help="file to the model to be used for prediction")
parser.add_argument("--df_tool", default="model/model_df_tool.pkl", help="import scaler and labelizer from a file")
parser.add_argument("-o", "--output", default="houses.csv", help="output file")
parser.add_argument("-v", "--verbosity", type=int, default=1, choices=[0, 1, 2], help="verbosity level: 0->silent, 1->simple print, 2->detailed print")
parser.add_argument("-d", "--drop_col_6_7_16", action="store_true", help="drop col 5, 6 and 16 (useless col, see pairplot")
parser.add_argument("-inan", "--impute_nan", choices=["ignore", "0", "mean", "adv"], type=str, default="mean", help="how to deal with NaN: None->NaN are removed ; 0->NaN replaced with 0 ; mean-> NaN replaced by mean ; adv -> adv imputer from skl")
args = parser.parse_args()

model = processing.LogReg()
try:
    df = dataframe.DataFrame(import_scale_and_label=args.df_tool)
    df.read_from_csv(args.file, header=True)
    model.load_model(args.model)
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file because : {}".format(err))
    exit(0)
df.drop_column([1])             # drop hogwart house column
if args.drop_col_6_7_16:
    df.drop_column([5, 6, 15])  # useless column (homogenous distri and correlated variable)
df.drop_nan_column()
if args.impute_nan != "ignore":
    df.data = impute_nan(df.data, args.impute_nan)
else:
    df.drop_nan_row()
df.scale(first_col=2)
try:
    y_pred, _ = model.predict(df.data[:, 1:], verbose=args.verbosity)
except ValueError as err:
    print("Can't use model '{}' to make prediction on '{}' data because : {}".format(args.model, args.file, err))
    exit(0)
res = df.labelizer[1].inverse_transform(y_pred)
try:
    with Path(args.output).open(mode='w', encoding='utf-8') as fp:
        fp.write("Index,Hogwarts House\n")
        for index, stud in enumerate(res):
            fp.write("{},{}\n".format(int(df.data[index, 0]), stud))
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
if args.verbosity >= 1:
    print("Output written to '{}'".format(args.output))
