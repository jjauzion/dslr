import pandas as pd
from sklearn import metrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--df_predict", default="houses.csv", help="path to the df with the predicted value")
parser.add_argument("--df_true", default="data/data_test.csv", help="path to the df with the TRUE value")
parser.add_argument("-d", "--drop_col_6_7_16", action="store_true", help="drop col 5, 6 and 15 (useless col, see pairplot")
parser.add_argument("-dnan", "--drop_nan", action="store_true", help="drop NaN rows (any)")
args = parser.parse_args()

df_pred = pd.read_csv(args.df_predict)
df_true = pd.read_csv(args.df_true)
if args.drop_col_6_7_16:
    df_true = df_true.drop(df_true.columns[[6, 7, 16]], axis=1)
if args.drop_nan:
    df_true = df_true.dropna(axis=0, how='any')
print(metrics.accuracy_score(df_true.iloc[:, 1], df_pred.iloc[:, 1]))
