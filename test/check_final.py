import pandas as pd
from sklearn import metrics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--df_predict", default="houses.csv", help="path to the df with the predicted value")
parser.add_argument("--df_true", default="data/data_test.csv", help="path to the df with the TRUE value")
args = parser.parse_args()

df_pred = pd.read_csv(args.df_predict)
df_true = pd.read_csv(args.df_true)
df_true = df_true.dropna(axis=0, how='any')
print(metrics.accuracy_score(df_true.iloc[:, 1], df_pred.iloc[:, 1]))
