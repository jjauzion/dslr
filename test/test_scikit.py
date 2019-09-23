import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model

pd.options.display.width = 0
df = pd.read_csv("data/dataset_train.csv")
df = df.dropna(axis=0)
conv = preprocessing.LabelEncoder()
conv.fit(df["Hogwarts House"])
conv.transform(["Ravenclaw"])
df["Hogwarts House"] = conv.transform(df["Hogwarts House"])
logreg = linear_model.LogisticRegression(max_iter=1000)
logreg.fit(df.iloc[:, 6:], df["Hogwarts House"])
y = logreg.decision_function(df.iloc[:, 6:])
y = y.argmax(axis=1)
truth = df["Hogwarts House"].to_numpy()
accuracy = (truth.shape[0] - np.count_nonzero(y - truth)) / truth.shape[0]
print("accuracy = {}%".format(accuracy * 100))
