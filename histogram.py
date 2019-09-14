import pandas as pd
import matplotlib.pylab as plt


def histo(df, course):
    plt.hist(df[df["Hogwarts House"] == "Ravenclaw"][course].dropna(), color='r', alpha=0.5)
    plt.hist(df[df["Hogwarts House"] == "Slytherin"][course].dropna(), color='g', alpha=0.5)
    plt.hist(df[df["Hogwarts House"] == "Gryffindor"][course].dropna(), color='y', alpha=0.5)
    plt.hist(df[df["Hogwarts House"] == "Hufflepuff"][course].dropna(), color='b', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv("data/dataset_train.csv")
    histo(df, "Astronomy")
    histo(df, "Care of Magical Creatures")
