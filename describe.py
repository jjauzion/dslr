from pathlib import Path
import numpy as np
import argparse


def read_from_csv(file):
    with Path(file).open(mode='r', encoding='utf-8') as fp:
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="file to describe, shall be csv format")
    args = parser.parse_args()

    try:
        df = np.gen
        df = read_from_csv(args.file)
    except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError) as err:
        print("Could not read file '{}' because : {}".format(Path(args.file), err))
    print(df)
