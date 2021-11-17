import pickle
import json
import re
from glob import glob as pyglob
from pathlib import Path as PyPath

import gcsfs
import numpy as np
import pandas as pd
from pathy import Pathy
from tqdm import tqdm


def end_match(path: str):
    if path.startswith("gs"):
        _path = Pathy(path).parent
        wcmatch = re.compile(str(path).replace(".", "\\.").replace("*", ".+"))
        pathlist = list(map(str, _path.iterdir()))
        pathlist = list(filter(lambda x: wcmatch.match(x) is not None, pathlist))
        return pathlist
    else:
        return pyglob(path)


def Path(path: str):
    if path.startswith("gs"):
        return Pathy(path)
    else:
        return PyPath(path)


def load_pickle(path):
    if path.startswith("gs"):
        fs = gcsfs.GCSFileSystem(project="oheadline")
        with fs.open(path, "rb") as f:
            out = pickle.load(f)
    else:
        with open(path, "rb") as f:
            out = pickle.load(f)

    return out


def save_pickle(path, file):
    if path.startswith("gs"):
        fs = gcsfs.GCSFileSystem(project="oheadline")
        with fs.open(path, "wb") as f:
            pickle.dump(file, f)
    else:
        with open(path, "wb") as f:
            pickle.dump(file, f)


def universal_open(path, mode):
    if path.startswith("gs"):
        fs = gcsfs.GCSFileSystem(project="oheadline")
        return fs.open(path, mode)
    else:
        return open(path, mode)


def load_pickle_chunked(path):
    def getpath(path):
        i = 0
        while True:
            yield i == 0, path + f".{i}"
            i += 1

    paths = getpath(path)
    dfs = []
    while True:
        first, _path = next(paths)
        try:
            dfs.append(pd.read_pickle(_path))
            print("Loading df from", _path)
        except FileNotFoundError as e:
            if first:
                raise e
            else:
                print(e)
                break
    return pd.concat(dfs)


def save_pickle_chunked(path, df):
    dfs = np.array_split(df, 1 + (len(df) // 10000))
    for i, df in tqdm(enumerate(dfs)):
        print("Saving df to", path + f".{i}")
        df.to_pickle(path + f".{i}")
