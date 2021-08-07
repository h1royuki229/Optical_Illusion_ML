import torch
import pickle


def dump_pickle(file, obj):
    with open(file, mode="wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_name):
    with open(file_name, mode="rb") as f:
        return pickle.load(f)