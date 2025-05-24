#!/usr/bin/env python3

import pickle


def read_pickle(path):
    """Read the qhistory file and return a DataFrame."""
    with open(path, "rb") as f:
        qhistory = pickle.load(f)
    return qhistory


if __name__ == "__main__":
    qhistory = read_pickle("snapshots/output_dict.pkl")
    print(qhistory["quant_hidden"]["history"].keys())
    # print(qhistory["quant_hidden"]["qconfig"]["weights"]["kernel"].__dir__())
