#!/usr/bin/env python3

import argparse

from data_analysis.plot import plot_flex_snapshot
from data_analysis.read import read_pickle


def plot_flex_for_all_layers(qhistory):
    layers = [key for key in qhistory.keys() if key != "global"]
    for layer in layers:
        plot_flex_snapshot(
            layer_name=layer,
            layer_history=qhistory[layer]["history"],
            # TODO(Fran): Here kernel is hardcoded, what should we do for other layers?....
            quantizer=qhistory[layer]["qconfig"]["weights"]["kernel"],
            accuracy_history=qhistory["global"]["accuracy"],
            output_path=f"snapshots/{layer}",
        )


def main(args):
    qhistory = read_pickle(args.input_path)
    plot_flex_for_all_layers(qhistory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        help="Path to the qhistory file",
        required=True,
    )
    args = parser.parse_args()
    main(args)
