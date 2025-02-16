#!/usr/bin/env python3


import argparse

import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from configs.qmodel import apply_quantization

from quantizers.uniform_quantizer import UniformQuantizer
from quantizers.flex_quantizer import FlexQuantizer
from utils.utils import VariableHistoryCallback, plot_snapshot


def generate_dataset():
    """Generate the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def main(bits, alpha, levels):
    (x_train, y_train), (x_test, y_test) = generate_dataset()

    layer_1 = Flatten(input_shape=(28, 28), name="input")
    layer_2 = Dense(128, activation="relu", name="hidden")
    layer_3 = Dense(10, activation="softmax", name="output")
    model = Sequential([layer_1, layer_2, layer_3])

    model.summary()

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        x_train,
        y_train,
        epochs=2,
        batch_size=1024,
        validation_data=(x_test, y_test),
    )

    qconfig = {
        "hidden": {
            "weights": {
                "kernel": FlexQuantizer(bits=bits, n_levels=levels , signed=True)
            },
            "activations": {
                "activation": UniformQuantizer(bits=bits, signed=False)
            }
        }
    }

    qmodel = apply_quantization(model, qconfig)

    qmodel.summary()
    qmodel.compile(
        optimizer=Adam(learning_rate=0.001 / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print([w.name for w in model.layers[1].weights])
    print([w.name for w in qmodel.layers[1].weights])


    alpha_callback = [VariableHistoryCallback(v) for v in qmodel.layers[1].weights if "alpha" in v.name][0]
    levels_callback = [VariableHistoryCallback(v) for v in qmodel.layers[1].weights if "levels" in v.name][0]
    thresholds_callback = [VariableHistoryCallback(v) for v in qmodel.layers[1].weights if "thresholds" in v.name][0]
    callbacks = [alpha_callback, levels_callback, thresholds_callback]
    hist = qmodel.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )

    plot_snapshot(
        alpha_hist=alpha_callback.get_history(),
        level_hist=levels_callback.get_history(),
        threshold_hist=thresholds_callback.get_history(),
        accuracy_hist=hist.history["accuracy"],
        signed=True,
        output_path="snapshots",
        bits=bits,
    )

    # qmodel.evaluate(x_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bits",
        type=int,
        default=6,
        help="number of bits for quantization",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="initial quantization limit",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size for training",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=10,
        help="number of levels for quantization",
    )
    args = parser.parse_args()
    main(args.bits, args.alpha, args.levels)
