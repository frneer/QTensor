#!/usr/bin/env python3


import argparse

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from configs.qmodel import apply_quantization
from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer
from utils.plot import (
    VariableHistoryCallback,
    plot_flex_snapshot,
    plot_training_history,
    plot_uniform_snapshot,
)


def generate_dataset():
    """Generate the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def main(bits, uniform, levels):
    (x_train, y_train), (x_test, y_test) = generate_dataset()
    tf.random.set_seed(42)

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
        epochs=10,
        batch_size=1024 // 4,
        validation_data=(x_test, y_test),
    )
    kernel_quantizer = FlexQuantizer(bits=bits, n_levels=levels, signed=True)
    if uniform:
        kernel_quantizer = UniformQuantizer(bits=bits, signed=True)

    qconfig = {
        "hidden": {
            "weights": {"kernel": kernel_quantizer},
            "activations": {
                "activation": UniformQuantizer(bits=bits, signed=False)
            },
        }
    }

    qmodel = apply_quantization(model, qconfig)

    qmodel.summary()
    qmodel.compile(
        optimizer=Adam(learning_rate=0.001 / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print([w.name for w in model.layers[1].weights])
    print([w.name for w in qmodel.layers[1].weights])

    alpha_callback = [
        VariableHistoryCallback(v)
        for v in qmodel.layers[1].weights
        if "alpha" in v.name
    ][0]
    if not uniform:
        levels_callback = [
            VariableHistoryCallback(v)
            for v in qmodel.layers[1].weights
            if "levels" in v.name
        ][0]
        thresholds_callback = [
            VariableHistoryCallback(v)
            for v in qmodel.layers[1].weights
            if "thresholds" in v.name
        ][0]

        callbacks = [alpha_callback, levels_callback, thresholds_callback]
    else:
        callbacks = [alpha_callback]
    initial_loss, initial_acc = qmodel.evaluate(x_train, y_train, verbose=0)
    hist = qmodel.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )

    alpha_history = [
        alpha_callback.get_pre_epoch_history()[0]
    ] + alpha_callback.get_history()
    if not uniform:
        levels_history = [
            levels_callback.get_pre_epoch_history()[0]
        ] + levels_callback.get_history()
        thresholds_history = [
            thresholds_callback.get_pre_epoch_history()[0]
        ] + thresholds_callback.get_history()
    # for the accuracy and loss history, we can just use the hist object
    # which already contains the full history including epoch 0
    # so no need to modify it
    accuracy_history = [initial_acc] + hist.history["accuracy"]
    loss_history = [initial_loss] + hist.history["loss"]

    print(f"Length of alpha history: {len(alpha_history)}")
    print(f"Length of accuracy history: {len(accuracy_history)}")

    # append pre-epoch first value only

    # print(f"Alpha pre-epoch history: {alpha_callback.get_pre_epoch_history()}")
    # print(f"Alpha history: {alpha_history}")
    # alpha_history = [alpha_callback.get_pre_epoch_history()[0]] + alpha_history

    if uniform:
        plot_uniform_snapshot(
            alpha_hist=alpha_history,
            accuracy_hist=accuracy_history,
            bits=bits,
            signed=True,
            output_path="snapshots",
        )
    else:
        plot_flex_snapshot(
            alpha_hist=alpha_history,
            level_hist=levels_history,
            threshold_hist=thresholds_history,
            accuracy_hist=accuracy_history,
            bits=bits,
            signed=True,
            output_path="snapshots",
        )

    history_dict = {
        "alpha": alpha_history,
        # "accuracy": accuracy_history,
        "loss": loss_history,
    }
    if not uniform:
        history_dict["levels"] = levels_history
        history_dict["thresholds"] = thresholds_history

    output_dir = "uniform" if uniform else "flex"
    output_filename = "training_history.png"
    plot_training_history(
        vars=history_dict,
        output_path=f"snapshots/{output_dir}/" + output_filename,
    )

    # plot_raining_history.png")

    # plot_flex_training_history(hist, callbacks, output_path="training_history.png")

    qmodel.evaluate(x_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bits",
        type=int,
        default=3,
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
    parser.add_argument(
        "--uniform",
        action="store_true",
        help="use uniform quantization",
    )
    args = parser.parse_args()
    main(args.bits, args.uniform, args.levels)
