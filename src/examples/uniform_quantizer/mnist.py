#!/usr/bin/env python3


import argparse

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from quantizers.uniform_quantizer import UniformQuantizer
from utils.utils import VariableHistoryCallback

from builder.builder import QBuilder


def generate_dataset():
    """Generate the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def create_model(bits, alpha, signed):
    builder = QBuilder(Sequential)
    builder.add(Flatten(input_shape=(28, 28), name="input"))
    builder.add(
        Dense(128, activation="relu", name="hidden"),
        quantizer=UniformQuantizer(
            bits=bits,
            initializer=tf.keras.initializers.Constant(alpha),
            signed=signed,
        ),
    )
    builder.add(Dense(10, activation="softmax", name="output"))
    return builder.build()


def plot_training_history(history, callbacks):
    """Plot the training history including loss, accuracy, and alpha
    variables."""
    _, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Plot training & validation loss values
    axs[0].plot(history.history["loss"])
    axs[0].plot(history.history["val_loss"])
    axs[0].set_title("Model loss")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(["Train", "Validation"], loc="upper left")
    axs[0].grid(which="both")

    # Plot training & validation accuracy values
    axs[1].plot(history.history["accuracy"])
    axs[1].plot(history.history["val_accuracy"])
    axs[1].set_title("Model accuracy")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(["Train", "Validation"], loc="upper left")
    axs[1].grid(which="both")

    # Plot alpha history
    for callback in callbacks:
        axs[2].plot(callback.get_history(), label=callback.variable.name)
    axs[2].set_title("Alpha history")
    axs[2].set_ylabel("Alpha")
    axs[2].set_xlabel("Epoch")
    axs[2].legend([callback.variable.name for callback in callbacks])
    axs[2].grid(which="both")

    plt.tight_layout()
    plt.savefig("training_history.png")


def main(bits, alpha, signed):
    (x_train, y_train), (x_test, y_test) = generate_dataset()

    quant_aware_model = create_model(bits, alpha, signed)
    quant_aware_model.summary()
    quant_aware_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    callbacks = [
        VariableHistoryCallback(v)
        for v in quant_aware_model.variables
        if "alpha" in v.name
    ]

    hist = quant_aware_model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
        callbacks=[callbacks],
    )

    plot_training_history(hist, callbacks)

    quant_aware_model.evaluate(x_test, y_test)


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
        "--signed",
        action="store_true",
        help="flag to enable signed quantization",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs for training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size for training",
    )
    args = parser.parse_args()
    main(args.bits, args.alpha, args.signed)
