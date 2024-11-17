#!/usr/bin/env python3


import argparse
from typing import List

import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import History
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow_model_optimization.quantization.keras import (
    quantize_annotate_layer,
    quantize_annotate_model,
    quantize_apply,
    quantize_scope,
)

from configs.configs import UniformQuantizeConfig
from utils.utils import VariableHistoryCallback


def generate_dataset():
    """Generate the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def simple_model(bits, alpha, signed):
    """Create a simple model for MNIST classification."""
    layer_1 = Flatten(input_shape=(28, 28), name="input")
    layer_2 = quantize_annotate_layer(
        Dense(128, activation="relu", name="hidden"),
        UniformQuantizeConfig(
            bits=bits,
            alpha=alpha,
            signed=signed,
        ),
    )
    layer_3 = Dense(10, activation="softmax", name="output")
    model = quantize_annotate_model(Sequential([layer_1, layer_2, layer_3]))
    return model


def lenet5_model(bits, alpha, signed):
    """Create a LeNet-5 model for MNIST classification."""
    model = Sequential()
    model.add(
        quantize_annotate_layer(
            Conv2D(6, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1)),
            UniformQuantizeConfig(
                bits=bits,
                alpha=alpha,
                signed=signed,
            ),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(
        quantize_annotate_layer(
            Conv2D(16, kernel_size=(5, 5), activation="relu"),
            UniformQuantizeConfig(
                bits=bits,
                alpha=alpha,
                signed=signed,
            ),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(
        quantize_annotate_layer(
            Dense(120, activation="relu"),
            UniformQuantizeConfig(
                bits=bits,
                alpha=alpha,
                signed=signed,
            ),
        )
    )
    model.add(
        quantize_annotate_layer(
            Dense(84, activation="relu"),
            UniformQuantizeConfig(
                bits=bits,
                alpha=alpha,
                signed=signed,
            ),
        )
    )
    model.add(
        quantize_annotate_layer(
            Dense(10, activation="softmax"),
            UniformQuantizeConfig(
                bits=bits,
                alpha=alpha,
                signed=signed,
            ),
        )
    )
    return model


models = {
    "lenet5": lenet5_model,
    "simple_dense": simple_model,
}


def plot_training_history(
    history: History, callbacks: List[VariableHistoryCallback], args: argparse.Namespace
):
    """Plot the training history including loss, accuracy, and alpha variables.

    Args:
        history (History): A Keras History object containing the training and validation loss and accuracy.
        callbacks (list): A list of callback objects that have a `get_history` method and a `variable` attribute.
        args (Namespace): A namespace object containing the configuration arguments for the script.
    """
    fig, axs = plt.subplots(3, 1, figsize=(20, 20))

    # Increase font sizes for readability
    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12

    # Plot training & validation loss values
    axs[0].plot(history.history["loss"], label="Train", linewidth=2)
    axs[0].plot(history.history["val_loss"], label="Validation", linewidth=2)
    axs[0].set_title("Model Loss", fontsize=title_fontsize)
    axs[0].set_ylabel("Loss", fontsize=label_fontsize)
    axs[0].set_xlabel("Epoch", fontsize=label_fontsize)
    axs[0].legend(loc="upper right", fontsize=label_fontsize)
    axs[0].grid(which="both", linestyle="--", linewidth=0.5)
    axs[0].tick_params(axis="both", labelsize=tick_fontsize)

    # Plot training & validation accuracy values
    axs[1].plot(history.history["accuracy"], label="Train", linewidth=2)
    axs[1].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
    axs[1].set_title("Model Accuracy", fontsize=title_fontsize)
    axs[1].set_ylabel("Accuracy", fontsize=label_fontsize)
    axs[1].set_xlabel("Epoch", fontsize=label_fontsize)
    axs[1].legend(loc="lower right", fontsize=label_fontsize)
    axs[1].grid(which="both", linestyle="--", linewidth=0.5)
    axs[1].tick_params(axis="both", labelsize=tick_fontsize)

    # Plot alpha history
    for callback in callbacks:
        axs[2].plot(callback.get_history(), label=callback.variable.name, linewidth=1.5)
    axs[2].set_title("Alpha History", fontsize=title_fontsize)
    axs[2].set_ylabel("Alpha", fontsize=label_fontsize)
    axs[2].set_xlabel("Epoch", fontsize=label_fontsize)
    axs[2].legend(loc="upper left", fontsize=label_fontsize, bbox_to_anchor=(1.05, 1))
    axs[2].grid(which="both", linestyle="--", linewidth=0.5)
    axs[2].tick_params(axis="both", labelsize=tick_fontsize)

    # Add text describing the args
    description = (
        f"Model: {args.model}\n"
        f"Bits: {args.bits}\n"
        f"Alpha: {args.alpha}\n"
        f"Signed: {args.signed}\n"
        f"Epochs: {args.epochs}\n"
        f"Batch size: {args.batch_size}"
    )
    plt.gcf().text(
        0.6,
        0.5,
        description,
        fontsize=12,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.5),
    )  # Adjust coordinates and style as needed

    # Adjust layout to leave space for the legend and text
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("training_history.png")


def main(args):
    (x_train, y_train), (x_test, y_test) = generate_dataset()
    model = models.get(args.model)(args.bits, args.alpha, args.signed)

    # Compile the model and get the variables to monitor
    with quantize_scope({"UniformQuantizeConfig": UniformQuantizeConfig}):
        quant_aware_model = quantize_apply(model)
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

    plot_training_history(hist, callbacks, args)

    quant_aware_model.evaluate(x_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=models.keys(),
        default="simple_dense",
        help="model architecture to use: 'lenet5' or 'simple_dense'",
    )
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
    main(args)
