#!/usr/bin/env python3
from typing import List

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import newaxis
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import History
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from builder.builder import QBuilder, GenerateConfig

from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer

from utils.utils import VariableHistoryCallback
from quantizers.uniform_quantizer import UniformQuantizer

def generate_dataset():
    """Generate the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train[..., newaxis]
    x_test = x_test[..., newaxis]
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def plot_training_history(
    history: History, callbacks: List[VariableHistoryCallback]
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

    # Adjust layout to leave space for the legend and text
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("training_history.png")

def lenet5_model(bits: int = 8, hidden_bits: int = 8):
    """Create a LeNet-5 model for MNIST classification."""
    builder = QBuilder(Sequential)
    builder.add(
        Conv2D(6, kernel_size=(5, 5), activation="relu", input_shape=(28, 28, 1)),
        weight_quantizer=UniformQuantizer(
            bits=bits,
            signed=True,
            initializer=tf.keras.initializers.Constant(0.1),
        ),
        activation_quantizer=UniformQuantizer(
            bits=bits,
            signed=False,
            initializer=tf.keras.initializers.Constant(0.1),
        )
    )
    builder.add(MaxPooling2D(pool_size=(2, 2)))
    builder.add(
        Conv2D(16, kernel_size=(5, 5), activation="relu"),
        weight_quantizer=UniformQuantizer(
            bits=hidden_bits,
            signed=True,
            initializer=tf.keras.initializers.Constant(0.1),
        ),
        activation_quantizer=UniformQuantizer(
            bits=hidden_bits,
            signed=False,
            initializer=tf.keras.initializers.Constant(0.1),
        )
    )
    builder.add(MaxPooling2D(pool_size=(2, 2)))
    builder.add(Flatten())
    builder.add(
        Dense(120, activation="relu"),
        weight_quantizer=UniformQuantizer(
            bits=hidden_bits,
            signed=True,
            initializer=tf.keras.initializers.Constant(0.1),
        ),
        activation_quantizer=UniformQuantizer(
            bits=hidden_bits,
            signed=False,
            initializer=tf.keras.initializers.Constant(0.1),
        )
    )
    builder.add(
        Dense(84, activation="relu"),
        weight_quantizer=UniformQuantizer(
            bits=hidden_bits,
            signed=True,
            initializer=tf.keras.initializers.Constant(0.1),
        ),
        activation_quantizer=UniformQuantizer(
            bits=hidden_bits,
            signed=False,
            initializer=tf.keras.initializers.Constant(0.1),
        )
    )
    builder.add(Dense(10))
    builder.add(Activation("softmax"))

    return builder.build()

(x_train, y_train), (x_test, y_test) = generate_dataset()

quant_aware_model = lenet5_model(bits=6, hidden_bits=5)

quant_aware_model.summary()
quant_aware_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
callbacks = [
    VariableHistoryCallback(v)
    for v in quant_aware_model.variables
    if "alpha" in v.name
]

EPOCHS = 10
BATCH_SIZE = 256*1
hist = quant_aware_model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test),
    callbacks=[callbacks],
)

plot_training_history(hist, callbacks)

quant_aware_model.evaluate(x_test, y_test)
