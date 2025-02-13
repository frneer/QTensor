#!/usr/bin/env python3


import argparse

from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from configs.qmodel import apply_quantization
from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer
from utils.utils import VariableHistoryCallback, plot_snapshot


def generate_dataset():
    """Generate the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def create_model_and_qconfig(args):
    if args.example_name == "mlp":
        layer_1 = Flatten(input_shape=(28, 28), name="input")
        layer_2 = Dense(128, activation="relu", name="hidden")
        layer_3 = Dense(10, activation="softmax", name="output")
        model = Sequential([layer_1, layer_2, layer_3])

        qconfig = {
                "hidden": {
                    "weights": {"kernel": FlexQuantizer(bits=args.bits, n_levels=args.levels , signed=True)},
                    "activations": {"activation": UniformQuantizer(bits=args.bits, signed=False)},
                }
            }

    if args.example_name == "lenet":
        model = Sequential([
            Conv2D(
                filters=6,
                kernel_size=(5, 5),
                activation="relu",
                padding="same",
                input_shape=(28, 28, 1)
                ),
            AveragePooling2D(pool_size=(2, 2), strides=2),
            Conv2D(
                filters=16,
                kernel_size=(5, 5),
                activation="relu"
                ),
            AveragePooling2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(120, activation="relu"),
            Dense(84, activation="relu"),
            Dense(10, activation="softmax")  # 10 classes (digits 0-9)
        ])

        qconfig = {
                "conv2d": {
                    "weights": {"kernel": FlexQuantizer(bits=args.bits, n_levels=args.levels , signed=True)},
                    "activations": {"activation": UniformQuantizer(bits=args.bits, signed=False)},
                },
                "conv2d_1": {
                    "weights": {"kernel": FlexQuantizer(bits=args.bits, n_levels=args.levels , signed=True)},
                    "activations": {"activation": UniformQuantizer(bits=args.bits, signed=False)},
                },
                "dense": {
                    "weights": {"kernel": FlexQuantizer(bits=args.bits, n_levels=args.levels , signed=True)},
                    "activations": {"activation": UniformQuantizer(bits=args.bits, signed=False)},
                },
                "dense_1": {
                    "weights": {"kernel": FlexQuantizer(bits=args.bits, n_levels=args.levels , signed=True)},
                    "activations": {"activation": UniformQuantizer(bits=args.bits, signed=False)},
                },
                "dense_2": {
                    "weights": {"kernel": FlexQuantizer(bits=args.bits, n_levels=args.levels , signed=True)},
                    "activations": {"activation": UniformQuantizer(bits=args.bits, signed=False)},
                },
            }

    return model, qconfig


def main(args):
    model, qconfig = create_model_and_qconfig(args)

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    qmodel.summary(line_length=120)
    hist = model.fit(
        x_train,
        y_train,
        epochs=2,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
    )

    qmodel = apply_quantization(model, qconfig)
    qmodel.summary(line_length=120)
    qmodel.compile(
        optimizer=Adam(learning_rate=0.001 / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        VariableHistoryCallback(v)
        for v in qmodel.variables
        if "alpha" in v.name
    ]
    print([w.name for w in model.layers[1].weights])
    print([w.name for w in qmodel.layers[1].weights])

    alpha_callback = [
        VariableHistoryCallback(v)
        for v in qmodel.layers[1].weights
        if "alpha" in v.name
    ][0]
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
    hist = qmodel.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )

    plot_training_history(hist, callbacks)
    plot_snapshot(
        alpha_hist=alpha_callback.get_history(),
        level_hist=levels_callback.get_history(),
        threshold_hist=thresholds_callback.get_history(),
        accuracy_hist=hist.history["accuracy"],
        signed=True,
        output_path="snapshots",
        bits=args.bits,
    )

    qmodel.evaluate(x_test, y_test)


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
        "--batch-size",
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
        "--example-name",
        choices=['mlp', 'lenet',],
        type=str,
        default='mlp',
        help="name of the example to be used",
    )
    args = parser.parse_args()
    main(args)

