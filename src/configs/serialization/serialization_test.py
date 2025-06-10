#!/usr/bin/env python3
import tempfile
import unittest

from tensorflow import keras

from configs.qmodel import apply_quantization
from configs.serialization.serialization import load_qmodel, save_qmodel
from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer


class TestSerialization(unittest.TestCase):
    def test_single_uniform(self):
        input_shape = [20]

        model = keras.Sequential(
            [
                keras.layers.Dense(
                    20, input_shape=input_shape, name="dense_1"
                ),
                keras.layers.Flatten(),
            ]
        )

        qconfig = {
            "dense_1": {
                "weights": {
                    "kernel": UniformQuantizer(8, name_suffix="_asdasd"),
                },
            }
        }
        quant_aware_model = apply_quantization(model, qconfig)

        quant_aware_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model_dir = tempfile.mkdtemp()
        save_qmodel(quant_aware_model, model_dir)

        load_qmodel(model_dir)

    def test_single_flex(self):
        input_shape = [20]

        model = keras.Sequential(
            [
                keras.layers.Dense(
                    20, input_shape=input_shape, name="dense_1"
                ),
                keras.layers.Flatten(),
            ]
        )

        qconfig = {
            "dense_1": {
                "weights": {
                    "kernel": FlexQuantizer(
                        bits=8, n_levels=10, signed=True, name_suffix="_asdasd"
                    ),
                },
            }
        }
        quant_aware_model = apply_quantization(model, qconfig)

        quant_aware_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model_dir = tempfile.mkdtemp()
        save_qmodel(quant_aware_model, model_dir)

        load_qmodel(model_dir)

    def test_lenet_5(self):
        model = keras.models.Sequential(
            [
                keras.layers.Conv2D(
                    filters=6,
                    kernel_size=(5, 5),
                    activation="relu",
                    padding="same",
                    input_shape=(28, 28, 1),
                ),
                keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),
                keras.layers.Conv2D(
                    filters=16, kernel_size=(5, 5), activation="relu"
                ),
                keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2),
                keras.layers.Flatten(),
                keras.layers.Dense(120, activation="relu"),
                keras.layers.Dense(84, activation="relu"),
                keras.layers.Dense(
                    10, activation="softmax"
                ),  # 10 classes (digits 0-9)
            ]
        )

        qconfig = {
            "conv2d": {
                "weights": {
                    "kernel": FlexQuantizer(bits=4, n_levels=10, signed=True)
                },
                "activations": {
                    "activation": UniformQuantizer(bits=4, signed=False)
                },
            },
            "conv2d_1": {
                "weights": {
                    "kernel": FlexQuantizer(bits=4, n_levels=10, signed=True)
                },
                "activations": {
                    "activation": UniformQuantizer(bits=4, signed=False)
                },
            },
            "dense": {
                "weights": {
                    "kernel": FlexQuantizer(bits=4, n_levels=10, signed=True)
                },
                "activations": {
                    "activation": UniformQuantizer(bits=4, signed=False)
                },
            },
            "dense_1": {
                "weights": {
                    "kernel": FlexQuantizer(bits=4, n_levels=10, signed=True)
                },
                "activations": {
                    "activation": UniformQuantizer(bits=4, signed=False)
                },
            },
            "dense_2": {
                "weights": {
                    "kernel": FlexQuantizer(bits=4, n_levels=10, signed=True)
                },
                "activations": {
                    "activation": UniformQuantizer(bits=4, signed=False)
                },
            },
        }

        quant_aware_model = apply_quantization(model, qconfig)

        quant_aware_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model_dir = tempfile.mkdtemp()
        save_qmodel(quant_aware_model, model_dir)

        load_qmodel(model_dir)


if __name__ == "__main__":
    unittest.main()
