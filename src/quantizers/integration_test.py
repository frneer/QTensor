#!/usr/bin/env python3
import unittest

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from configs.qmodel import apply_quantization
from quantizers.uniform_quantizer import UniformQuantizer


class TestQuantizers(unittest.TestCase):
    def test_quantizers(self):
        model = Sequential(
            [
                Dense(10, activation="relu", input_shape=(20,), name="dense1"),
                Dense(5, activation="softmax"),
            ]
        )

        qconfig = {
            "dense1": {
                "weights": {
                    "kernel": UniformQuantizer(
                        bits=4,
                        signed=True,
                    ),
                    "bias": UniformQuantizer(
                        bits=4,
                        signed=True,
                    ),
                },
                "activations": {
                    "activation": UniformQuantizer(
                        bits=4,
                        signed=True,
                    )
                },
            }
        }

        apply_quantization(model, qconfig)
        # print(quantized_model.weights)


if __name__ == "__main__":
    unittest.main()
