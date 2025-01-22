#!/usr/bin/env python3

import unittest
from quantizers.uniform_quantizer import UniformQuantizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

class TestGenerateConfig(unittest.TestCase):
    def test_generate_config(self):
        config = GenerateConfig(
            UniformQuantizer(
                bits=8,
                signed=True,
            ),
        )

        layer = Dense(2, activation="relu")
        layer.build(input_shape=(5,))
        config.get_weights_and_quantizers(layer)
class TestBuilder(unittest.TestCase):
    def test_builder(self):
        builder = QBuilder(Sequential)
        model = builder \
            .add(Dense(2, activation="relu", name="hidden", input_shape=(2,)), quantizer=UniformQuantizer(bits=8, signed=True)) \
            .build()


if __name__ == "__main__":
    unittest.main()


if __name__ == '__main__':
    unittest.main()
