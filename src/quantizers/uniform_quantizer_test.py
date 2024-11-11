#!/usr/bin/env python

import unittest

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow_model_optimization.quantization.keras import (
    quantize_annotate_layer,
    quantize_apply,
    quantize_scope,
)
from uniform_quantizer import UniformQuantizer

from configs.configs import UniformQuantizeConfig


class TestUniformQuantizer(unittest.TestCase):
    def setUp(self) -> None:
        model = Sequential(
            [
                quantize_annotate_layer(
                    Dense(10, activation="relu", input_shape=(10,)),
                    UniformQuantizeConfig(alpha=1),
                ),
            ]
        )
        with quantize_scope({"UniformQuantizeConfig": UniformQuantizeConfig}):
            quantize_apply(model)

    def test_can_instantiate_uniform_quantizer(self):
        UniformQuantizer(bits=8, alpha=1, signed=True)

    # TODO(Fran): We may need to add more tests here.


if __name__ == "__main__":
    unittest.main()
