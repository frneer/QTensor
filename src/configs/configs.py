"""Configs This file contains multiple useful quantizer configurations that can
be used in the quantization process.

Feel free to add more configurations as needed, or define them directly on your
code.
"""

from typing import List, Tuple

import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import (
    QuantizeConfig,
)
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import (
    Quantizer,
)

from quantizers.uniform_quantizer import UniformQuantizer


class UniformQuantizeConfig(QuantizeConfig):
    def __init__(
        self,
        bits: int = 8,
        alpha: float = 1.0,
        signed: bool = True,
    ):
        self.bits = bits
        self.alpha = alpha
        self.signed = signed

    # This defines how to quantize weights
    # TODO(Fran): How can I change the config for biases only?
    def get_weights_and_quantizers(
        self, layer
    ) -> List[Tuple[tf.Variable, Quantizer]]:
        return [
            (
                layer.kernel,
                UniformQuantizer(
                    bits=self.bits,
                    alpha=self.alpha,
                    signed=self.signed,
                    name_suffix="_kernel",
                ),
            )
        ]

    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]

    # This defines how to quantize activations
    def get_activations_and_quantizers(self, layer):
        return [
            (
                layer.activation,
                UniformQuantizer(
                    bits=self.bits,
                    alpha=self.alpha,
                    signed=self.signed,
                    name_suffix="_activation",
                ),
            )
        ]

    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]

    # This defines how to quantize outputs
    # TODO(Fran): WTF is this output is it before the activation function or after?
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {
            "bits": self.bits,
            "signed": self.signed,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
