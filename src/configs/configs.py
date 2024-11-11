"""Configs This file contains multiple useful quantizer configurations that can
be used in the quantization process.

Feel free to add more configurations as needed, or define them directly on your
code.
"""

from typing import List, Tuple, Optional

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
        regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    ):
        self.bits = bits
        self.alpha = alpha
        self.signed = signed
        self.regularizer = regularizer

        print(f"Using regularizer: {regularizer}")
        self.weight_quantizer = UniformQuantizer(
            bits=self.bits,
            alpha=self.alpha,
            signed=self.signed,
            name_suffix="_kernel",
            regularizer=self.regularizer
        )

        self.bias_quantizer = UniformQuantizer(
            bits=self.bits,
            alpha=self.alpha,
            signed=self.signed,
            name_suffix="_bias",
            regularizer=self.regularizer
        )

        self.activation_quantizer = UniformQuantizer(
            bits=self.bits,
            alpha=self.alpha,
            signed=self.signed,
            name_suffix="_activation",
            regularizer=self.regularizer
        )

    # This defines how to quantize weights
    def get_weights_and_quantizers(
        self, layer
    ) -> List[Tuple[tf.Variable, Quantizer]]:
        return [
            (
                layer.kernel,
                self.weight_quantizer,
            ),
            (
                layer.bias,
                self.bias_quantizer,
            )
        ]

    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]
        layer.bias = quantize_weights[1]

    # This defines how to quantize activations
    def get_activations_and_quantizers(self, layer):
        return [
            (
                layer.activation,
                self.activation_quantizer,
            )
        ]

    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]

    # This defines how to quantize outputs
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
