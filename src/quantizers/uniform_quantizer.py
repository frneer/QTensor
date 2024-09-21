#!/usr/bin/env python

"""This module implements a uniform quantizer for quantizing weights and
activations."""


import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import (
    Quantizer,
    _QuantizeHelper,
)


class UniformQuantizer(_QuantizeHelper, Quantizer):
    """This class implements an uniform quantizer."""

    def __init__(
        self,
        bits: int,
        alpha: float,
        signed: bool = True,
        name_suffix: str = "",
    ):
        """
        :param bits: number of bits for quantization
        :param alpha: initial quantization range
        :param signed: Flag to enable signed quantization
        :param name_suffix: suffix to be added to the layer qParameter names
        """
        super(UniformQuantizer, self).__init__()

        self.bits = bits
        self.alpha = alpha
        self.signed = signed
        self.name_suffix = name_suffix

    def build(self, tensor_shape, name, layer):
        alpha = layer.add_weight(
            name + "_alpha",
            initializer=tf.keras.initializers.Constant(self.alpha),
            trainable=True,
            dtype=tf.float32,
        )
        return {"alpha": alpha}

    def __call__(self, inputs, training, weights, **kwargs):
        min_clip = -weights["alpha"] if self.signed else 0
        max_clip = weights["alpha"]
        clipped = tf.clip_by_value(inputs, min_clip, max_clip)
        return self.quantize_values(clipped, weights["alpha"])

    @tf.custom_gradient
    def quantize_values(self, input, alpha):
        """Uniform quantization.

        :param input: input tensor
        :returns: quantized input tensor
        """
        quantization_levels = 2**self.bits

        scale_factor = quantization_levels / alpha
        if self.signed:
            scale_factor /= 2
        scaled_input = input * scale_factor

        quantized_output = tf.math.floor(scaled_input) / scale_factor

        def grad(upstream):
            # Gradient for inputs is STE
            grad_input = tf.ones_like(input) * upstream

            # Transparent gradient for alpha
            # TODO(Fran): This might not be right.
            grad_alpha = tf.zeros_like(alpha)

            return grad_input, grad_alpha

        return quantized_output, grad

    def get_config(self):
        return {
            "bits": self.bits,
            "alpha": self.alpha,
            "signed": self.signed,
            "name_suffix": self.name_suffix,
        }
