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
        return self.quantize_values(inputs, weights["alpha"])

    @tf.custom_gradient
    def quantize_values(self, inputs, alpha):
        """Uniform quantization.

        :param input: input tensor
        :returns: quantized input tensor
        """

        # Clip values between -alpha and alpha - 1
        min_clip = - alpha
        max_clip = alpha * (2**(self.bits - 1) - 1) / 2**(self.bits - 1)
        # if not self.signed:
        #     min_clip = 0
        #     max_clip = alpha
        clipped_inputs = tf.clip_by_value(inputs, min_clip, max_clip)

        # Do the actual quantization
        quantization_levels = 2**self.bits
        scale_factor = quantization_levels / alpha
        scale_factor /= 2  # it's signed
        scaled_inputs = clipped_inputs * scale_factor
        quantized_output = tf.math.floor(scaled_inputs) / scale_factor

        # Compute custom gradient (STE)
        def grad(upstream):
            # Gradient only flows through if the input is within the clipping range
            grad_input = tf.where(
                    tf.logical_and(
                        tf.greater_equal(inputs, min_clip),
                        tf.less_equal(inputs, max_clip),
                    ),
                    upstream,
                    tf.zeros_like(inputs))

            # If values is within min and max clip: grad_alpha = 0,
            # else < min_clip: grad_alpha = -upstream, else > max_clip: grad_alpha = upstream
            # Finally reduce sum to match the shape of alpha
            grad_alpha = tf.where(
                tf.less(inputs, min_clip),
                -upstream,  # Negative gradient when inputs are less than min_clip
                tf.where(
                    tf.greater(inputs, max_clip),
                    upstream,  # Positive gradient when inputs are greater than max_clip
                    tf.zeros_like(inputs)  # No gradient when inputs are within bounds
                )
            )
            grad_alpha = tf.reduce_sum(grad_alpha)

            return grad_input, grad_alpha

        return quantized_output, grad

    def get_config(self):
        return {
            "bits": self.bits,
            "alpha": self.alpha,
            "signed": self.signed,
            "name_suffix": self.name_suffix,
        }
