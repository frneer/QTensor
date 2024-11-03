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

            # Compute gradient for alpha
            grad_alpha = tf.zeros_like(inputs)
            step_size = alpha / (quantization_levels / 2)

            # Iterate through all intervals for gradient contributions
            for k in range(-quantization_levels // 2, quantization_levels // 2):
                lower_bound = k * step_size
                upper_bound = (k + 1) * step_size
                gradient_value = k / (quantization_levels / 2 - 1)  # Normalized gradient step based on position

                if k == -quantization_levels // 2:
                    # Special case for the left-most interval (-inf to the first upper limit)
                    grad_alpha += tf.where(
                        tf.less(inputs, upper_bound),
                        gradient_value * upstream,
                        tf.zeros_like(inputs)
                    )
                elif k == (quantization_levels // 2) - 1:
                    # Special case for the right-most interval (last lower limit to +inf)
                    grad_alpha += tf.where(
                        tf.greater_equal(inputs, lower_bound),
                        gradient_value * upstream,
                        tf.zeros_like(inputs)
                    )
                else:
                    # General case for intermediate intervals
                    grad_alpha += tf.where(
                        tf.logical_and(tf.greater_equal(inputs, lower_bound), tf.less(inputs, upper_bound)),
                        gradient_value * upstream,
                        tf.zeros_like(inputs)
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
