#!/usr/bin/env python

"""This module implements a uniform quantizer for quantizing weights and
activations."""

from typing import Optional

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
        regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
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
        self.regularizer = regularizer

    def build(self, tensor_shape, name: str, layer: tf.keras.layers.Layer):
        alpha = layer.add_weight(
            name + "_alpha",
            initializer=tf.keras.initializers.Constant(self.alpha),
            trainable=True,
            dtype=tf.float32,
            regularizer=self.regularizer,
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

            """
            # Compute the quantization step size
            step_size = alpha / (quantization_levels // 2)

            # Generate quantization boundaries centered around 0
            k_values = tf.range(-quantization_levels // 2, quantization_levels // 2, dtype=tf.float32)
            boundaries = k_values * step_size

            # Define lower and upper boundaries for each interval
            lower_bounds = boundaries[:-1]  # Excludes the last boundary (used as lower limit)
            upper_bounds = boundaries[1:]   # Excludes the first boundary (used as upper limit)

            # Compute gradient scaling factors for each interval (normalized step size)
            gradient_scaling_factors = k_values[:-1] / (quantization_levels / 2 - 1)

            # Expand inputs and upstream for broadcasting across intervals
            inputs_expanded = tf.expand_dims(inputs, -1)
            upstream_expanded = tf.expand_dims(upstream, -1)

            # Compute the gradient contributions for all intervals at once
            grad_alpha_matrix = tf.where(
                tf.logical_and(
                    inputs_expanded >= lower_bounds,
                    inputs_expanded < upper_bounds
                ),
                gradient_scaling_factors * upstream_expanded,
                tf.zeros_like(inputs_expanded)
            )

            # Special handling for the first (left-most) and last (right-most) intervals
            left_most_contribution = tf.where(
                inputs < upper_bounds[0],
                gradient_scaling_factors[0] * upstream,
                tf.zeros_like(upstream)
            )[..., tf.newaxis]

            right_most_contribution = tf.where(
                inputs >= lower_bounds[-1],
                gradient_scaling_factors[-1] * upstream,
                tf.zeros_like(upstream)
            )[..., tf.newaxis]

            # Concatenate all contributions into a single tensor
            grad_alpha_matrix = tf.concat(
                [left_most_contribution, grad_alpha_matrix, right_most_contribution],
                axis=-1
            )

            # Sum over all interval contributions to produce the final gradient for `alpha`
            grad_alpha = tf.reduce_sum(grad_alpha_matrix, axis=-1)
            """
            # Compute gradient wrt alpha
            grad_alpha = tf.reduce_sum((quantized_output * upstream) / alpha)

            return grad_input, grad_alpha

        return quantized_output, grad

    def get_config(self):
        return {
            "bits": self.bits,
            "alpha": self.alpha,
            "signed": self.signed,
            "name_suffix": self.name_suffix,
        }
