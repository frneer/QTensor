#!/usr/bin/env python

"""This module implements a uniform quantizer for quantizing weights and
activations."""

from typing import Optional

import numpy as np
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
        alpha_initializer: tf.keras.initializers.Constant,
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
        self.alpha_initializer = alpha_initializer
        self.signed = signed
        self.name_suffix = name_suffix
        self.regularizer = regularizer

        self.quantization_levels = 2**self.bits

    def build(self, tensor_shape, name: str, layer: tf.keras.layers.Layer):
        class PositiveConstraint(tf.keras.constraints.Constraint):
            def __call__(self, w):
                # Use epsilon instead of zero to avoid dividing by zero during backpropagation
                return tf.clip_by_value(w, tf.keras.backend.epsilon(), np.inf)

        alpha = layer.add_weight(
            name + "_alpha",
            initializer=self.alpha_initializer,
            trainable=True,
            dtype=tf.float32,
            regularizer=self.regularizer,
            constraint=PositiveConstraint(),
        )
        return {"alpha": alpha}

    def __call__(self, inputs, training, weights, **kwargs):
        return self.quantize_values(inputs, weights["alpha"])

    def min_clip(self, alpha):
        return -alpha if self.signed else 0

    def max_clip(self, alpha):
        return (
            alpha * (2 ** (self.bits - 1) - 1) / 2 ** (self.bits - 1)
            if self.signed
            else alpha
        )

    def scale_factor(self, alpha):
        scale_factor = self.quantization_levels / alpha
        if self.signed:
            scale_factor /= 2
        return scale_factor

    @tf.custom_gradient
    def quantize_values(self, inputs, alpha):
        """Uniform quantization.

        :param input: input tensor
        :returns: quantized input tensor
        """
        # Clip values between -alpha and alpha - 1
        min_clip = self.min_clip(alpha)
        max_clip = self.max_clip(alpha)
        clipped_inputs = tf.clip_by_value(inputs, min_clip, max_clip)

        # Do the actual quantization
        scaled_inputs = clipped_inputs * self.scale_factor(alpha)
        quantized_output = tf.math.floor(scaled_inputs) / self.scale_factor(alpha)

        # Compute custom gradient (STE)
        def grad(upstream):
            # Gradient only flows through if the input is within the clipping range
            grad_input = tf.where(
                tf.logical_and(
                    tf.greater_equal(inputs, min_clip),
                    tf.less_equal(inputs, max_clip),
                ),
                upstream,
                tf.zeros_like(inputs),
            )

            # Compute gradient wrt alpha
            grad_alpha = tf.reduce_sum((quantized_output * upstream) / alpha)

            return grad_input, grad_alpha

        return quantized_output, grad

    def get_config(self):
        return {
            "bits": self.bits,
            "alpha_initializer": self.alpha_initializer,
            "signed": self.signed,
            "name_suffix": self.name_suffix,
        }
