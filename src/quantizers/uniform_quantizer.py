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
    """An uniform quantizer algorithm support both signed and unsigned
    quantization.

    This class is ment to be used with the QuantizeConfig class to quantize the
    weights of a given layer. The quantization levels are uniformly distributed
    between the minimum and maximum values of the input tensor. The alpha
    parameter is learned during training.
    """

    def __init__(
        self,
        bits: int,
        signed: bool = True,
        name_suffix: str = "",
        initializer: tf.keras.initializers.Constant = tf.keras.initializers.Constant(
            1.0
        ),
        regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
    ):
        """Constructor.

        :param bits(int): Number of bits to use for quantization.
        :param signed(bool): Whether to use signed or unsigned quantization. By default, signed quantization is used.
        :param name_suffix(str): Suffix to append to the layer name.
        :param initializer(tf.keras.initializers.Initializer): Initializer for the alpha parameter.
        :param regularizer(tf.keras.regularizers.Regularizer): Regularizer for the alpha parameter.
        """
        super(UniformQuantizer, self).__init__()

        self.bits = bits
        self.signed = signed
        self.name_suffix = name_suffix
        self.initializer = initializer
        self.regularizer = regularizer

        self.n_levels = 2**self.bits

        self.alpha = None

    def build(self, tensor_shape, name: str, layer: tf.keras.layers.Layer):
        class PositiveConstraint(tf.keras.constraints.Constraint):
            """Constrains alpha to be positive."""

            def __call__(self, w):
                # Use epsilon to avoid dividing by zero during backpropagation.
                return tf.clip_by_value(w, tf.keras.backend.epsilon(), np.inf)

        alpha = layer.add_weight(
            name.join("_alpha"),
            initializer=self.initializer,
            trainable=True,
            dtype=tf.float32,
            regularizer=self.regularizer,
            constraint=PositiveConstraint(),
        )
        self.alpha = alpha
        return {"alpha": alpha}

    def __call__(self, inputs, training, weights, **kwargs):
        return self.quantize(inputs, weights["alpha"])

    def range(self):
        return 2 * self.alpha if self.signed else self.alpha

    def delta(self):
        return self.range() / self.n_levels

    def levels(self):
        """Compute the quantization levels."""
        start = -self.alpha if self.signed else 0
        return tf.range(start, start + self.range(), self.delta())

    @tf.custom_gradient
    def quantize(self, x, alpha):
        """Uniform quantization.

        :param x: input tensor
        :param alpha: alpha parameter
        :returns: quantized input tensor
        """
        # Capture alpha
        self.alpha = alpha

        # Compute quantization levels
        levels = self.levels()

        # Clip input values between min and max levels (function is zero outside the range)
        clipped_x = tf.clip_by_value(x, levels[0], levels[-1])

        # Quantize input values
        q = self.delta() * tf.math.floor(clipped_x / self.delta())

        def grad(upstream):
            # Gradient only flows through if the input is within range
            ## Use STE to estimate the gradient
            dq_dx = tf.where(
                tf.logical_and(
                    tf.greater_equal(x, levels[0]),
                    tf.less_equal(x, levels[-1]),
                ),
                upstream,
                tf.zeros_like(x),
            )

            # Compute the gradient for alpha
            dq_dalpha = tf.reduce_sum(q / alpha * upstream)

            return dq_dx, dq_dalpha

        return q, grad

    def get_config(self):
        return {
            "bits": self.bits,
            "signed": self.signed,
            "name_suffix": self.name_suffix,
            "initializer": tf.keras.initializers.serialize(self.initializer),
            "regularizer": tf.keras.regularizers.serialize(self.regularizer),
        }
