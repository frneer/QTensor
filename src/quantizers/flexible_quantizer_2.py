#!/usr/bin/env python

"""This module implements a uniform quantizer for quantizing weights and
activations."""


import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import (
    Quantizer,
    _QuantizeHelper,
)


class FlexibleQuantizer(_QuantizeHelper, Quantizer):
    """An flexible quantizer algorithm support both signed and unsigned
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
        name: str = "",
    ):
        """Constructor.

        :param bits(int): Number of bits to use for quantization.
        :param signed(bool): Whether to use signed or unsigned quantization. By default, signed quantization is used.
        :param name(str): Suffix to append to the layer name.
        :param initializer(tf.keras.initializers.Initializer): Initializer for the alpha parameter.
        :param regularizer(tf.keras.regularizers.Regularizer): Regularizer for the alpha parameter.
        """
        super(FlexibleQuantizer, self).__init__()

        self.bits = bits
        self.signed = signed
        self.name = name
        self.alpha = None  # this is the range of the quantizer
        self.levels = None  # these are possible output values
        self.thresholds = None  # these are the boundaries between levels

    def build(self, tensor_shape, name: str, layer: tf.keras.layers.Layer):
        class PositiveConstraint(tf.keras.constraints.Constraint):
            """Constrains alpha to be positive."""

            def __call__(self, w):
                # Use epsilon to avoid dividing by zero during backpropagation.
                return tf.clip_by_value(w, tf.keras.backend.epsilon(), np.inf)

        class ClippedAndOrderedConstraint(tf.keras.constraints.Constraint):
            """Constrains the values to be ordered."""

            def __init__(self, alpha):
                self.alpha = alpha

            def __call__(self, w):
                ret = tf.clip_by_value(w, -self.alpha, self.alpha)
                return tf.sort(ret)

        alpha = layer.add_weight(
            name.join("_alpha"),
            initializer=self.initializer,
            trainable=True,
            dtype=tf.float32,
            regularizer=self.regularizer,
            constraint=PositiveConstraint(),
        )

        levels = layer.add_weight(
            name.join("_levels"),
            shape=(self.n_levels,),
            trainable=True,
            dtype=tf.float32,
            constraint=ClippedAndOrderedConstraint(alpha),
        )

        thresholds = layer.add_weight(
            name.join("_thresholds"),
            shape=(self.n_levels - 1,),
            trainable=True,
            dtype=tf.float32,
            constraint=ClippedAndOrderedConstraint(alpha),
        )
        self.alpha = alpha
        self.levels = levels
        self.thresholds = thresholds
        return {"alpha": alpha, "levels": levels, "thresholds": thresholds}

    def __call__(self, inputs, training, weights, **kwargs):
        return self.quantize(inputs, **weights)

    def range(self):
        return 2 * self.alpha if self.signed else self.alpha

    def delta(self):
        return self.range() / self.n_levels

    # def levels(self):
    #     """Compute the quantization levels."""
    #     start = -self.alpha if self.signed else 0
    #     return tf.range(start, start + self.range(), self.delta())

    @tf.custom_gradient
    def quantize(self, x, alpha, levels, thresholds):
        # Capture alpha
        self.alpha = alpha

        # Capture quantization levels
        self.levels = levels

        # Capture thresholds
        self.thresholds = thresholds

        # Do we want to save the quantization values? I think not...
        # This is just for forward pass
        # self.levels = self.quantize_levels(alpha)
        levels = self.quantize_levels(x, alpha)

        # Quantize input values
        q = tf.zeros_like(x)

        # Handle special case -alpha, th[0]
        low_limit = -self.alpha if self.signed else 0
        idx = tf.where(
            tf.logical_and(
                tf.less(x, self.thresholds[0]), tf.greater(x, low_limit)
            )
        )
        # |---|---|---|
        # More efficients ways to do this.

        q[idx] = self.levels[0]

        for i in range(self.thresholds):
            idx = tf.where(
                tf.logical_and(
                    tf.greater_equal(x, self.thresholds[i]),
                    tf.less_equal(x, self.thresholds[i + 1]),
                )
            )

            q[idx] = self.levels[i + 1]

        # Set derivatives
        # dq_dx --> STE
        # dq_dlevel --> ... I don't know (amount of weights in the level)
        ## 3 niveles, 4 entradas
        ## 1 0 0
        ## 1 0 0
        ## 0 0 1
        ## 0 1 0
        ## dq_delevel = [2, 1, 1]
        ##
        # dq_dthresholds
        # STE por partes delta_y
        # por cada level
        # Implement these like q vs t0, dominion entre los niveles circundantes del threshold

    @tf.custom_gradient
    def quantize_levels(self, x, alpha):
        """Uniform quantization.

        :param x: input tensor
        :param alpha: alpha parameter
        :returns: quantized input tensor
        """
        # Quantize input values
        # Problem is that levels doesn't have the upper limit set right
        # That was handled by tf range
        q = self.delta() * tf.math.floor(self.levels / self.delta())

        def grad(upstream):
            # Gradient only flows through if the input is within range
            ## Use STE to estimate the gradient
            dq_dx = tf.where(
                tf.logical_and(
                    tf.greater_equal(x, self.levels[0]),
                    tf.less_equal(x, self.levels[-1]),
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
            "initializer": self.initializer,
            "regularizer": self.regularizer,
        }
