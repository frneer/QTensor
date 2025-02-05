#!/usr/bin/env python

"""This module implements a uniform quantizer for quantizing weights and
activations."""


import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import (
    Quantizer,
    _QuantizeHelper,
)


class FlexQuantizer(_QuantizeHelper, Quantizer):
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
        super(FlexQuantizer, self).__init__()

        self.bits = bits
        self.signed = signed
        self.name = name
        self.alpha = None  # this is the range of the quantizer
        self.levels = None  # these are possible output values
        self.thresholds = None  # these are the boundaries between levels

        self.n_levels = 2**self.bits

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
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            dtype=tf.float32,
            regularizer=None,
            constraint=PositiveConstraint(),
        )

        levels = layer.add_weight(
            name.join("_levels"),
            initializer=tf.keras.initializers.RandomUniform(-alpha, alpha),
            shape=(self.n_levels,),
            trainable=True,
            dtype=tf.float32,
            constraint=ClippedAndOrderedConstraint(alpha),
        )

        thresholds = layer.add_weight(
            name.join("_thresholds"),
            initializer=tf.keras.initializers.RandomUniform(-alpha, alpha),
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

        return self.quantize(inputs, weights["alpha"], weights["levels"], weights["thresholds"])

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
        qlevels = self.quantize_levels(self.levels, alpha)


        # Handle special case -alpha, th[0]
        low_limit = -self.alpha if self.signed else 0

        # Quantize input values
        q = tf.zeros_like(x)
        q = tf.where(
            tf.logical_and(
                tf.greater(x, low_limit), tf.less(x, self.thresholds[0])
            ),
            qlevels[0],
            q
        )
        # |---|---|---|
        # Are there more efficients ways to do clustering?

        for i in range(self.thresholds.shape[0] - 1):
            q = tf.where(
                tf.logical_and(
                    tf.greater_equal(x, self.thresholds[i]),
                    tf.less_equal(x, self.thresholds[i + 1]),
                ),
                qlevels[i + 1],
                q
            )

        def grad(upstream):
            ## dq_dx --> STE
            dq_dx = tf.where(
                tf.logical_and(
                    tf.greater_equal(x, low_limit),
                    tf.less_equal(x, self.thresholds[-1]),
                ),
                upstream,
                tf.zeros_like(x),
            )

            #  dq_dlevel --> ... amount of weights in the level
            # Example:
            # 3 levels, 4 inputs
            ## 1 0 0 -- > u1 0 0
            ## 1 0 0 -- > u2 0 0
            ## 0 0 1 -- > 0 0 u3
            ## 0 1 0 -- > 0 u4 0
            ## ==> dq_dlevel = [2, 1, 1]
            # Maybe use digitize?
            bin_indices = tf.searchsorted(qlevels, tf.reshape(q, (-1,)), side='left')
            print(f"BIN_INDICES shape: {bin_indices.shape}")
            # x = [0.1, 0.3, 0.7, 1.5, 2.5] size = 5
            # q(x) = [0, 0.4, 1, 1, 3] size =5
            # q_digit = [0, 1, 2, 2, 3] size = 5
            q_one_hot = tf.one_hot(bin_indices, depth=qlevels.shape[0])
            # q_one_hot =
            # 1 0 0 0
            # 0 1 0 0
            # 0 0 1 0
            # 0 0 1 0
            # 0 0 0 1
            print(f"Q_ONE_HOT shape: {q_one_hot.shape}")
            print(f"UPSTREAM shape: {upstream.shape}")
            print(f"QLEVELS shape: {qlevels.shape}")
            print(f"UPSTEAM with reshaped: {tf.reshape(upstream, (-1, 1)).shape}")
            dq_dlevel = tf.matmul(tf.transpose(q_one_hot), tf.reshape(upstream, (-1, 1)))
            dq_dlevel = tf.reshape(dq_dlevel, (-1,))
            print(f"DQ_DL shape: {dq_dlevel.shape}")
            print(f"DQ_DL: {dq_dlevel}")
            # dq_dlevel = tf.zeros_like(qlevels)


            # Handle special case alpha (it's a threshold)
            # Think about composing the thresholds and alphafor this?
            dq_dalpha = 0
            delta_y = qlevels[1] - qlevels[0]
            delta_x = thresholds[1] - thresholds[0]
            slope = delta_y / delta_x
            # idx = tf.where(q == qlevels[0])  # # Only those on that level, right?
            dq_dalpha = slope * tf.reduce_sum(upstream)

            # dq_dthresholds --> piecewise-STE
            dq_dthresholds = tf.zeros_like(thresholds)

            for i in range(self.thresholds.shape[0]):
                # i + 2 because first level is already handled
                delta_y = qlevels[i + 2] - qlevels[i + 1]
                delta_x = thresholds[i + 1] - thresholds[i]

                dq_dthresholds[i] = slope * tf.reduce_sum(upstream)


            return dq_dx, dq_dalpha, dq_dlevel, dq_dthresholds

        return q, grad


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
