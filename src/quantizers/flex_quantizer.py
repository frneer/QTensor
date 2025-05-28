"""This module implements a uniform quantizer for quantizing weights and
activations."""

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import (
    Quantizer,
    _QuantizeHelper,
)

from quantizers.common import max_value, min_value
from quantizers.constraints.constraints import (
    LevelConstraint,
    PositiveConstraint,
    ThresholdConstraint,
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
        n_levels: int,
        signed: bool = True,
        name_suffix: str = "",
    ):
        """Constructor.

        :param bits(int): Number of bits to use for quantization.
        :param signed(bool): Whether to use signed or unsigned quantization. By
            default, signed quantization is used.
        :param name(str): Suffix to append to the layer name.
        :param initializer(tf.keras.initializers.Initializer): Initializer for
            the alpha parameter.
        :param regularizer(tf.keras.regularizers.Regularizer): Regularizer for
            the alpha parameter.
        """
        super(FlexQuantizer, self).__init__()

        self.bits = bits
        self.signed = signed

        self.n_levels = n_levels
        self.m_levels = 2**self.bits

        self.alpha = None  # defines the range of the quantizer
        self.levels = None  # possible output values
        self.thresholds = None  # boundaries between levels

        self.name_suffix = name_suffix

    def build(self, tensor_shape, name: str, layer: tf.keras.layers.Layer):

        alpha = layer.add_weight(
            name=f"{name}{self.name_suffix}_alpha",
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            dtype=tf.float32,
            # regularizer=tf.keras.regularizers.L2(0.01),
            constraint=PositiveConstraint(),
        )
        self.alpha = alpha

        levels = layer.add_weight(
            name=f"{name}{self.name_suffix}_levels",
            initializer=tf.keras.initializers.Constant(
                np.linspace(
                    min_value(self.alpha, self.signed),
                    max_value(self.alpha, self.m_levels, self.signed),
                    self.n_levels,
                )
            ),
            shape=(self.n_levels,),
            trainable=True,
            dtype=tf.float32,
            constraint=LevelConstraint(self.alpha, self.m_levels, self.signed),
        )
        self.levels = levels

        thresholds = layer.add_weight(
            name=f"{name}{self.name_suffix}_thresholds",
            initializer=tf.keras.initializers.Constant(
                np.linspace(
                    min_value(self.alpha, self.signed),
                    self.alpha,
                    self.n_levels + 1,
                )
            ),
            shape=(self.n_levels + 1,),
            trainable=True,
            dtype=tf.float32,
            constraint=ThresholdConstraint(self.alpha, self.signed),
        )
        self.thresholds = thresholds

        return {"alpha": alpha, "levels": levels, "thresholds": thresholds}

    def __call__(self, inputs, training, weights, **kwargs):
        return self.quantize(
            inputs, weights["alpha"], weights["levels"], weights["thresholds"]
        )

    def range(self):
        return 2 * self.alpha if self.signed else self.alpha

    def delta(self):
        return self.range() / self.m_levels

    def quantize_op(self, x):
        # Quantize levels (uniform quantization)
        qlevels = self.delta() * tf.math.floor(self.levels / self.delta())
        # TODO(Colo): I think we can replace
        #   `qlevels = self.delta() * tf.math.floor(self.levels / self.delta())`
        # with
        #   `qlevels = self.qlevels`
        # and compute
        #   `self.qlevels = self.delta() * tf.math.floor(self.levels / self.delta())`
        # before
        #   `q = self.quantize_op(x)`
        # in the `quantize` function.

        # Quantize input
        q = tf.zeros_like(x)
        for i in range(self.thresholds.shape[0] - 1):
            q = tf.where(
                tf.logical_and(
                    tf.greater_equal(x, self.thresholds[i]),
                    tf.less_equal(x, self.thresholds[i + 1]),
                ),
                qlevels[i],
                q,
            )

        return q

    @tf.custom_gradient
    def quantize(self, x, alpha, levels, thresholds):
        # Capture the values of the parameters
        self.alpha = alpha
        self.levels = levels
        self.thresholds = thresholds

        q = self.quantize_op(x)

        qlevels = self.delta() * tf.math.floor(self.levels / self.delta())

        def grad(upstream):
            ##### dq_dx uses STE #####
            dq_dx = tf.where(
                tf.logical_and(
                    tf.greater_equal(x, self.thresholds[0]),
                    tf.less_equal(
                        x, self.thresholds[-1]
                    ),  # should it be alpha?
                ),
                upstream,
                tf.zeros_like(x),
            )

            ##### dq_dlevel #####

            # General idea is to match the input to a level and then
            # match that with the part of the upstream that is associated with that level
            # 3 levels, 4 inputs
            ## 1 0 0 -- > u1 0 0
            ## 1 0 0 -- > u2 0 0
            ## 0 0 1 -- > 0 0 u3
            ## 0 1 0 -- > 0 u4 0
            ## ==> dq_dlevel = [2, 1, 1]

            # So first we find the bin index of each input
            # x = [0.1, 0.3, 0.7, 1.5, 2.5] size = 5
            # q(x) = [0, 0.4, 1, 1, 3] size = 5
            # bin_indices = [0, 1, 2, 2, 3] size = 5
            bin_indices = tf.searchsorted(
                qlevels, tf.reshape(q, (-1,)), side="left"
            )

            # Then we one-hot encode the bin indices
            # q_one_hot =
            # 1 0 0 0
            # 0 1 0 0
            # 0 0 1 0
            # 0 0 1 0
            # 0 0 0 1
            q_one_hot = tf.one_hot(bin_indices, depth=qlevels.shape[0])

            # Finally we multiply the one-hot encoded bin indices with the upstream
            # transforming this into a 1D array
            dq_dlevel = tf.matmul(
                tf.transpose(q_one_hot), tf.reshape(upstream, (-1, 1))
            )
            dq_dlevel = tf.reshape(dq_dlevel, (-1,))

            ##### dq_dalpha is STE (same as in uniform quantizer) #####
            dq_dalpha = tf.reduce_sum(q / alpha * upstream)

            ##### dq_dthresholds using piecewise-STE #####
            dq_dthresholds = tf.zeros_like(thresholds)

            for i in range(1, self.thresholds.shape[0] - 1):
                delta_y = qlevels[i - 1] - qlevels[i]
                delta_x = thresholds[i + 1] - thresholds[i - 1]

                # Only those associated with the 'x' values that
                # Fall within the range of the two borderline levels
                masked_upstream = tf.where(
                    tf.logical_and(
                        tf.greater_equal(x, self.thresholds[i - 1]),
                        tf.less_equal(x, self.thresholds[i + 1]),
                    ),
                    upstream,
                    tf.zeros_like(x),
                )

                update_value = (
                    delta_y / delta_x * tf.reduce_sum(masked_upstream)
                )

                dq_dthresholds = tf.tensor_scatter_nd_update(
                    dq_dthresholds, indices=[[i]], updates=[update_value]
                )

            return dq_dx, dq_dalpha, dq_dlevel, dq_dthresholds

        return q, grad

    def get_config(self):
        return {
            "bits": self.bits,
            "signed": self.signed,
            "name_suffix": self.name_suffix,
            "initializer": self.initializer,
            "regularizer": self.regularizer,
        }
