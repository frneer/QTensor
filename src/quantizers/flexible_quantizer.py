#!/usr/bin/env python

"""This module implements a uniform quantizer for quantizing weights and
activations."""


import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import (
    Quantizer,
    _QuantizeHelper,
)

from typing import Iterable, Optional
import numpy as np

class FlexibleQuantizer(_QuantizeHelper, Quantizer):
    """This class implements an uniform quantizer."""

    def __init__(
        self,
        bits: int,
        alpha: float,
        n_clusters: Optional[int] = None,
        cc: Optional[Iterable] = None,
        cl: Optional[Iterable] = None,
        signed: Optional[bool] = True,
        name_suffix: str = "",
    ):
        """
        :param bits: number of bits for quantization
        :param alpha: initial quantization range
        :param signed: Flag to enable signed quantization
        :param n_clusters: number of clusters for quantization
        :param cc: cluster centers
        :param cl: cluster limits
        :param name_suffix: suffix to be added to the layer qParameter names

        """
        super(FlexibleQuantizer, self).__init__()

        # Enforce exclusivity between n_clusters and (cc, cl)
        if n_clusters is not None and (cc is not None or cl is not None):
            raise ValueError("Provide either 'n_clusters' or ('cc', 'cl'), but not both.")
        if n_clusters is None and (cc is None or cl is None):
            raise ValueError("You must provide both 'cc' and 'cl' if 'n_clusters' is not provided.")


        self.bits = bits
        self.alpha = alpha
        print("alpha", alpha)
        self.signed = signed
        self.n_ch = 1  # TODO(Fran): Extend this for cnn; probably need to cross check the size with the layer being quantized.

        self.name_suffix = name_suffix

        self.n_clusters = n_clusters if n_clusters else len(cc)
        width = alpha / self.n_clusters  # signed/unsigned?
        if signed:
            width = 2 * width

        if cc:
            assert len(cc) == self.n_clusters, "cc must have length 2**bits"
        self.cc = cc if cc else np.linspace(-alpha + width / 2, alpha - width / 2, self.n_clusters)

        if cl:
            assert len(cl) == self.n_clusters-1, "cl must have length 2**bits-1"
        self.cl = cl if cl else np.linspace(-alpha + width / 2, alpha - width / 2, self.n_clusters-1)

    def build(self, tensor_shape, name, layer):

        class OrderedConstraint(tf.keras.constraints.Constraint):
            def __init__(self, alpha):
                self.alpha = alpha
            def __call__(self, variable):
                # Sort the variable values along the last axis (assumed to be the feature axis)
                # sorted_variable = tf.sort(variable, axis=0)
                # low_limit = sorted_variable[0]
                # high_limit = sorted_variable[-1]
                # hard_limit = tf.maximum(abs(low_limit), abs(high_limit))


                # Ensure that the weights stay between min_val and max_val
                return tf.clip_by_value(variable, -self.alpha, self.alpha)

            def get_config(self):
                return {}
        class PositiveConstraint(tf.keras.constraints.Constraint):
            def __call__(self, w):
                # Use epsilon instead of zero to avoid dividing by zero during backpropagation
                return tf.clip_by_value(w, tf.keras.backend.epsilon(), np.inf)


        alpha = layer.add_weight(
            name + '_alpha',
            initializer=tf.keras.initializers.Constant(self.alpha),
            trainable=True,
            dtype=tf.float32,
            constraint=PositiveConstraint(),
        )
        cc = layer.add_weight(
                name + '_cluster_centers',
                shape=(self.n_clusters, self.n_ch),
                initializer=tf.constant_initializer(self.cc),
                trainable=True,
                constraint=OrderedConstraint(alpha),
                )
        cl = layer.add_weight(
                name + '_cluster_limits',
                shape=(self.n_clusters-1, self.n_ch),
                initializer=tf.constant_initializer(self.cl),
                trainable=True,
                constraint=OrderedConstraint(alpha),
                )

        return {"cc": cc, "cl": cl}

    def __call__(self, inputs, training, weights, **kwargs):
        alpha = tf.reduce_max(tf.abs(weights["cc"]))
        # min_clip = -alpha if self.signed else 0
        # max_clip = alpha
        # clipped = tf.clip_by_value(inputs, min_clip, max_clip)
        clipped = inputs
        return self.quantize_values(clipped, alpha, weights["cc"], weights["cl"])

    # @tf.custom_gradient
    def quantize_values(self, input, alpha, cc, cl):
        """Flexible quantization function."""
        # Porque tenemos un parametro alpha si dsp usamos el CC para esto.
        # No deberia ser cluster liit? no va entre centro  y centro o value y value.
        # Quantize centers
        ccq = self.quantize_centers(cc, alpha)

        # Quantize values
        wq  = self.quantize_weights(input, ccq, cl)

        return wq

    @tf.custom_gradient
    def quantize_centers(self, input, alpha):
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

    @tf.custom_gradient
    def quantize_weights(self, wc, ccq, cl):
        """
        Quantize weights
        """
        # Todo replace the implementation with a more efficient one
        ch = 0
        wq = tf.multiply(tf.ones(tf.shape(wc)), ccq[0, ch])

        for i in range(0, self.n_clusters - 1):
            condition = tf.greater(wc, cl[i, ch])
            wq = tf.where(condition, ccq[i + 1, ch], wq)

        flat_indices = tf.searchsorted(cl, tf.reshape(wc, [-1]), side='right') - 1

        # Gather the cluster values using the matched indices
        # matched_values = tf.gather(ccq, indices)
        # matched_values = tf.squeeze(matched_values, axis=-1)


        def grad(upstream):
            # Gradient for inputs is STE
            grad_input = tf.ones_like(wc) * upstream

            # Gradient for cluster centers
            grad_ccq = tf.zeros_like(ccq)
            for cluster_center in range(self.n_clusters - 1):  # Use tf.shape to get the size
                # Create a condition tensor for occurrences of each cluster center
                condition = tf.equal(wc, ccq[cluster_center])

                # Count the occurrences
                n_occur = tf.reduce_sum(tf.cast(condition, tf.float32))

                # Update the gradient for the current cluster center
                grad_ccq = tf.tensor_scatter_nd_update(grad_ccq, [[cluster_center]], [[n_occur]])

            # Gradient for cluster limits
            grad_cl = tf.zeros_like(cl)





            return grad_input, grad_ccq, grad_cl

        return wq, grad

    def get_config(self):
        return {
            "bits": self.bits,
            "alpha": self.alpha,
            "signed": self.signed,
            "name_suffix": self.name_suffix,
        }
