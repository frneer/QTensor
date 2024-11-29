#!/usr/bin/env python

import unittest

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from uniform_quantizer import UniformQuantizer


def plot_histogram(ax, tensor, label, min_clip, max_clip):
    """Plot histogram of tensor values with min_clip and max_clip lines."""
    ax.hist(tensor.numpy().flatten(), bins=20, alpha=0.75, label=label)
    ax.set_title(f"Histogram of {label}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.axvline(min_clip, color="r", linestyle="dashed", linewidth=2, label="Min clip")
    ax.axvline(max_clip, color="g", linestyle="dashed", linewidth=2, label="Max clip")
    ax.legend()


class TestUniformQuantizer(unittest.TestCase):
    """Test class for the UniformQuantizer class."""

    def setUp(self):
        self.input_shape = (4, 4)
        self.input_tensor = tf.constant(
            np.random.uniform(-3.0, 3.0, size=self.input_shape),
            dtype=tf.float32,
        )
        self.mock_layer = tf.keras.layers.Layer(name="test")

    def test_can_instantiate_uniform_quantizer(self):
        """Test that verifies that the UniformQuantizer can be instantiated."""
        quantizer = UniformQuantizer(
            bits=4,
            alpha_initializer=tf.keras.initializers.Constant(1.0),
            signed=True,
            name_suffix="_test",
        )

    def test_can_build_weights(self):
        """Test that verifies that the weights can be built."""
        quantizer = UniformQuantizer(
            bits=4,
            alpha_initializer=tf.keras.initializers.Constant(1.0),
            signed=True,
            name_suffix="_test",
        )
        weights = quantizer.build(self.input_shape, "test", self.mock_layer)
        self.assertDictEqual(weights, {"alpha": weights["alpha"]})

    def assert_weights_within_limits(self, bits, signed):
        """Test template to verify that all output values are within the range
        determined by alpha.

        :param bits(int): number of bits for quantization
        """
        # Build the quantizer weights
        quantizer = UniformQuantizer(
            bits=4,
            alpha_initializer=tf.keras.initializers.Constant(1.0),
            signed=signed,
            name_suffix="_test",
        )
        weights = quantizer.build(self.input_shape, "test", self.mock_layer)
        output = quantizer(self.input_tensor, training=True, weights=weights)

        # Retrieve the alpha value
        alpha = weights["alpha"].numpy()

        # Check that all output values are within the range determined by alpha
        min_clip = quantizer.min_clip(alpha)
        max_clip = quantizer.max_clip(alpha)

        # Plot histogram of input and output values
        _, axs = plt.subplots(2, 1, figsize=(10, 8))
        plot_histogram(axs[0], self.input_tensor, "Input values", min_clip, max_clip)
        plot_histogram(axs[1], output, "Output values", min_clip, max_clip)
        plt.tight_layout()
        plt.savefig(f'input_output_values_{"signed" if signed else "unsigned"}.png')

        # Assert all output values are within the range [min_clip, max_clip]
        self.assertLessEqual(
            np.max(output.numpy()),
            max_clip,
            f"Quantized values are above max_clip",
        )
        self.assertGreaterEqual(
            np.min(output.numpy()),
            min_clip,
            f"Quantized values are below min_clip",
        )

    def test_assert_weights_within_limits_signed(self):
        self.assert_weights_within_limits(bits=4, signed=True)

    def test_assert_weights_within_limits_unsigned(self):
        self.assert_weights_within_limits(bits=4, signed=False)

    def test_quantizer_unique_values(self):
        """Test that verifies that unique values are <= 2^bits."""
        bits = 4
        quantizer = UniformQuantizer(
            bits=bits,
            alpha_initializer=tf.keras.initializers.Constant(1.0),
            signed=True,
            name_suffix="_test",
        )

        # Build the quantizer weights
        self.input_shape = (4, 4)
        weights = quantizer.build(self.input_shape, "test", self.mock_layer)

        # Generate a random input tensor
        self.input_tensor = tf.constant(
            np.random.uniform(-3.0, 3.0, size=self.input_shape),
            dtype=tf.float32,
        )

        # Call the quantizer
        output = quantizer(self.input_tensor, training=True, weights=weights)

        # Check that all output values are unique
        self.assertLessEqual(
            len(np.unique(output.numpy())),
            2**bits,
            f"Quantized values are not unique",
        )


if __name__ == "__main__":
    unittest.main()
