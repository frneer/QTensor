#!/usr/bin/env python

import unittest

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from quantizers.uniform_quantizer import UniformQuantizer


def plot_histogram(ax, tensor, label, min, max):
    """Plot histogram of tensor values with min and max lines."""
    ax.hist(tensor.numpy().flatten(), bins=20, alpha=0.75, label=label)
    ax.set_title(f"Histogram of {label}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.axvline(
        min, color="r", linestyle="dashed", linewidth=2, label="Min clip"
    )
    ax.axvline(
        max, color="g", linestyle="dashed", linewidth=2, label="Max clip"
    )
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
        UniformQuantizer(
            bits=3,
            initializer=tf.keras.initializers.Constant(1.0),
            signed=True,
            name_suffix="_test",
        )

    def test_can_build_weights(self):
        """Test that verifies that the weights can be built."""
        quantizer = UniformQuantizer(
            bits=4,
            signed=True,
            name_suffix="_test",
        )
        weights = quantizer.build(self.input_shape, "test", self.mock_layer)
        self.assertDictEqual(
            weights, {"alpha": weights["alpha"], "levels": weights["levels"]}
        )

    # TODO(Fran): Consider using a fixture here?
    def assert_weights_within_limits(self, bits, signed):
        """Test template to verify that all output values are within the range
        determined by alpha.

        :param bits(int): number of bits for quantization
        """
        # Build the quantizer weights
        quantizer = UniformQuantizer(
            bits=bits,
            signed=signed,
            name_suffix="_test",
        )
        weights = quantizer.build(self.input_shape, "test", self.mock_layer)
        output = quantizer(self.input_tensor, training=True, weights=weights)

        # Check that all output values are within the range determined by alpha
        quantizer_levels = quantizer.compute_levels()
        min = quantizer_levels[0]
        max = quantizer_levels[-1]

        # Plot histogram of input and output values
        _, axs = plt.subplots(2, 1, figsize=(10, 8))
        plot_histogram(axs[0], self.input_tensor, "Input values", min, max)
        plot_histogram(axs[1], output, "Output values", min, max)
        plt.tight_layout()
        plt.savefig(
            f'input_output_values_{"signed" if signed else "unsigned"}.png'
        )

        # Assert all output values are within the range.
        self.assertLessEqual(
            np.max(output.numpy()),
            max,
            f"Quantized values are above the quantizer expected range",
        )
        self.assertGreaterEqual(
            np.min(output.numpy()),
            min,
            f"Quantized values are below the quantizer expected range",
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

    def test_expected_levels(self):
        """Test that the levels generated are as expected."""
        quantizer = UniformQuantizer(
            bits=3,
            initializer=tf.keras.initializers.Constant(1.0),
            signed=True,
            name_suffix="_test",
        )

        quantizer.build(self.input_shape, "test", self.mock_layer)

        levels = quantizer.compute_levels()
        expected_n_levels = 2**3
        self.assertEqual(levels.shape.num_elements(), expected_n_levels)

        expected_levels = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
        self.assertListEqual(list(levels), expected_levels)

    def test_quantizer_levels_getitem(self):
        # Verify that the __getitem__ method works as expected
        # With both positive and negative indices
        quantizer = UniformQuantizer(
            bits=3,
            initializer=tf.keras.initializers.Constant(1.0),
            signed=True,
            name_suffix="_test",
        )

        quantizer.build(self.input_shape, "test", self.mock_layer)

        levels = quantizer.compute_levels()
        self.assertEqual(levels[0], -1.0)
        self.assertEqual(levels[2], -0.5)
        self.assertEqual(levels[7], 0.75)
        self.assertEqual(levels[-1], 0.75)

    def test_expected_levels_reflects_in_output_signed(self):
        """Test that verifies that the output values are within the expected
        levels."""
        bits = 3
        alpha = 3.0
        quantizer = UniformQuantizer(
            bits=bits,
            initializer=tf.keras.initializers.Constant(alpha),
            signed=True,
            name_suffix="_test",
        )

        # Build the quantizer weights
        self.input_shape = (10000, 4)
        weights = quantizer.build(self.input_shape, "test", self.mock_layer)

        # Generate a random input tensor
        self.input_tensor = tf.constant(
            np.random.uniform(-3.0, 3.0, size=self.input_shape),
            dtype=tf.float32,
        )

        # Call the quantizer
        output = quantizer(self.input_tensor, training=True, weights=weights)
        output_set = sorted(set(output.numpy().flatten()))
        expected_set = list(quantizer.compute_levels())

        self.assertListEqual(output_set, expected_set)

    def test_expected_levels_reflects_in_output_unsigned(self):
        """Test that verifies that the output values are within the expected
        levels."""
        bits = 3
        alpha = 3.0
        quantizer = UniformQuantizer(
            bits=bits,
            initializer=tf.keras.initializers.Constant(alpha),
            signed=False,
            name_suffix="_test",
        )

        # Build the quantizer weights
        self.input_shape = (10000, 4)
        weights = quantizer.build(self.input_shape, "test", self.mock_layer)

        # Generate a random input tensor
        self.input_tensor = tf.constant(
            np.random.uniform(0.0, 3.0, size=self.input_shape),
            dtype=tf.float32,
        )

        # Call the quantizer
        output = quantizer(self.input_tensor, training=True, weights=weights)
        output_set = sorted(set(output.numpy().flatten()))
        expected_set = list(quantizer.compute_levels())

        self.assertListEqual(output_set, expected_set)


if __name__ == "__main__":
    unittest.main()
