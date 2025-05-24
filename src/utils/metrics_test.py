#!/usr/bin/env python3

import unittest

import tensorflow as tf

from configs.qmodel import apply_quantization
from quantizers.uniform_quantizer import UniformQuantizer
from utils.metrics import (
    compute_space_complexity_model,
    compute_space_complexity_quantize,
)


# From tensorflow internal code
def _compute_memory_size(weight):
    weight_counts = weight.shape.num_elements()
    per_param_size = weight.dtype.size
    return weight_counts * per_param_size


def weight_memory_size(weights):
    """Compute the memory footprint for weights based on their dtypes.

    Args:
        weights: An iterable contains the weights to compute weight size.

    Returns:
        The total memory size (in bits) of the weights.
    """
    unique_weights = {id(w): w for w in weights}.values()
    total_memory_size = 0
    for w in unique_weights:
        total_memory_size += _compute_memory_size(w)
    return total_memory_size


class TestMetrics(unittest.TestCase):
    def test_compute_space_complexity_uniform_only(self):
        """Verify that for a uniform configuration a layer size is as
        expected."""
        layer = tf.keras.layers.Dense(10, input_shape=(5,), name="dense_1")
        layer.build((None, 5))  # Build the layer to initialize weights
        qconfig = {
            "dense_1": {
                "weights": {
                    "kernel": UniformQuantizer(bits=4, signed=True),
                    "bias": UniformQuantizer(bits=4, signed=True),
                },
            },
        }
        model = tf.keras.Sequential([layer])

        qmodel = apply_quantization(model, qconfig)
        qmodel.build((None, 5))
        # Run an inference to have access to the variables.
        qmodel(tf.random.normal((1, 5)))
        # Compute quantized size
        quantized_size = compute_space_complexity_quantize(qmodel.layers[1])

        kernel_expected_size = (
            layer.kernel.shape.num_elements() * 4
        )  # 4 bits for kernel
        bias_expected_size = layer.bias.shape.num_elements() * 4
        expected_size = kernel_expected_size + bias_expected_size

        self.assertEqual(quantized_size, expected_size)

    def test_compute_non_quantized_model(self):
        """Verify that computing the size of the model."""
        layer = tf.keras.layers.Dense(30, input_shape=(5,), name="dense_1")
        layer.build((None, 5))
        model = tf.keras.Sequential([layer])

        size = compute_space_complexity_model(model) / 8  # To bytes
        size_according_to_tensorflow = weight_memory_size(model.weights)
        self.assertEqual(size, size_according_to_tensorflow)

    def test_compare(self):
        def test_verify_proportional_to_base_size(bits):
            layer = tf.keras.layers.Dense(10, input_shape=(5,), name="dense_1")
            layer.build((None, 5))  # Build the layer to initialize weights
            qconfig = {
                "dense_1": {
                    "weights": {
                        "kernel": UniformQuantizer(bits=bits, signed=True),
                        "bias": UniformQuantizer(bits=bits, signed=True),
                    },
                },
            }
            model = tf.keras.Sequential([layer])
            model.build((None, 5))
            model(tf.random.normal((1, 5)))

            qmodel = apply_quantization(model, qconfig)
            qmodel.build((None, 5))
            qmodel(tf.random.normal((1, 5)))
            non_quantized_size = compute_space_complexity_model(model)
            quantized_size = compute_space_complexity_model(qmodel)

            # We expect weights size proportionally smaller
            size_scale_factor = bits / 32
            self.assertEqual(
                quantized_size, non_quantized_size * size_scale_factor
            )

        for bits in [2, 4, 6, 8, 10, 12, 16]:
            with self.subTest(val=bits):
                test_verify_proportional_to_base_size(bits)


if __name__ == "__main__":
    unittest.main()
