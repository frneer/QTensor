#!/usr/bin/env python3

import unittest
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import (
    QuantizeWrapperV2,
)

from configs.qmodel import apply_quantization
from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer
from utils.metrics import (
    compute_space_complexity_model,
    compute_space_complexity_quantize,
)


def apply_flex_dict(qmodel, alpha_dict, levels_dict, thresholds_dict):
    """Sets the internal state (alpha, levels, thresholds) of FlexQuantizers
    within a quantized model by directly finding and assigning to the live
    tf.Variable objects."""
    for layer in qmodel.layers:
        if not isinstance(layer, QuantizeWrapperV2):
            continue

        orig_layer_name = layer.layer.name
        if orig_layer_name in alpha_dict:
            # Find all variables in the wrapper layer and create a map by name
            var_map = {v.name: v for v in layer.weights}

            # Iterate through the types ('kernel', 'bias') we might want to change
            for attr_type in ["kernel", "bias"]:
                new_alpha = alpha_dict.get(orig_layer_name, {}).get(attr_type)
                new_levels = levels_dict.get(orig_layer_name, {}).get(
                    attr_type
                )
                new_thresholds = thresholds_dict.get(orig_layer_name, {}).get(
                    attr_type
                )

                # Construct the expected variable names and assign if they exist
                if new_alpha is not None:
                    # Note: TFMOT might name variables slightly differently.
                    # This searches for common patterns.
                    for name_pattern in [
                        f"/{attr_type}_alpha:0",
                        f"_{attr_type}_alpha:0",
                    ]:
                        var_name = layer.name + name_pattern
                        if var_name in var_map:
                            var_map[var_name].assign(new_alpha)

                if new_levels is not None:
                    for name_pattern in [
                        f"/{attr_type}_levels:0",
                        f"_{attr_type}_levels:0",
                    ]:
                        var_name = layer.name + name_pattern
                        if var_name in var_map:
                            var_map[var_name].assign(new_levels)

                if new_thresholds is not None:
                    for name_pattern in [
                        f"/{attr_type}_thresholds:0",
                        f"_{attr_type}_thresholds:0",
                    ]:
                        var_name = layer.name + name_pattern
                        if var_name in var_map:
                            var_map[var_name].assign(new_thresholds)


def check_weights(qlayer):
    qlayer_weights = qlayer.get_weights()
    qconfig = qlayer.quantize_config
    weights_and_quantizers = qconfig.get_weights_and_quantizers(qlayer.layer)
    weights_from_config = [
        weight_and_quantizer[0]
        for weight_and_quantizer in weights_and_quantizers
    ]
    quantizers = [
        weight_and_quantizer[1]
        for weight_and_quantizer in weights_and_quantizers
    ]

    qlayer_weights = qlayer.get_weights()

    # Original weights from the quantized layer
    weights_in_layer = qlayer_weights[0]
    print("Non-quantized weights in layer:")
    print(weights_in_layer)
    print()

    weights_quantized_from_config = weights_from_config[0].numpy()
    print("Weights from config:")
    print(weights_quantized_from_config)
    print()

    quantizer = quantizers[0] if quantizers else None
    print("Quantizer levels:")
    print(quantizer.levels.numpy())
    print()

    weights_manually_quantized = quantizer.quantize_op(
        qlayer_weights[0]
    ).numpy()
    print("Manually quantized weights:")
    print(weights_manually_quantized)
    print()

    assert np.array_equal(
        weights_quantized_from_config, weights_manually_quantized
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

    def test_understanding_weights(self):
        """Verify that we can access the weights of a quantized layer."""
        layer = tf.keras.layers.Dense(10, input_shape=(5,), name="dense_1")
        layer.build((None, 5))
        qconfig = {
            "dense_1": {
                "weights": {
                    "kernel": FlexQuantizer(bits=2, n_levels=4, signed=True),
                    "bias": FlexQuantizer(bits=2, n_levels=4, signed=True),
                },
            },
        }
        model = tf.keras.Sequential([layer])
        qmodel = apply_quantization(model, qconfig)
        qmodel.build((None, 5))
        # Run an inference to have access to the variables.
        qmodel(tf.random.normal((1, 5)))
        # Access the quantized layer
        qlayer = qmodel.get_layer("quant_dense_1")

        check_weights(qlayer)
        # Define the quantizer parameters based on our ideal state
        alpha = 1.0
        # 2. Calculate the expected complexity based on a known data distribution
        ideal_levels = np.array([-0.8, -0.2, 0.3, 0.9], dtype=np.float32)
        midpoints = (ideal_levels[:-1] + ideal_levels[1:]) / 2.0
        thresholds = np.concatenate(([-alpha], midpoints, [alpha])).astype(
            np.float32
        )

        apply_flex_dict(
            qmodel,
            alpha_dict={"dense_1": {"kernel": alpha}},
            levels_dict={"dense_1": {"kernel": ideal_levels}},
            thresholds_dict={"dense_1": {"kernel": thresholds}},
        )
        qmodel(tf.random.normal((1, 5)))  # Force weight creation
        check_weights(qlayer)

    def test_compute_space_complexity_flex_only(self):
        """Verify that for a flex configuration a layer size is as expected."""
        # 1. Setup the initial layer and model
        layer = tf.keras.layers.Dense(
            10, input_shape=(5,), name="dense_1", use_bias=False
        )
        model = tf.keras.Sequential([layer])
        model.build((None, 5))

        # Define the FlexQuantizer configuration
        qconfig = {
            "dense_1": {
                "weights": {
                    "kernel": FlexQuantizer(bits=4, n_levels=4, signed=True),
                },
            },
        }

        # 2. Calculate the expected complexity based on a known data distribution
        ideal_levels = np.array([-0.8, -0.2, 0.3, 0.9], dtype=np.float32)
        ideal_weight_data = np.random.choice(
            ideal_levels, size=(5, 10), replace=True
        )

        counter = Counter(ideal_weight_data.flatten())
        total_elements = sum(counter.values())
        emp_probs = np.array(list(counter.values())) / total_elements
        entropy = -np.sum(emp_probs * np.log2(emp_probs))

        huffman_size = ideal_weight_data.size * entropy
        levels_size = len(ideal_levels) * 4  # n_levels * bits
        expected_size = huffman_size + levels_size

        # 3. Apply quantization to get the qmodel structure
        qmodel = apply_quantization(model, qconfig)

        # 4. Force weight creation by calling the model with a dummy input.
        dummy_input_shape = (1,) + model.input_shape[1:]
        qmodel(tf.random.normal(dummy_input_shape))

        # 5. Inject the known state into the qmodel
        q_layer = qmodel.get_layer("quant_dense_1")

        # Find the actual tf.Variable for the kernel.
        kernel_var = None
        for v in q_layer.trainable_weights:
            if v.name.endswith("kernel:0"):
                kernel_var = v
                break

        self.assertIsNotNone(
            kernel_var, "Could not find the kernel variable to assign."
        )
        kernel_var.assign(ideal_weight_data)

        kernel_weights = qmodel.get_layer("quant_dense_1").get_weights()
        print("Kernel Weights before assignment:")
        print(kernel_weights)

        # Define the quantizer parameters based on our ideal state
        alpha = 1.0
        # KEY CHANGE: Calculate thresholds correctly to match the quantizer's expected variable shape.
        # The shape should be (n_levels + 1) to include outer bounds.
        midpoints = (ideal_levels[:-1] + ideal_levels[1:]) / 2.0
        thresholds = np.concatenate(([-alpha], midpoints, [alpha])).astype(
            np.float32
        )

        # Use our helper to set the FlexQuantizer's internal state
        apply_flex_dict(
            qmodel,
            alpha_dict={"dense_1": {"kernel": alpha}},
            levels_dict={"dense_1": {"kernel": ideal_levels}},
            thresholds_dict={"dense_1": {"kernel": thresholds}},
        )
        qmodel(tf.random.normal(dummy_input_shape))  # Force assigment

        # 6. Compute the quantized size using the metric function
        quantized_size = compute_space_complexity_quantize(q_layer)

        # 7. Assert that the computed size matches the theoretical expected size
        self.assertAlmostEqual(quantized_size, expected_size, places=6)

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
