#!/usr/bin/env python3

import unittest
from collections import Counter

import numpy as np
import tensorflow as tf
from metrics import (
    compute_huffman_nominal_complexity,
    compute_space_complexity_model,
    compute_space_complexity_quantize,
)
from tensorflow.keras import layers, models

from configs.qmodel import apply_quantization
from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer


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


class TestComputeHuffmanNominalComplexity(unittest.TestCase):
    """Unit tests for `compute_huffman_nominal_complexity`."""

    def test_all_identical(self):
        """Entropy should be zero when every symbol is the same."""
        q = tf.constant([7, 7, 7, 7], dtype=tf.int32)
        expected = 0.0  # N=4 => 0 bits
        self.assertAlmostEqual(
            compute_huffman_nominal_complexity(q), expected, places=6
        )

    def test_balanced_two_symbols(self):
        """50 % / 50 % distribution => entropy = 1 bit per symbol."""
        q = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        expected = 4.0  # N=4 => 4 bits
        self.assertAlmostEqual(
            compute_huffman_nominal_complexity(q), expected, places=6
        )

    def test_three_to_one_ratio(self):
        """75 % / 25 % distribution => entropy ~0.811278 bits per symbol."""
        q = tf.constant([0, 0, 0, 1], dtype=tf.int32)
        entropy = -(
            0.75 * np.log2(0.75) + 0.25 * np.log2(0.25)
        )  # ~0.811278 bits
        expected = 4 * entropy  # N=4 => ~3.2451 bits
        self.assertAlmostEqual(
            compute_huffman_nominal_complexity(q), expected, places=6
        )

    def test_larger_vector_distribution(self):
        """100 elements: 50*0, 30*1, 20*2 =>"""
        # build the vector
        vals = [0] * 50 + [1] * 30 + [2] * 20
        q = tf.constant(vals, dtype=tf.int32)

        # compute expected: N=100, p0=0.5, p1=0.3, p2=0.2
        ps = np.array([0.5, 0.3, 0.2])
        entropy = -np.sum(ps * np.log2(ps))  # ~1.485475
        expected = 100 * entropy  # N=100 => ~148.5475

        self.assertAlmostEqual(
            compute_huffman_nominal_complexity(q), expected, places=6
        )


def apply_alpha_dict(qmodel, alpha_dict):
    """TODO(Colo): This function will is implemented in branch
    colo/model_evalution in QTensor/src/examples/functions.py.

    When merged, import that functions insted of redefining it here.
    """
    for layer in qmodel.layers:
        orig_layer_name = layer.name
        if orig_layer_name.startswith("quant_"):
            orig_layer_name = orig_layer_name[len("quant_") :]

        if orig_layer_name in alpha_dict:
            for alpha_type in ["kernel", "bias", "activation"]:
                new_alpha = alpha_dict[orig_layer_name].get(alpha_type, None)
                if new_alpha is not None:
                    for v in layer.weights:
                        if "alpha" in v.name and alpha_type in v.name:
                            v.assign(new_alpha)
                            # print(f"Updated {v.name} ({alpha_type}) with new alpha value {new_alpha}")
                        elif (
                            alpha_type == "activation"
                            and "post_activation" in v.name
                            and "alpha" in v.name
                        ):
                            v.assign(new_alpha)
                            # print(f"Updated {v.name} (activation) with new alpha value {new_alpha}")

    return qmodel


class TestLeNetQuantizedComplexity(unittest.TestCase):
    def setUp(self):
        # build a small LeNet-5 for, say, 10 classes over 28×28×1 inputs
        categories = 10
        input_shape = (None, 28, 28, 1)
        self.model = models.Sequential(
            [
                layers.Conv2D(
                    6,
                    kernel_size=5,
                    activation="relu",
                    padding="same",
                    name="conv2d",
                ),
                layers.AveragePooling2D(name="pool1"),
                layers.Conv2D(
                    16, kernel_size=5, activation="relu", name="conv2d_1"
                ),
                layers.AveragePooling2D(name="pool2"),
                layers.Flatten(name="flatten"),
                layers.Dense(120, activation="relu", name="dense"),
                layers.Dense(84, activation="relu", name="dense_1"),
                layers.Dense(categories, activation="softmax", name="dense_2"),
            ]
        )
        self.model.build(input_shape)
        self.model.compile(
            loss="categorical_crossentropy", metrics=["accuracy"]
        )
        # run one dummy inference so any lazy weights are created
        self.model(tf.random.normal((1, 28, 28, 1)))
        self.kernel_shape = list()
        self.bias_shape = list()
        self.kernel_size = list()
        self.bias_size = list()
        for layer in self.model.layers:
            if hasattr(layer, "kernel"):
                shape = layer.kernel.shape
                self.kernel_shape.append(shape)
                self.kernel_size.append(shape.num_elements())
            if hasattr(layer, "bias"):
                shape = layer.bias.shape
                self.bias_shape.append(shape)
                self.bias_size.append(shape.num_elements())

    def test_uniform_quantizer_space_complexity(self):
        """All weights quantized using uniform quantizer."""
        layer_names = ["conv2d", "conv2d_1", "dense", "dense_1", "dense_2"]
        kernel_bits = [7, 6, 5, 4, 3]
        bias_bits = [3, 4, 5, 6, 7]
        kernel_alphas = [1.0] * 5
        bias_alphas = [1.0] * 5

        # 1) define an 8‐bit uniform quantizer on every kernel
        qconfig = {}
        for i, layer in enumerate(layer_names):
            qconfig[layer] = dict()
            qconfig[layer]["weights"] = dict()
            qconfig[layer]["weights"]["kernel"] = UniformQuantizer(
                bits=kernel_bits[i], signed=True
            )
            qconfig[layer]["weights"]["bias"] = UniformQuantizer(
                bits=bias_bits[i], signed=True
            )
        # for key in qconfig:
        #    print(f'{key}: {qconfig[key]}')

        # 4) compute expected size
        expected_bits = 0
        for kb, ks, bb, bs in zip(
            kernel_bits, self.kernel_size, bias_bits, self.bias_size
        ):
            expected_bits += kb * ks + bb * bs

        # 2) apply quantization and build
        qmodel = apply_quantization(self.model, qconfig)
        qmodel.build(self.model.input_shape)
        qmodel(tf.random.normal((1, 28, 28, 1)))

        # 3) set alphas
        alpha_dict = {
            layer_name: {"kernel": kalpha, "bias": balpha}
            for layer_name, kalpha, balpha in zip(
                qconfig, kernel_alphas, bias_alphas
            )
        }
        apply_alpha_dict(qmodel, alpha_dict)

        # 5) Check result
        computed_bits = compute_space_complexity_model(qmodel)
        print(f"expected_bits={expected_bits}")
        print(f"computed_bits={computed_bits}")
        self.assertEqual(computed_bits, expected_bits)

    def test_flex_quantizer_space_complexity(self):
        """All weights quantized using flexible quantizer."""

        def random_probability_vector(n, epsilon=1e-8):
            vec = (
                np.random.rand(n) + epsilon
            )  # ensure all elements are strictly > 0
            return vec / vec.sum()

        def equal_probability_vector(n, epsilon=1e-8):
            vec = np.ones(n)
            return vec / vec.sum()

        def increasing_probability_vector(n, epsilon=1e-8):
            vec = np.arange(1, n + 1)
            return vec / vec.sum()

        layer_names = ["conv2d", "conv2d_1", "dense", "dense_1", "dense_2"]
        kernel_bits = [7, 6, 5, 4, 3]
        bias_bits = [3, 4, 5, 6, 7]
        # kernel_n_levels = [25, 12, 13, 5, 2]
        # bias_n_levels = [3, 7, 15, 7, 14]
        kernel_n_levels = [2] * 5  # TEST: for levels = 2
        bias_n_levels = [2] * 5  # TEST: for levels = 2
        kernel_alphas = [1.0] * 5
        bias_alphas = [1.0] * 5
        kernel_probabilities = []
        bias_probabilities = []
        for kl, bl in zip(kernel_n_levels, bias_n_levels):
            # kernel_probabilities.append(random_probability_vector(kl))
            # bias_probabilities.append(random_probability_vector(bl))
            # kernel_probabilities.append(equal_probability_vector(kl)) # TEST: equiprobabilities
            # bias_probabilities.append(equal_probability_vector(bl)) # TEST: equiprobabilities
            kernel_probabilities.append(
                increasing_probability_vector(kl)
            )  # TEST: increasing probabilities
            bias_probabilities.append(
                increasing_probability_vector(bl)
            )  # TEST: increasing probabilities

        # 1) build a qconfig where every layer's kernel & bias uses a FlexQuantizer
        qconfig = {}
        for i, layer in enumerate(layer_names):
            qconfig[layer] = dict()
            qconfig[layer]["weights"] = dict()
            qconfig[layer]["weights"]["kernel"] = FlexQuantizer(
                bits=kernel_bits[i], n_levels=kernel_n_levels[i], signed=True
            )
            qconfig[layer]["weights"]["bias"] = FlexQuantizer(
                bits=bias_bits[i], n_levels=bias_n_levels[i], signed=True
            )
        # for key in qconfig:
        #    print(f'{key}: {qconfig[key]}')

        # 2) compute expected total bits
        expected_bits = 0.0
        kernels = []
        biases = []
        # pack both “kernel” and “bias” data into a single list of groups
        groups = [
            (
                self.kernel_shape,
                self.kernel_size,
                kernel_bits,
                kernel_n_levels,
                kernel_alphas,
                kernel_probabilities,
                kernels,
            ),
            (
                self.bias_shape,
                self.bias_size,
                bias_bits,
                bias_n_levels,
                bias_alphas,
                bias_probabilities,
                biases,
            ),
        ]
        for (
            shape_list,
            size_list,
            bits_list,
            levels_list,
            alphas_list,
            probs_list,
            container,
        ) in groups:
            for shape, size, bits, n_levels, alpha, probs in zip(
                shape_list,
                size_list,
                bits_list,
                levels_list,
                alphas_list,
                probs_list,
            ):
                # 1) build the set of valid quantized values
                valid_values = np.linspace(-alpha, alpha, num=2**bits + 1)[:-1]

                # 2) pick exactly `n_levels` of them, then sample your weight‐vector
                values = np.sort(
                    np.random.choice(
                        valid_values, size=n_levels, replace=False
                    )
                )
                vector = np.random.choice(
                    values, size=size, replace=True, p=probs
                )
                weight = vector.reshape(shape)

                # store the weight‐tensor
                container.append(weight)

                # 3) recompute empirical probabilities from the sampled weights
                counter = Counter(weight.flatten())
                sorted_items = sorted(counter.items())
                counter_keys, counter_values = zip(*sorted_items)
                emp_probs = np.array(counter_values) / sum(counter_values)
                print(f"probs={probs}, emp_probs={emp_probs}")

                # 4) entropy and Huffman bits
                entropy = -np.sum(emp_probs * np.log2(emp_probs))
                expected_bits += size * entropy
                expected_bits += n_levels * bits

                # 5) sanity checks
                #    a) all entries are in the valid set
                mask = np.isin(weight, valid_values)
                assert np.all(
                    mask
                ), f"These values are not valid: {weight[~mask]}"
                #    b) no more unique levels than n_levels
                unique_vals = np.unique(weight)
                assert unique_vals.size <= n_levels, (
                    f"Expected <= {n_levels} unique values, but found "
                    f"{unique_vals.size}: {unique_vals}"
                )

        # 3) Set weights to the model
        weights = list()
        for k, b in zip(kernels, biases):
            weights.append(k)
            weights.append(b)
        self.model.set_weights(weights)

        # 4) apply quantization & init everything
        qmodel = apply_quantization(self.model, qconfig)
        qmodel.build(self.model.input_shape)
        qmodel(tf.random.normal((1, 28, 28, 1)))

        # 5) set alphats
        alpha_dict = {
            layer_name: {"kernel": kalpha, "bias": balpha}
            for layer_name, kalpha, balpha in zip(
                qconfig, kernel_alphas, bias_alphas
            )
        }
        apply_alpha_dict(qmodel, alpha_dict)

        # 6) compare to your implementation
        computed_bits = compute_space_complexity_model(qmodel)
        print(f"expected_bits={expected_bits}")
        print(f"computed_bits={computed_bits}")
        # self.assertEqual(computed_bits, expected_bits)
        self.assertAlmostEqual(computed_bits, expected_bits, places=6)


if __name__ == "__main__":
    unittest.main()
