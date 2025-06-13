#!/usr/bin/env python3

import unittest
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import (
    QuantizeWrapperV2,
)

from configs.qmodel import apply_quantization
from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer
from utils.metrics import compute_space_complexity_model


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


def create_lenet_model(categories):
    model = models.Sequential(
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
    return model


def create_single_conv2d_model():
    model = models.Sequential(
        [
            layers.Conv2D(
                32,
                kernel_size=5,
                activation="relu",
                padding="same",
                name="conv2d",
            ),
        ]
    )
    return model


class TestLeNetQuantizedComplexity(unittest.TestCase):
    def setUp(self):
        # build a small LeNet-5 for, say, 10 classes over 28×28×1 inputs
        categories = 10
        input_shape = (None, 28, 28, 1)
        self.model_lenet = create_lenet_model(categories)
        self.model_lenet.build(input_shape)

        input_shape = (None, 28, 28, 1)
        self.model_single_conv2d = create_single_conv2d_model()
        self.model_single_conv2d.build(input_shape)

        input_shape = (None, 10)
        self.model_single_dense = models.Sequential(
            [
                layers.Dense(20, activation="relu", name="dense"),
            ]
        )
        self.model_single_dense.build(input_shape)

    def setup_model(self, model):
        self.model = model
        self.model.compile(
            loss="categorical_crossentropy", metrics=["accuracy"]
        )
        # run one dummy inference so any lazy weights are created
        input_shape = self.model.input_shape
        input_shape = (1,) + input_shape[1:]
        self.model(tf.random.normal(input_shape))
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
        # self.model.summary()

    def gen_qconfig(
        self,
        qtype,
        layer_names,
        kernel_bits,
        bias_bits,
        kernel_n_levels=None,
        bias_n_levels=None,
    ):
        qconfig = {}
        for i, layer in enumerate(layer_names):
            qconfig[layer] = dict()
            qconfig[layer]["weights"] = dict()
        if qtype == "uniform":
            # 1) define an 8‐bit uniform quantizer on every kernel
            for i, layer in enumerate(layer_names):
                qconfig[layer]["weights"]["kernel"] = UniformQuantizer(
                    bits=kernel_bits[i], signed=True
                )
                qconfig[layer]["weights"]["bias"] = UniformQuantizer(
                    bits=bias_bits[i], signed=True
                )

        elif qtype == "flexible":
            # 1) build a qconfig where every layer's kernel & bias uses a FlexQuantizer
            for i, layer in enumerate(layer_names):
                qconfig[layer]["weights"]["kernel"] = FlexQuantizer(
                    bits=kernel_bits[i],
                    n_levels=kernel_n_levels[i],
                    signed=True,
                )
                qconfig[layer]["weights"]["bias"] = FlexQuantizer(
                    bits=bias_bits[i], n_levels=bias_n_levels[i], signed=True
                )

        else:
            raise ValueError(f"Invalid qtype ({qtype})")

        # DEBUG: Print qconfig
        # for key in qconfig:
        #    print(f'{key}: {qconfig[key]}')
        return qconfig

    def random_probability_vector(self, n, epsilon=1e-8):
        vec = np.random.rand(n) + epsilon
        return vec / vec.sum()

    def equal_probability_vector(self, n):
        vec = np.ones(n)
        return vec / vec.sum()

    def increasing_probability_vector(self, n):
        vec = np.arange(1, n + 1)
        return vec / vec.sum()

    def base_uniform_quantizer_space_complexity(
        self,
        model,
        layer_names,
        kernel_bits,
        bias_bits,
        kernel_alphas,
        bias_alphas,
    ):
        """All weights quantized using uniform quantizer."""

        self.setup_model(model)

        qconfig = self.gen_qconfig(
            "uniform", layer_names, kernel_bits, bias_bits
        )

        # 4) compute expected size
        expected_bits = 0
        for kb, ks, bb, bs in zip(
            kernel_bits, self.kernel_size, bias_bits, self.bias_size
        ):
            expected_bits += kb * ks + bb * bs

        # 2) apply quantization and build
        input_shape = self.model.input_shape
        qmodel = apply_quantization(self.model, qconfig)
        qmodel.build(input_shape)
        input_shape = (1,) + input_shape[1:]
        qmodel(tf.random.normal(input_shape))

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
        self.assertEqual(computed_bits, expected_bits)

    def base_flex_quantizer_space_complexity(
        self,
        model,
        layer_names,
        kernel_bits,
        bias_bits,
        kernel_n_levels,
        bias_n_levels,
        kernel_probabilities,
        bias_probabilities,
        kernel_alphas,
        bias_alphas,
    ):
        """All weights quantized using flexible quantizer."""

        self.setup_model(model)

        qconfig = self.gen_qconfig(
            "flexible",
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_n_levels,
            bias_n_levels,
        )

        # 2) compute expected total bits
        expected_bits = 0.0
        kernels = []
        biases = []
        kvvalues = []
        bvvalues = []
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
                kvvalues,
            ),
            (
                self.bias_shape,
                self.bias_size,
                bias_bits,
                bias_n_levels,
                bias_alphas,
                bias_probabilities,
                biases,
                bvvalues,
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
            vvalues,
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
                vvalues.append(values)
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
        input_shape = self.model.input_shape
        qmodel = apply_quantization(self.model, qconfig)
        qmodel.build(input_shape)
        input_shape = (1,) + input_shape[1:]
        qmodel(tf.random.normal(input_shape))

        # 5) set alphats
        # alpha_dict = {
        #    layer_name: {"kernel": kalpha, "bias": balpha}
        #    for layer_name, kalpha, balpha in zip(
        #        qconfig, kernel_alphas, bias_alphas
        #    )
        # }
        alpha_dict = {}
        levels_dict = {}
        thresholds_dict = {}
        for layer_name, kalpha, balpha, k, b in zip(
            qconfig, kernel_alphas, bias_alphas, kvvalues, bvvalues
        ):
            klevels = k
            blevels = b
            kthresholds = [-kalpha] + list((k[1:] + k[:-1]) / 2) + [kalpha]
            bthresholds = [-balpha] + list((b[1:] + b[:-1]) / 2) + [balpha]
            alpha_dict[layer_name] = {"kernel": kalpha, "bias": balpha}
            levels_dict[layer_name] = {"kernel": klevels, "bias": blevels}
            thresholds_dict[layer_name] = {
                "kernel": kthresholds,
                "bias": bthresholds,
            }
        apply_flex_dict(qmodel, alpha_dict, levels_dict, thresholds_dict)
        qmodel(tf.random.normal(input_shape))

        # 6) compare to your implementation
        computed_bits = compute_space_complexity_model(qmodel)
        self.assertAlmostEqual(computed_bits, expected_bits, places=6)

    def test_uniform_quantizer_space_complexity_single_dense(self):
        model = self.model_single_dense
        layer_names = [
            "dense",
        ]
        kernel_bits = [
            6,
        ]
        bias_bits = [
            4,
        ]
        kernel_alphas = [1.0] * 1
        bias_alphas = [1.0] * 1
        self.base_uniform_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_alphas,
            bias_alphas,
        )

    def test_uniform_quantizer_space_complexity_single_conv2d(self):
        model = self.model_single_conv2d
        layer_names = [
            "conv2d",
        ]
        kernel_bits = [
            6,
        ]
        bias_bits = [
            4,
        ]
        kernel_alphas = [1.0] * 1
        bias_alphas = [1.0] * 1
        self.base_uniform_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_alphas,
            bias_alphas,
        )

    def test_uniform_quantizer_space_complexity_lenet(self):
        model = self.model_lenet
        layer_names = ["conv2d", "conv2d_1", "dense", "dense_1", "dense_2"]
        kernel_bits = [7, 6, 5, 4, 3]
        bias_bits = [3, 4, 5, 6, 7]
        kernel_alphas = [1.0] * 5
        bias_alphas = [1.0] * 5
        self.base_uniform_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_alphas,
            bias_alphas,
        )

    def test_flex_quantizer_space_complexity_single_dense_1(self):
        model = self.model_single_dense
        layer_names = [
            "dense",
        ]
        kernel_bits = [
            6,
        ]
        bias_bits = [
            4,
        ]
        kernel_n_levels = [2] * 1  # TEST: for levels = 2
        bias_n_levels = [2] * 1  # TEST: for levels = 2
        kernel_alphas = [1.0] * 1
        bias_alphas = [1.0] * 1
        kernel_probabilities = []
        bias_probabilities = []
        for kl, bl in zip(kernel_n_levels, bias_n_levels):
            kernel_probabilities.append(
                self.equal_probability_vector(kl)
            )  # TEST: equiprobabilities
            bias_probabilities.append(
                self.equal_probability_vector(bl)
            )  # TEST: equiprobabilities
        self.base_flex_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_n_levels,
            bias_n_levels,
            kernel_probabilities,
            bias_probabilities,
            kernel_alphas,
            bias_alphas,
        )

    def test_flex_quantizer_space_complexity_single_dense_2(self):
        model = self.model_single_dense
        layer_names = [
            "dense",
        ]
        kernel_bits = [
            6,
        ]
        bias_bits = [
            4,
        ]
        kernel_n_levels = [13] * 1
        bias_n_levels = [8] * 1
        kernel_alphas = [1.0] * 1
        bias_alphas = [1.0] * 1
        kernel_probabilities = []
        bias_probabilities = []
        for kl, bl in zip(kernel_n_levels, bias_n_levels):
            kernel_probabilities.append(
                self.equal_probability_vector(kl)
            )  # TEST: equiprobabilities
            bias_probabilities.append(
                self.equal_probability_vector(bl)
            )  # TEST: equiprobabilities
        self.base_flex_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_n_levels,
            bias_n_levels,
            kernel_probabilities,
            bias_probabilities,
            kernel_alphas,
            bias_alphas,
        )

    def test_flex_quantizer_space_complexity_single_dense_3(self):
        model = self.model_single_dense
        layer_names = [
            "dense",
        ]
        kernel_bits = [
            6,
        ]
        bias_bits = [
            4,
        ]
        kernel_n_levels = [13] * 1
        bias_n_levels = [8] * 1
        kernel_alphas = [1.0] * 1
        bias_alphas = [1.0] * 1
        kernel_probabilities = []
        bias_probabilities = []
        for kl, bl in zip(kernel_n_levels, bias_n_levels):
            kernel_probabilities.append(self.increasing_probability_vector(kl))
            bias_probabilities.append(self.increasing_probability_vector(bl))
        self.base_flex_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_n_levels,
            bias_n_levels,
            kernel_probabilities,
            bias_probabilities,
            kernel_alphas,
            bias_alphas,
        )

    def test_flex_quantizer_space_complexity_single_dense_4(self):
        model = self.model_single_dense
        layer_names = [
            "dense",
        ]
        kernel_bits = [
            6,
        ]
        bias_bits = [
            4,
        ]
        kernel_n_levels = [13] * 1
        bias_n_levels = [8] * 1
        kernel_alphas = [1.0] * 1
        bias_alphas = [1.0] * 1
        kernel_probabilities = []
        bias_probabilities = []
        for kl, bl in zip(kernel_n_levels, bias_n_levels):
            kernel_probabilities.append(self.random_probability_vector(kl))
            bias_probabilities.append(self.random_probability_vector(bl))
        self.base_flex_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_n_levels,
            bias_n_levels,
            kernel_probabilities,
            bias_probabilities,
            kernel_alphas,
            bias_alphas,
        )

    def test_flex_quantizer_space_complexity_single_conv2d_1(self):
        model = self.model_single_conv2d
        layer_names = [
            "conv2d",
        ]
        kernel_bits = [
            6,
        ]
        bias_bits = [
            4,
        ]
        kernel_n_levels = [2] * 1  # TEST: for levels = 2
        bias_n_levels = [2] * 1  # TEST: for levels = 2
        kernel_alphas = [1.0] * 1
        bias_alphas = [1.0] * 1
        kernel_probabilities = []
        bias_probabilities = []
        for kl, bl in zip(kernel_n_levels, bias_n_levels):
            kernel_probabilities.append(
                self.equal_probability_vector(kl)
            )  # TEST: equiprobabilities
            bias_probabilities.append(
                self.equal_probability_vector(bl)
            )  # TEST: equiprobabilities
        self.base_flex_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_n_levels,
            bias_n_levels,
            kernel_probabilities,
            bias_probabilities,
            kernel_alphas,
            bias_alphas,
        )

    def test_flex_quantizer_space_complexity_single_conv2d_2(self):
        model = self.model_single_conv2d
        layer_names = [
            "conv2d",
        ]
        kernel_bits = [
            6,
        ]
        bias_bits = [
            4,
        ]
        kernel_n_levels = [13] * 1
        bias_n_levels = [8] * 1
        kernel_alphas = [1.0] * 1
        bias_alphas = [1.0] * 1
        kernel_probabilities = []
        bias_probabilities = []
        for kl, bl in zip(kernel_n_levels, bias_n_levels):
            kernel_probabilities.append(
                self.equal_probability_vector(kl)
            )  # TEST: equiprobabilities
            bias_probabilities.append(
                self.equal_probability_vector(bl)
            )  # TEST: equiprobabilities
        self.base_flex_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_n_levels,
            bias_n_levels,
            kernel_probabilities,
            bias_probabilities,
            kernel_alphas,
            bias_alphas,
        )

    def test_flex_quantizer_space_complexity_single_conv2d_3(self):
        model = self.model_single_conv2d
        layer_names = [
            "conv2d",
        ]
        kernel_bits = [
            6,
        ]
        bias_bits = [
            4,
        ]
        kernel_n_levels = [13] * 1
        bias_n_levels = [8] * 1
        kernel_alphas = [1.0] * 1
        bias_alphas = [1.0] * 1
        kernel_probabilities = []
        bias_probabilities = []
        for kl, bl in zip(kernel_n_levels, bias_n_levels):
            kernel_probabilities.append(self.increasing_probability_vector(kl))
            bias_probabilities.append(self.increasing_probability_vector(bl))
        self.base_flex_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_n_levels,
            bias_n_levels,
            kernel_probabilities,
            bias_probabilities,
            kernel_alphas,
            bias_alphas,
        )

    def test_flex_quantizer_space_complexity_single_conv2d_4(self):
        model = self.model_single_conv2d
        layer_names = [
            "conv2d",
        ]
        kernel_bits = [
            6,
        ]
        bias_bits = [
            4,
        ]
        kernel_n_levels = [13] * 1
        bias_n_levels = [8] * 1
        kernel_alphas = [1.0] * 1
        bias_alphas = [1.0] * 1
        kernel_probabilities = []
        bias_probabilities = []
        for kl, bl in zip(kernel_n_levels, bias_n_levels):
            kernel_probabilities.append(self.random_probability_vector(kl))
            bias_probabilities.append(self.random_probability_vector(bl))
        self.base_flex_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_n_levels,
            bias_n_levels,
            kernel_probabilities,
            bias_probabilities,
            kernel_alphas,
            bias_alphas,
        )

    def test_flex_quantizer_space_complexity_lenet_1(self):
        model = self.model_lenet
        layer_names = ["conv2d", "conv2d_1", "dense", "dense_1", "dense_2"]
        kernel_bits = [7, 6, 5, 4, 3]
        bias_bits = [3, 4, 5, 6, 7]
        kernel_n_levels = [2] * 5  # TEST: for levels = 2
        bias_n_levels = [2] * 5  # TEST: for levels = 2
        kernel_alphas = [1.0] * 5
        bias_alphas = [1.0] * 5
        kernel_probabilities = []
        bias_probabilities = []
        for kl, bl in zip(kernel_n_levels, bias_n_levels):
            kernel_probabilities.append(
                self.equal_probability_vector(kl)
            )  # TEST: equiprobabilities
            bias_probabilities.append(
                self.equal_probability_vector(bl)
            )  # TEST: equiprobabilities
        self.base_flex_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_n_levels,
            bias_n_levels,
            kernel_probabilities,
            bias_probabilities,
            kernel_alphas,
            bias_alphas,
        )

    def test_flex_quantizer_space_complexity_lenet_2(self):
        model = self.model_lenet
        layer_names = ["conv2d", "conv2d_1", "dense", "dense_1", "dense_2"]
        kernel_bits = [7, 6, 5, 4, 3]
        bias_bits = [3, 4, 5, 6, 7]
        kernel_n_levels = [25, 12, 13, 5, 2]
        bias_n_levels = [3, 7, 15, 7, 14]
        kernel_alphas = [1.0] * 5
        bias_alphas = [1.0] * 5
        kernel_probabilities = []
        bias_probabilities = []
        for kl, bl in zip(kernel_n_levels, bias_n_levels):
            kernel_probabilities.append(
                self.equal_probability_vector(kl)
            )  # TEST: equiprobabilities
            bias_probabilities.append(
                self.equal_probability_vector(bl)
            )  # TEST: equiprobabilities
        self.base_flex_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_n_levels,
            bias_n_levels,
            kernel_probabilities,
            bias_probabilities,
            kernel_alphas,
            bias_alphas,
        )

    def test_flex_quantizer_space_complexity_lenet_3(self):
        model = self.model_lenet
        layer_names = ["conv2d", "conv2d_1", "dense", "dense_1", "dense_2"]
        kernel_bits = [7, 6, 5, 4, 3]
        bias_bits = [3, 4, 5, 6, 7]
        kernel_n_levels = [25, 12, 13, 5, 2]
        bias_n_levels = [3, 7, 15, 7, 14]
        kernel_alphas = [1.0] * 5
        bias_alphas = [1.0] * 5
        kernel_probabilities = []
        bias_probabilities = []
        for kl, bl in zip(kernel_n_levels, bias_n_levels):
            kernel_probabilities.append(
                self.increasing_probability_vector(kl)
            )  # TEST: increasing probabilities
            bias_probabilities.append(
                self.increasing_probability_vector(bl)
            )  # TEST: increasing probabilities
        self.base_flex_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_n_levels,
            bias_n_levels,
            kernel_probabilities,
            bias_probabilities,
            kernel_alphas,
            bias_alphas,
        )

    def test_flex_quantizer_space_complexity_lenet_4(self):
        model = self.model_lenet
        layer_names = ["conv2d", "conv2d_1", "dense", "dense_1", "dense_2"]
        kernel_bits = [7, 6, 5, 4, 3]
        bias_bits = [3, 4, 5, 6, 7]
        kernel_n_levels = [25, 12, 13, 5, 2]
        bias_n_levels = [3, 7, 15, 7, 14]
        kernel_alphas = [1.0] * 5
        bias_alphas = [1.0] * 5
        kernel_probabilities = []
        bias_probabilities = []
        for kl, bl in zip(kernel_n_levels, bias_n_levels):
            kernel_probabilities.append(self.random_probability_vector(kl))
            bias_probabilities.append(self.random_probability_vector(bl))
        self.base_flex_quantizer_space_complexity(
            model,
            layer_names,
            kernel_bits,
            bias_bits,
            kernel_n_levels,
            bias_n_levels,
            kernel_probabilities,
            bias_probabilities,
            kernel_alphas,
            bias_alphas,
        )


if __name__ == "__main__":
    unittest.main()
