#!/usr/bin/env python3
# Integration of saving models and metrics

import tempfile
import unittest

import numpy as np
import tensorflow as tf

from configs.qmodel import apply_quantization
from configs.serialization.serialization import load_qmodel, save_qmodel
from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer
from utils.metrics import compute_space_complexity_model


class TestIntegrationMetricsSerialization(unittest.TestCase):
    def test_save_and_load_model(self):
        """Test saving and loading a model with metrics."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, input_shape=(5,), name="dense_1"),
                tf.keras.layers.Dense(5, name="dense_2"),
            ]
        )

        qconfig = {
            "dense_1": {
                "weights": {
                    "kernel": UniformQuantizer(bits=4, signed=True),
                    "bias": UniformQuantizer(bits=4, signed=True),
                },
            },
            "dense_2": {
                "weights": {
                    "kernel": UniformQuantizer(bits=4, signed=True),
                    "bias": UniformQuantizer(bits=4, signed=True),
                },
            },
        }
        qmodel = apply_quantization(model, qconfig)
        qmodel.build((None, 5))

        tmpdir = tempfile.mkdtemp()
        save_qmodel(qmodel, tmpdir)
        loaded_model = load_qmodel(tmpdir)
        # make an inference to ensure the model is loaded correctly
        loaded_model(tf.random.normal((1, 5)))

        original_weights = {w.name: w.numpy() for w in qmodel.weights}
        loaded_weights = {w.name: w.numpy() for w in loaded_model.weights}

        # First, check that the set of weight names is identical
        self.assertEqual(
            set(original_weights.keys()),
            set(loaded_weights.keys()),
            "Models have different sets of weight names.",
        )

        # Now, compare each weight tensor by name
        for name, orig_w in original_weights.items():
            loaded_w = loaded_weights[name]
            # print(f"Comparing weight tensor: {name}")
            # print(f"Weights: {orig_w}")
            # print(f"Loaded: {loaded_w}")
            np.testing.assert_allclose(
                orig_w,
                loaded_w,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Weight tensor '{name}' differs.",
            )

        self.assertEqual(
            compute_space_complexity_model(qmodel),
            compute_space_complexity_model(loaded_model),
        )

    def test_save_and_load_model_flex(self):
        """Test saving and loading a model with metrics."""
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, input_shape=(5,), name="dense_1"),
                tf.keras.layers.Dense(5, name="dense_2"),
            ]
        )

        qconfig = {
            "dense_1": {
                "weights": {
                    "kernel": FlexQuantizer(bits=4, n_levels=5, signed=True),
                    "bias": FlexQuantizer(bits=4, n_levels=4, signed=True),
                },
            },
            "dense_2": {
                "weights": {
                    "kernel": FlexQuantizer(bits=4, n_levels=5, signed=True),
                    "bias": FlexQuantizer(bits=4, n_levels=4, signed=True),
                },
            },
        }
        qmodel = apply_quantization(model, qconfig)
        qmodel.build((None, 5))

        tmpdir = tempfile.mkdtemp()
        save_qmodel(qmodel, tmpdir)
        loaded_model = load_qmodel(tmpdir)
        # make an inference to ensure the model is loaded correctly
        loaded_model(tf.random.normal((1, 5)))

        original_weights = {w.name: w.numpy() for w in qmodel.weights}
        loaded_weights = {w.name: w.numpy() for w in loaded_model.weights}

        # First, check that the set of weight names is identical
        self.assertEqual(
            set(original_weights.keys()),
            set(loaded_weights.keys()),
            "Models have different sets of weight names.",
        )

        # Now, compare each weight tensor by name
        for name, orig_w in original_weights.items():
            loaded_w = loaded_weights[name]
            # print(f"Comparing weight tensor: {name}")
            # print(f"Weights: {orig_w}")
            # print(f"Loaded: {loaded_w}")
            np.testing.assert_allclose(
                orig_w,
                loaded_w,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"Weight tensor '{name}' differs.",
            )

        self.assertEqual(
            compute_space_complexity_model(qmodel),
            compute_space_complexity_model(loaded_model),
        )


if __name__ == "__main__":
    unittest.main()
