#!/usr/bin/python3

import unittest

from tensorflow_model_optimization.python.core.quantization.keras.quantizers import (
    Quantizer,
)
from tensorflow_model_optimization.python.core.quantization.keras.utils import (
    serialize_keras_object,
)

from configs.generate_config import (
    GenerateConfig,
    flatten_nested_dict,
    get_nested_attribute,
    set_nested_attribute,
)


class GenerateConfigTest(unittest.TestCase):
    """Test class for the GenerateConfig class."""

    def setUp(self):
        self.quantizer = unittest.mock.Mock(spec=Quantizer)
        self.quantizer.get_config.return_value = {
            "bits": 8,
            "signed": True,
            "name_suffix": "_asdasd",
            "initializer": "Constant",
            "regularizer": None,
        }
        self.layer = unittest.mock.Mock()

    def test_can_instantiate_generate_config(self):
        """Test that verifies that the GenerateConfig can be instantiated."""
        GenerateConfig()

    def test_can_use_weight_quantizer(self):
        """Test that verifies that the GenerateConfig can be instantiated with
        weights."""
        config = GenerateConfig(weights={"kernel": self.quantizer})
        self.assertListEqual(
            config.get_weights_and_quantizers(self.layer),
            [(self.layer.kernel, self.quantizer)],
        )

    def test_can_use_weight_attribute_quantizer_dict_one_element(self):
        """Test that verifies that the GenerateConfig can be instantiated with
        weights."""
        config = GenerateConfig(weights={"kernel": self.quantizer})
        self.assertListEqual(
            config.get_weights_and_quantizers(self.layer),
            [(self.layer.kernel, self.quantizer)],
        )

    def test_can_use_weight_attribute_quantizer_dict(self):
        """Test that verifies that the GenerateConfig can be instantiated with
        weights."""
        config = GenerateConfig(
            weights={"kernel": self.quantizer, "bias": self.quantizer}
        )
        self.assertListEqual(
            config.get_weights_and_quantizers(self.layer),
            [
                (self.layer.kernel, self.quantizer),
                (self.layer.bias, self.quantizer),
            ],
        )

    def test_can_use_activation_quantizer(self):
        """Test that verifies that the GenerateConfig can be instantiated with
        activations."""
        config = GenerateConfig(activations={"activation": self.quantizer})
        self.assertListEqual(
            config.get_activations_and_quantizers(self.layer),
            [(self.layer.activation, self.quantizer)],
        )

    def test_can_use_activation_attribute_quantizer_dict_one_element(self):
        """Test that verifies that the GenerateConfig can be instantiated with
        activations."""
        config = GenerateConfig(activations={"activation": self.quantizer})
        self.assertListEqual(
            config.get_activations_and_quantizers(self.layer),
            [(self.layer.activation, self.quantizer)],
        )

    def test_can_use_activation_attribute_quantizer_dict(self):
        """Test that verifies that the GenerateConfig can be instantiated with
        activations."""
        config = GenerateConfig(
            activations={"activation": self.quantizer, "other": self.quantizer}
        )
        self.assertListEqual(
            config.get_activations_and_quantizers(self.layer),
            [
                (self.layer.activation, self.quantizer),
                (self.layer.other, self.quantizer),
            ],
        )

    def test_can_set_quantize_activations(self):
        """Test that verifies that the GenerateConfig can set quantize
        activations."""
        config = GenerateConfig(activations={"activation": self.quantizer})
        config.set_quantize_activations(self.layer, [self.layer.activation])
        self.assertEqual(self.layer.activation, self.layer.activation)

    def test_can_get_config(self):
        """Test that verifies that the GenerateConfig can get the
        configuration."""
        config = GenerateConfig(
            weights={"kernel": self.quantizer},
            activations={"activation": self.quantizer},
        )

        expected_config = {
            "weights": {"kernel": serialize_keras_object(self.quantizer)},
            "activations": {
                "activation": serialize_keras_object(self.quantizer)
            },
        }

        self.assertDictEqual(config.get_config(), expected_config)

    def test_can_get_config_with_dict(self):
        """Test that verifies that the GenerateConfig can get the
        configuration."""
        config = GenerateConfig(
            weights={"kernel": self.quantizer, "bias": self.quantizer},
            activations={
                "activation": self.quantizer,
                "other": self.quantizer,
            },
        )

        expected_config = {
            "weights": {
                "kernel": serialize_keras_object(self.quantizer),
                "bias": serialize_keras_object(self.quantizer),
            },
            "activations": {
                "activation": serialize_keras_object(self.quantizer),
                "other": serialize_keras_object(self.quantizer),
            },
        }

        self.assertDictEqual(config.get_config(), expected_config)

    def test_nested_weight_config(self):
        """Test that verifies that the GenerateConfig can be instantiated with
        nested weights."""
        config = GenerateConfig(
            weights={
                "cell": {
                    "kernel": self.quantizer,
                    "recurrent_kernel": self.quantizer,
                    "bias": self.quantizer,
                },
                "kernel": self.quantizer,
            }
        )
        self.assertListEqual(
            config.get_weights_and_quantizers(self.layer),
            [
                (self.layer.cell.kernel, self.quantizer),
                (self.layer.cell.recurrent_kernel, self.quantizer),
                (self.layer.cell.bias, self.quantizer),
                (self.layer.kernel, self.quantizer),
            ],
        )


class FlattenNestedDictTest(unittest.TestCase):
    def setUp(self):
        self.quantizer = unittest.mock.Mock(spec=Quantizer)

    def test_parse_nested_dict(self):
        """Test that verifies that can parse a nested dictionary."""
        test_dict = {
            "cell": {
                "kernel": self.quantizer,
                "recurrent_kernel": self.quantizer,
                "bias": self.quantizer,
            },
            "kernel": self.quantizer,
            "bias": self.quantizer,
        }

        self.assertDictEqual(
            flatten_nested_dict(test_dict),
            {
                "cell.kernel": self.quantizer,
                "cell.recurrent_kernel": self.quantizer,
                "cell.bias": self.quantizer,
                "kernel": self.quantizer,
                "bias": self.quantizer,
            },
        )

    def test_parse_nested_dict_is_idempotent(self):
        """Test that verifies that parsing a non-nested dictionary returns the
        same dictionary (Idempotent)."""
        test_dict = {
            "cell": {
                "kernel": self.quantizer,
                "recurrent_kernel": self.quantizer,
                "bias": self.quantizer,
            },
            "kernel": self.quantizer,
            "bias": self.quantizer,
        }

        self.assertDictEqual(
            flatten_nested_dict(flatten_nested_dict(test_dict)),
            flatten_nested_dict(test_dict),
        )


class GetNestedAttributeTest(unittest.TestCase):
    def setUp(self):
        self.quantizer = unittest.mock.Mock(spec=Quantizer)
        self.layer = unittest.mock.Mock()

    def test_get_nested_attribute(self):
        """Test that verifies that can get a nested attribute."""
        test_dict = {
            "cell": {
                "kernel": self.quantizer,
                "recurrent_kernel": self.quantizer,
                "bias": self.quantizer,
            },
            "kernel": self.quantizer,
            "bias": self.quantizer,
        }

        self.assertEqual(
            get_nested_attribute(self.layer, "cell.kernel"),
            self.layer.cell.kernel,
        )
        self.assertEqual(
            get_nested_attribute(self.layer, "cell.recurrent_kernel"),
            self.layer.cell.recurrent_kernel,
        )
        self.assertEqual(
            get_nested_attribute(self.layer, "cell.bias"), self.layer.cell.bias
        )
        self.assertEqual(
            get_nested_attribute(self.layer, "kernel"), self.layer.kernel
        )
        self.assertEqual(
            get_nested_attribute(self.layer, "bias"), self.layer.bias
        )


class SetNestedAttributeTest(unittest.TestCase):
    def setUp(self):
        self.quantizer = unittest.mock.Mock(spec=Quantizer)
        self.layer = unittest.mock.Mock()

    def test_set_nested_attribute(self):
        """Test that verifies that can set a nested attribute."""
        test_dict = {
            "cell": {
                "kernel": self.quantizer,
                "recurrent_kernel": self.quantizer,
                "bias": self.quantizer,
            },
            "kernel": self.quantizer,
            "bias": self.quantizer,
        }

        set_nested_attribute(self.layer, "cell.kernel", self.quantizer)
        self.assertEqual(self.layer.cell.kernel, self.quantizer)

        set_nested_attribute(
            self.layer, "cell.recurrent_kernel", self.quantizer
        )
        self.assertEqual(self.layer.cell.recurrent_kernel, self.quantizer)

        set_nested_attribute(self.layer, "cell.bias", self.quantizer)
        self.assertEqual(self.layer.cell.bias, self.quantizer)

        set_nested_attribute(self.layer, "kernel", self.quantizer)
        self.assertEqual(self.layer.kernel, self.quantizer)

        set_nested_attribute(self.layer, "bias", self.quantizer)
        self.assertEqual(self.layer.bias, self.quantizer)


if __name__ == "__main__":
    unittest.main()
