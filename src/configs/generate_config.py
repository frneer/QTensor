"""GenerateConfig This module contains an automatic generation of a
quantization configuration for a Keras model."""

from typing import Tuple

from tensorflow import Tensor
from tensorflow.keras.layers import Layer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import (
    QuantizeConfig,
)
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import (
    Quantizer,
)
from tensorflow_model_optimization.python.core.quantization.keras.utils import (
    deserialize_keras_object,
    serialize_keras_object,
)

AttributeQuantizerDict = dict[str, Quantizer | dict[str, Quantizer]]


def get_nested_attribute(obj: object, attribute: str) -> object:
    """Get a nested attribute from an object."""
    attributes = attribute.split(".")
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj


def set_nested_attribute(obj: object, attribute: str, value: object):
    """Set a nested attribute from an object."""
    attributes = attribute.split(".")
    for attr in attributes[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attributes[-1], value)


def flatten_nested_dict(nested_dict: dict, prefix: str = "") -> dict:
    """Parse a nested dictionary into a flat dictionary."""
    flat_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            flat_dict.update(flatten_nested_dict(value, f"{prefix}{key}."))
        else:
            flat_dict[f"{prefix}{key}"] = value
    return flat_dict


class GenerateConfig(QuantizeConfig):
    """Generate a quantization configuration for a Keras model."""

    def __init__(
        self,
        weights: AttributeQuantizerDict = None,
        activations: AttributeQuantizerDict = None,
    ):
        self._raw_weights = weights or {}
        self._raw_activations = activations or {}
        self.weights = flatten_nested_dict(self._raw_weights)
        self.activations = flatten_nested_dict(self._raw_activations)

    def _serialize_recursively(self, nested_obj):
        """Recursively traverses a nested dict and serializes any Keras object
        (like a Quantizer) it finds."""
        if isinstance(nested_obj, dict):
            # If it's a dict, recurse on its values
            return {
                key: self._serialize_recursively(value)
                for key, value in nested_obj.items()
            }
        elif hasattr(nested_obj, "get_config"):
            # Base case: If it's a serializable object (a Quantizer), serialize it.
            return serialize_keras_object(nested_obj)
        else:
            # It's some other primitive type, return it as is.
            return nested_obj

    def get_weights_and_quantizers(
        self, layer: Layer
    ) -> list[Tuple[Tensor, Quantizer]]:
        weights_and_quantizers = []
        for weight_attr, quantizer in self.weights.items():
            weights_and_quantizers.append(
                (get_nested_attribute(layer, weight_attr), quantizer)
            )
        return weights_and_quantizers

    def set_quantize_weights(
        self, layer: Layer, quantize_weights: list[Tensor]
    ):
        for attribute, quantized_weight in zip(
            self.weights.keys(), quantize_weights
        ):
            set_nested_attribute(layer, attribute, quantized_weight)

    def get_activations_and_quantizers(
        self, layer: Layer
    ) -> list[Tuple[Tensor, Quantizer]]:
        activations_and_quantizers = []
        for activation_attribute, quantizer in self.activations.items():
            activations_and_quantizers.append(
                (get_nested_attribute(layer, activation_attribute), quantizer)
            )
        return activations_and_quantizers

    def set_quantize_activations(
        self, layer: Layer, quantize_activations: list[Tensor]
    ):
        for attribute, quantized_activation in zip(
            self.activations.keys(), quantize_activations
        ):
            set_nested_attribute(layer, attribute, quantized_activation)

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        """Correctly serialize by recursively traversing the nested
        dictionaries."""
        return {
            "weights": self._serialize_recursively(self._raw_weights),
            "activations": self._serialize_recursively(self._raw_activations),
        }

    @classmethod
    def from_config(cls, config):
        """
        FIX: Use a recursive helper function to deserialize the nested structure
        before passing it to the constructor.
        """

        def _deserialize_recursively(nested_config):
            # Base Case: If the dict is a serialized Keras object, deserialize it.
            if (
                isinstance(nested_config, dict)
                and "class_name" in nested_config
            ):
                return deserialize_keras_object(nested_config)

            # Recursive Step: If it's a dict container, recurse on its values.
            if isinstance(nested_config, dict):
                return {
                    key: _deserialize_recursively(value)
                    for key, value in nested_config.items()
                }

            # Handle lists of items
            if isinstance(nested_config, list):
                return [
                    _deserialize_recursively(item) for item in nested_config
                ]

            # It's a primitive type, return as is.
            return nested_config

        return cls(
            weights=_deserialize_recursively(config["weights"]),
            activations=_deserialize_recursively(config["activations"]),
        )
