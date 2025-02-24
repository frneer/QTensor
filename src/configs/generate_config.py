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
        weights: AttributeQuantizerDict = {},
        activations: AttributeQuantizerDict = {},
    ):
        self.weights = flatten_nested_dict(weights)
        self.activations = flatten_nested_dict(activations)

    def get_weights_and_quantizers(
        self, layer: Layer
    ) -> list[Tuple[Tensor, Quantizer]]:
        weights_and_quantizers = []
        for weight_attr, quantizer in self.weights.items():
            weights_and_quantizers.append(
                (get_nested_attribute(layer, weight_attr), quantizer)
            )
        return weights_and_quantizers

    def set_quantize_weights(self, layer: Layer, quantize_weights: list[Tensor]):
        for attribute, quantized_weight in zip(self.weights.keys(), quantize_weights):
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
        return {"weights": self.weights, "activations": self.activations}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
