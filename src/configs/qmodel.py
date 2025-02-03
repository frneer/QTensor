"""QModel.

This module contains the utilities to quantize a Keras model.
"""

from generate_config import AttributeQuantizerDict, GenerateConfig
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, clone_model
from tensorflow_model_optimization.quantization.keras import (
    quantize_annotate_layer,
    quantize_apply,
    quantize_scope,
)

from quantizers.uniform_quantizer import UniformQuantizer

# {layer_name: AttributeQuantizerDict}
LayerQuantizerDict = dict[str, AttributeQuantizerDict]


def clone_layer(layer: Layer):
    clone = layer.__class__.from_config(layer.get_config())

    # If the layer has an input shape, build it so we can copy the weights
    if hasattr(layer, "input_shape"):
        clone.build(layer.input_shape)

    clone.set_weights(layer.get_weights())

    return clone


def quantize_layer(layer: Layer, layer_quantizers: AttributeQuantizerDict):
    return quantize_annotate_layer(
        clone_layer(layer), GenerateConfig(**layer_quantizers)
    )


def quantize_model(model: Model, quantizers: LayerQuantizerDict):
    def clone_function(layer: Layer):
        if layer.name in quantizers:
            return quantize_layer(layer, quantizers[layer.name])
        else:
            return clone_layer(layer)

    return clone_model(
        model, clone_function=lambda layer: clone_function(layer)
    )


def apply_quantization(model: Model, quantizers: LayerQuantizerDict):
    # TODO(Fran): get below dict objects from the quantizers passed in add method (Constant comes from UniformQuantizer)
    # So maybe if there are custom objects to register each class should have a method to return them
    custom_objects = {}
    custom_objects["GenerateConfig"] = GenerateConfig
    custom_objects["UniformQuantizer"] = UniformQuantizer
    custom_objects["Constant"] = Constant

    with quantize_scope(custom_objects):
        return quantize_apply(quantize_model(model, quantizers))
