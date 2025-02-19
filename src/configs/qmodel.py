"""QModel.

This module contains the utilities to quantize a Keras model.
"""

from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model, clone_model
from tensorflow_model_optimization.quantization.keras import (
    quantize_annotate_layer,
    quantize_apply,
    quantize_scope,
)

from configs.generate_config import AttributeQuantizerDict, GenerateConfig
from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer

# {layer_name: AttributeQuantizerDict}
LayerQuantizerDict = dict[str, AttributeQuantizerDict]


def clone_layer(layer: Layer):
    """Clone a Keras layer.

    It also copies the weights from the original layer to the clone.
    """
    clone = layer.__class__.from_config(layer.get_config())

    # If the layer has an input shape, build it so we can copy the weights
    if hasattr(layer, "input_shape"):
        clone.build(layer.input_shape)

    clone.set_weights(layer.get_weights())

    return clone


def quantize_layer(layer: Layer, layer_quantizers: AttributeQuantizerDict):
    """Quantize a Keras layer. Annotates a cloned layer with the quantizers
    specified in the layer_quantizers dict.

    param(layer): Keras layer to quantize.
    param(layer_quantizers): Dictionary with the quantizers to apply to the layer.

    Example:

    ```python
    quantize_layer(
        Dense(128, activation="relu", name="hidden"),
        {
            "weights": {"kernel": UniformQuantizer(8), "bias": UniformQuantizer(8)},
            "activations": {"activation": UniformQuantizer(8)},
        },
    )
    ```
    """
    return quantize_annotate_layer(
        clone_layer(layer), GenerateConfig(**layer_quantizers)
    )


def quantize_model(model: Model, quantizers: LayerQuantizerDict):
    """Quantize a Keras model.

    param(model): Keras model to quantize. param(quantizers): Dictionary with
    the quantizers to apply to the model.
    """

    def clone_function(layer: Layer):
        if layer.name in quantizers:
            return quantize_layer(layer, quantizers[layer.name])
        else:
            return clone_layer(layer)

    return clone_model(
        model, clone_function=lambda layer: clone_function(layer)
    )


def apply_quantization(model: Model, quantizers: LayerQuantizerDict):
    """Apply quantization to a Keras model.

    param(model): Keras model to quantize.
    param(quantizers): Dictionary with the quantizers to apply to the model.

    Example:
    ```python
    layer_2 = Dense(128, activation="relu", name="hidden")
    apply_quantization(
        Sequential([layer_1, layer_2, layer_3]),
        {
            "hidden": {
                "weights": {"kernel": UniformQuantizer(8), "bias": UniformQuantizer(8)},
                "activations": {"activation": UniformQuantizer(8)},
            }
        },
    )
    ```
    """

    model_layers = [layer.name for layer in model.layers]
    for layer_name in quantizers.keys():
        if layer_name not in model_layers:
            raise ValueError(
                f"Layer {layer_name} not found in model {model.name}."
            )
    # TODO(Fran): get below dict objects from the quantizers passed in add method (Constant comes from UniformQuantizer)
    # So maybe if there are custom objects to register each class should have a method to return them
    custom_objects = {}
    custom_objects["GenerateConfig"] = GenerateConfig
    custom_objects["UniformQuantizer"] = UniformQuantizer
    custom_objects["FlexQuantizer"] = FlexQuantizer
    custom_objects["Constant"] = Constant

    with quantize_scope(custom_objects):
        return quantize_apply(quantize_model(model, quantizers))
