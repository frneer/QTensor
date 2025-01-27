#!/usr/bin/env python3
from typing import Optional, Type, Dict, List, Tuple

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig
from tensorflow_model_optimization.quantization.keras import quantize_scope, quantize_apply, quantize_annotate_layer

from quantizers.uniform_quantizer import UniformQuantizer

from tensorflow.keras import layers, models
import json

from typing import Dict

# Define an Inception Module
def inception_module(inputs, filter_1x1, filter_3x3_reduce, filter_3x3, filter_5x5_reduce, filter_5x5, filter_pool_proj):
    # 1x1 Convolution
    conv1x1 = layers.Conv2D(filter_1x1, kernel_size=(1, 1), padding='same', activation='relu')(inputs)

    # 3x3 Convolution
    conv3x3_reduce = layers.Conv2D(filter_3x3_reduce, kernel_size=(1, 1), padding='same', activation='relu')(inputs)
    conv3x3 = layers.Conv2D(filter_3x3, kernel_size=(3, 3), padding='same', activation='relu')(conv3x3_reduce)

    # 5x5 Convolution
    conv5x5_reduce = layers.Conv2D(filter_5x5_reduce, kernel_size=(1, 1), padding='same', activation='relu')(inputs)
    conv5x5 = layers.Conv2D(filter_5x5, kernel_size=(5, 5), padding='same', activation='relu')(conv5x5_reduce)

    # Max Pooling and Projection
    maxpool = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    maxpool_proj = layers.Conv2D(filter_pool_proj, kernel_size=(1, 1), padding='same', activation='relu')(maxpool)

    # Concatenate all the filters
    return layers.concatenate([conv1x1, conv3x3, conv5x5, maxpool_proj], axis=-1)

if __name__ == "__main__":
    input_shape = (224, 224, 3)
    inputs = tf.keras.Input(shape=input_shape)
    outputs = inception_module(inputs, 64, 96, 128, 16, 32, 32)
    model = models.Model(inputs, outputs)
    # print(model.layers)
    # json_model = json.loads(model.to_json())["config"]
    # print(json_model.keys())
    # print(json_model["layers"][5])

    config = model.get_config()
    print([layer["name"] for layer in config["layers"]])
    print(config["input_layers"])
    print(config["output_layers"])
    print(config.keys())

    # What I want

    # QuantizeModel(model, quantizers={"name": {"weight": UniformQuantizer, "activation": UniformQuantizer}}).build()


class GenerateConfig(QuantizeConfig):
    def __init__(self, weight_quantizer: Optional[Quantizer] = None, activation_quantizer: Optional[Quantizer] = None):
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

    def get_weights_and_quantizers(self, layer: tf.keras.layers.Layer) -> List[Tuple[tf.Tensor, Quantizer]]:
        if self.weight_quantizer is not None:
            return [(layer.weights, self.weight_quantizer)]
        return []

    def set_quantize_weights(self, layer: tf.keras.layers.Layer, quantize_weights):
        layer.weights = quantize_weights[0]

    def get_activations_and_quantizers(self, layer: tf.keras.layers.Layer) -> List[Tuple[tf.Tensor, Quantizer]]:
        if self.activation_quantizer is not None and hasattr(layer, "activation"):
            return [(layer.activation, self.activation_quantizer)]
        return []

    def set_quantize_activations(self, layer: tf.keras.layers.Layer, quantize_activations):
        if hasattr(layer, "activation"):
            layer.activation = quantize_activations[0]

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {
            "weight_quantizer": self.weight_quantizer,
            "activation_quantizer": self.activation_quantizer,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


QConfig = Dict[str, Quantizer]  # Dict of quantizers {weights: Quantizer, activation: Quantizer}
QMap = Dict[str, QConfig] # Dict of layers {layer_name: QConfig}

class QuantizeModel():
    def __init__(self, model: tf.keras.models.model , quantizers: QMap):
        self.model = model
        self.quantizers = quantizers

    def _quantize_layer(self, layer: tf.keras.layers.Layer, layer_quantizers: QConfig):
        layer = self._clone_layer(layer)
        return quantize_annotate_layer(layer, GenerateConfig(**layer_quantizers))

    def _clone_layer(self, layer: tf.keras.layers.Layer):
        return layer.__class__.from_config(layer.get_config())

    def _quantize_model(self, model: tf.keras.models.Model, quantizers: QMap):
        new_model = model.__class__(model.inputs, model.outputs)
        for layer in model.layers:
            if layer.name in self.quantizers:
                new_model.add(self._quantize_layer(layer, quantizers[layer.name]))
            else:
                new_model.add(self._clone_layer(layer))
        return new_model

    def build(self, *args, **kwargs):
        # TODO(Fran): get below dict objects from the quantizers passed in add method (Constant comes from UniformQuantizer)
        # So maybe if there are custom objects to register each class should have a method to return them
        custom_objects = {}
        custom_objects["GenerateConfig"] = GenerateConfig
        custom_objects["UniformQuantizer"] = UniformQuantizer
        custom_objects["Constant"] = tf.keras.initializers.Constant

        model = self._quantize_model(self.model)
        with quantize_scope(custom_objects):
            return quantize_apply(model)


    # def build(self) -> tf.keras.models.Model:
    #     # TODO(Fran): get below dict objects from the quantizers passed in add method (Constant comes from UniformQuantizer)
    #     # So maybe if there are custom objects to register each class should have a method to return them
    #     custom_objects = {}
    #     custom_objects["GenerateConfig"] = GenerateConfig
    #     custom_objects["UniformQuantizer"] = UniformQuantizer
    #     custom_objects["Constant"] = tf.keras.initializers.Constant


    #     print(*self.model.layers, sep="\n")
    #     with quantize_scope(custom_objects):
    #         return quantize_apply(self.model)


