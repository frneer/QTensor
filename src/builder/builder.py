#!/usr/bin/env python3
from typing import Optional, Type, Dict, List, Tuple

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig
from tensorflow_model_optimization.quantization.keras import quantize_scope, quantize_apply, quantize_annotate_layer

from quantizers.uniform_quantizer import UniformQuantizer

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

class QBuilder:
    def __init__(self, model: Type[tf.keras.models.Sequential]):
        self.model = model

    def add(self, layer: tf.keras.layers.Layer, quantizer: Optional[Quantizer] = None, weight_quantizer: Optional[Dict[str, Quantizer]] = None, activation_quantizer: Optional[Quantizer] = None):
        """Add a layer to the model. If any quantizer is passed, it will be used for all the quantizers.
        If a specific quantizer is passed, it will be used for that specific quantizer (overriding the general one).
        If no quantizer is passed, no quantization will be applied to that layer.

        :param layer(tf.keras.layers.Layer): Layer to add to the model
        :param quantizer(Optional[Quantizer]): General quantizer to use for all the quantizers
        :param weight_quantizer([Quantizer]): Quantizer to use for the weights
        :param bias_quantizer([Quantizer]): Quantizer to use for the bias
        :param activation_quantizer([Quantizer]): Quantizer to use for the activation
        """
        weight_quantizer = weight_quantizer or quantizer
        activation_quantizer = activation_quantizer or quantizer

        self.model.add(quantize_annotate_layer(layer, GenerateConfig(weight_quantizer, activation_quantizer)))

        return self

    def build(self) -> tf.keras.models.Model:
        # TODO(Fran): get below dict objects from the quantizers passed in add method (Constant comes from UniformQuantizer)
        # So maybe if there are custom objects to register each class should have a method to return them
        custom_objects = {}
        custom_objects["GenerateConfig"] = GenerateConfig
        custom_objects["UniformQuantizer"] = UniformQuantizer
        custom_objects["Constant"] = tf.keras.initializers.Constant


        print(*self.model.layers, sep="\n")
        with quantize_scope(custom_objects):
            return quantize_apply(self.model)
