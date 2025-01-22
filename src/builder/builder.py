#!/usr/bin/env python3
from typing import Optional, Type

import tensorflow as tf

from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig
from tensorflow_model_optimization.quantization.keras import quantize_scope, quantize_apply, quantize_annotate_layer

from quantizers.uniform_quantizer import UniformQuantizer

class GenerateConfig(QuantizeConfig):
    def __init__(self, weight_quantizer: Optional[Quantizer], bias_quantizer: Optional[Quantizer] = None, activation_quantizer: Optional[Quantizer] = None):
        self.weight_quantizer = weight_quantizer
        self.bias_quantizer = bias_quantizer
        self.activation_quantizer = activation_quantizer

    def get_weights_and_quantizers(self, layer):
        weights_and_quantizers = []
        if self.weight_quantizer is not None:
            weights_and_quantizers.append((layer.kernel, self.weight_quantizer))
        if self.bias_quantizer is not None:
            weights_and_quantizers.append((layer.bias, self.bias_quantizer))
        return weights_and_quantizers

    def set_quantize_weights(self, layer, quantize_weights):
        if len(quantize_weights) == 1:
            layer.kernel = quantize_weights[0]
        if len(quantize_weights) == 2:
            layer.kernel = quantize_weights[0]
            layer.bias = quantize_weights[1]

        raise(ValueError("Unsupported number of quantized weights"))

    def get_activations_and_quantizers(self, layer):
        activations_and_quantizers = []
        if self.activation_quantizer is not None:
            activations_and_quantizers.append((layer.activation, self.activation_quantizer))
        return activations_and_quantizers

    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {
            "weight_quantizer": self.weight_quantizer,
            "bias_quantizer": self.bias_quantizer,
            "activation_quantizer": self.activation_quantizer,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class QBuilder:
    def __init__(self, model: Type[tf.keras.models.Model]):
        self.model = model()

    def add(self, layer: tf.keras.layers.Layer, quantizer: Optional[Quantizer] = None, weight_quantizer: Optional[Quantizer] = None, bias_quantizer: Optional[Quantizer] = None, activation_quantizer: Optional[Quantizer] = None):
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
        bias_quantizer = bias_quantizer or quantizer
        activation_quantizer = activation_quantizer or quantizer

        if any([weight_quantizer, bias_quantizer, activation_quantizer]):
            layer = quantize_annotate_layer(layer, GenerateConfig(weight_quantizer, bias_quantizer, activation_quantizer))
        self.model.add(layer)
        return self

    def build(self) -> tf.keras.models.Model:
        # TODO(Fran): get below dict objects from the quantizers passed in add method (Constant comes from UniformQuantizer)
        # So maybe if there are custom objects to register each class should have a method to return them
        custom_objects = {}
        custom_objects["GenerateConfig"] = GenerateConfig
        custom_objects["UniformQuantizer"] = UniformQuantizer
        custom_objects["Constant"] = tf.keras.initializers.Constant

        with quantize_scope(custom_objects):
            return quantize_apply(self.model)
