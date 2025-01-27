#!/usr/bin/env python3

import unittest

from builder import GenerateConfig, QBuilder
from tensorflow import Variable
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer

# class TestGenerateConfig(unittest.TestCase):

#     def setUp(self):
#         self.layer = unittest.mock.Mock(spec=Layer)
#         self.layer_with_activation = unittest.mock.Mock(spec=Layer)
#         self.layer_with_activation.activation = Variable(0)
#         self.weight_quantizer = unittest.mock.Mock(spec=Quantizer)
#         self.activation_quantizer = unittest.mock.Mock(spec=Quantizer)

#     def test_generate_config(self):
#         GenerateConfig(
#             weight_quantizer=self.weight_quantizer,
#             activation_quantizer=self.activation_quantizer
#         )

#     def test_weight_only(self):
#         config = GenerateConfig(
#             weight_quantizer=self.weight_quantizer,
#         )

#         self.assertListEqual(config.get_weights_and_quantizers(self.layer), [(self.layer.weights, self.weight_quantizer)])
#         self.assertListEqual(config.get_activations_and_quantizers(self.layer), [])

#     def test_activation_only(self):
#         config = GenerateConfig(
#             activation_quantizer=self.activation_quantizer,
#         )

#         self.assertListEqual(config.get_weights_and_quantizers(self.layer), [])
#         self.assertListEqual(config.get_activations_and_quantizers(self.layer_with_activation), [(self.layer_with_activation.activation, self.activation_quantizer)])

#     def test_weight_and_activation(self):
#         config = GenerateConfig(
#             weight_quantizer=self.weight_quantizer,
#             activation_quantizer=self.activation_quantizer
#         )

#         self.assertListEqual(config.get_weights_and_quantizers(self.layer), [(self.layer.weights, self.weight_quantizer)])
#         self.assertListEqual(config.get_activations_and_quantizers(self.layer_with_activation), [(self.layer_with_activation.activation, self.activation_quantizer)])

class TestBuilder(unittest.TestCase):
    def setUp(self):
        self.model = Sequential()
        self.layer_1 = unittest.mock.MagicMock(spec=Layer).build(input_shape=(1, 1))
        self.layer_2 = unittest.mock.MagicMock(spec=Layer).build(input_shape=(1, 1))
        self.builder = QBuilder(self.model)

    # def test_builder_single_non_quantized_layer(self):
    #     self.builder.add(self.layer_1)
    #     self.model.add.assert_called_once()

    #     self.builder.build()

    def test_builder_single_quantized_layer(self):
        weight_quantizer = unittest.mock.Mock(spec=Quantizer)
        activation_quantizer = unittest.mock.Mock(spec=Quantizer)
        self.builder.add(self.layer_1, weight_quantizer=weight_quantizer, activation_quantizer=activation_quantizer)
        # self.model.add.assert_called_once()

        self.builder.build()


# Only weights break... list index out of range?
# #
# # If none is there, no quantization will be applied... maybe empty config isntead of None?
# # with quantize_scope(custom_objects):
# #     return quantize_apply(self.model)
# return self.model



if __name__ == "__main__":
    unittest.main()
