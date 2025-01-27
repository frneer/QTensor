# from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import (
#     QuantizeConfig,
# )
# from tensorflow_model_optimization.python.core.quantization.keras.quantizers import (
#     Quantizer,
# )

# from quantizers.uniform_quantizer import UniformQuantizer
# import tensorflow as tf

# # from configs import AnyConfig

# from typing import List, Tuple

# class UniformQuantizeConfigPapota(QuantizeConfig):
#     def __init__(self, weight_bits: int = 8, bias_bits: int = 8, activation_bits: int = 8, alpha: float = 1.0, activation_signed: bool = False):
#         self.weight_quantizer = UniformQuantizer(
#             bits=weight_bits,
#             initializer=tf.keras.initializers.Constant(alpha),
#             signed=True,
#             name_suffix="_kernel",
#             regularizer=tf.keras.regularizers.l2(0.01),
#         )

#         self.bias_quantizer = UniformQuantizer(
#             bits=bias_bits,
#             initializer=tf.keras.initializers.Constant(alpha),
#             signed=True,
#             name_suffix="_bias",
#             regularizer=tf.keras.regularizers.l2(0.01),
#         )

#         self.activation_quantizer = UniformQuantizer(
#             bits=activation_bits,
#             initializer=tf.keras.initializers.Constant(alpha),
#             signed=activation_signed,
#             name_suffix="_activation",
#             regularizer=tf.keras.regularizers.l2(0.01),
#         )

#     # This defines how to quantize weights
#     def get_weights_and_quantizers(self, layer) -> List[Tuple[tf.Variable, Quantizer]]:
#         return [
#             (
#                 layer.kernel,
#                 self.weight_quantizer,
#             ),
#             (
#                 layer.bias,
#                 self.bias_quantizer,
#             ),
#         ]

#     def set_quantize_weights(self, layer, quantize_weights):
#         layer.kernel = quantize_weights[0]
#         layer.bias = quantize_weights[1]

#     # This defines how to quantize activations
#     def get_activations_and_quantizers(self, layer):
#         return [
#             (
#                 layer.activation,
#                 self.activation_quantizer,
#             )
#         ]

#     def set_quantize_activations(self, layer, quantize_activations):
#         layer.activation = quantize_activations[0]

#     # This defines how to quantize outputs
#     def get_output_quantizers(self, layer):
#         return []

#     def get_config(self):
#         return {
#         }
