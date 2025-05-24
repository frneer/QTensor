from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras.quantize_layer import (
    QuantizeLayer,
)
from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import (
    QuantizeWrapperV2,
)

from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer


def compute_space_complexity_quantize(qlayer: QuantizeWrapperV2) -> float:
    """Compute the uniform space complexity of a layer based on its
    quantization configuration.

    returns:
        The space complexity of the layer in bits.
    """
    if not isinstance(qlayer, QuantizeWrapperV2):
        raise ValueError("Layer is not a QuantizeWrapperV2")

    total_layer_size = 0
    qconfig = qlayer.quantize_config

    # Assumption: order is the same for layer.weights and get_weights_and_quantizers
    weights_and_quantizers = qconfig.get_weights_and_quantizers(qlayer.layer)
    weights = qlayer.weights[: len(weights_and_quantizers)]

    for weight, weight_and_quantizer in zip(weights, weights_and_quantizers):
        quantizer = weight_and_quantizer[1]
        if isinstance(quantizer, UniformQuantizer):
            weight_size = weight.shape.num_elements() * quantizer.bits
        elif isinstance(quantizer, FlexQuantizer):
            qweight = quantizer.quantize_op(weight)
            counter = Counter(qweight.numpy().flatten())
            total = sum(counter.values())
            probabilities = np.array(
                [freq / total for freq in counter.values()]
            )
            entropy = -np.sum(probabilities * np.log2(probabilities))
            weight_size = weight.shape.num_elements() * entropy
        else:
            raise ValueError(f"Unknown quantizer type: {type(quantizer)}")
        total_layer_size += weight_size

    return total_layer_size


def compute_huffman_nominal_complexity(qweights):
    """Compute the nominal complexity of a huffman codification of the
    quantized weights."""
    N = qweights.shape.num_elements()
    counter = Counter(qweights.numpy().flatten())
    total = sum(counter.values())
    probabilities = np.array([freq / total for freq in counter.values()])
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return N * entropy


def compute_space_complexity(layer):
    """Compute the space complexity for a normal layer."""
    total_layer_size = 0
    for weight in layer.weights:
        weight_size = (
            8 * weight.dtype.size * weight.shape.num_elements()
        )  # bits
        total_layer_size += weight_size

    return total_layer_size


def compute_space_complexity_model(model: tf.keras.Model) -> float:
    """Compute the uniform space complexity of a model based on its
    quantization configuration."""
    total_space_complexity = 0

    for layer in model.layers:
        print(f"Layer {layer}")
        if isinstance(layer, QuantizeWrapperV2):
            layer_size = compute_space_complexity_quantize(layer)
        elif isinstance(layer, QuantizeLayer):
            # Verify if there's no layer we need to keep of this type.
            print(f"Skipping {layer.name}")
            continue
        else:
            layer_size = compute_space_complexity(layer)
        print(f"Layer: {layer.name}, Size: {layer_size} bits")
        total_space_complexity += layer_size

    return total_space_complexity
