#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys
from tqdm import tqdm

#def apply_bn_folding(model: tf.keras.Model, merge_activation: bool = False) -> tf.keras.Model:
#    """
#    Creates a new model where each Conv2D or DepthwiseConv2D layer immediately followed by a 
#    BatchNormalization layer is fused into a single layer with updated weights.
#    Additionally, if merge_activation is True, a ReLU activation (either via an Activation('relu')
#    or a ReLU layer) immediately following the BN (or directly after the convolution if no BN)
#    is merged into the convolution layer.
#    
#    When folding BN, we compute:
#      new_kernel = kernel * (gamma / sqrt(moving_variance + epsilon))
#      new_bias   = beta - (gamma * moving_mean) / sqrt(moving_variance + epsilon) + bias * (gamma / sqrt(moving_variance + epsilon))
#    
#    This implementation now handles cases where the BN layer is configured with center=False or scale=False.
#    
#    Note:
#      - The model should be in inference mode (training=False) so that BN uses its moving averages.
#      - This function currently supports Sequential models.
#    """
#
#    def is_relu_activation(layer):
#        # Check if the layer represents a ReLU activation.
#        if isinstance(layer, tf.keras.layers.ReLU):
#            return True
#        if isinstance(layer, tf.keras.layers.Activation) and layer.get_config().get('activation') == 'relu':
#            return True
#        return False
#
#    new_layers = []
#    i = 0
#    while i < len(model.layers):
#        layer = model.layers[i]
#        # Process Conv2D or DepthwiseConv2D layers.
#        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
#            # Case 1: Convolution followed by BatchNormalization.
#            if (i + 1 < len(model.layers) and 
#                isinstance(model.layers[i+1], tf.keras.layers.BatchNormalization)):
#                conv_layer = layer
#                bn_layer = model.layers[i+1]
#                conv_weights = conv_layer.get_weights()
#                # Determine bias: if not used, create zeros.
#                if conv_layer.use_bias:
#                    bias = conv_weights[1]
#                else:
#                    if isinstance(conv_layer, tf.keras.layers.Conv2D):
#                        bias = np.zeros(conv_weights[0].shape[-1])
#                    else:  # DepthwiseConv2D
#                        bias = np.zeros(conv_weights[0].shape[2] * conv_weights[0].shape[3])
#                kernel = conv_weights[0]
#
#                # Get BN configuration and weights.
#                bn_config = bn_layer.get_config()
#                bn_weights = bn_layer.get_weights()
#                # Handle scale (gamma) and center (beta) flags.
#                if bn_config.get('scale', True):
#                    gamma = bn_weights[0]
#                else:
#                    # When scale is False, gamma is effectively 1.
#                    gamma = np.ones(kernel.shape[-1])
#                if bn_config.get('center', True):
#                    # If scale is True, weights order is [gamma, beta, moving_mean, moving_variance].
#                    # If scale is False, order is usually [beta, moving_mean, moving_variance].
#                    if bn_config.get('scale', True):
#                        beta = bn_weights[1]
#                    else:
#                        beta = bn_weights[0]
#                else:
#                    beta = np.zeros(gamma.shape)
#
#                # Get moving_mean and moving_variance.
#                if bn_config.get('scale', True):
#                    moving_mean = bn_weights[2]
#                    moving_variance = bn_weights[3]
#                else:
#                    moving_mean = bn_weights[1]
#                    moving_variance = bn_weights[2]
#
#                epsilon = bn_layer.epsilon
#
#                # Compute scaling factor.
#                scale = gamma / np.sqrt(moving_variance + epsilon)
#
#                # Fold the kernel weights.
#                if isinstance(conv_layer, tf.keras.layers.Conv2D):
#                    # For Conv2D, kernel shape: (kernel_h, kernel_w, in_channels, out_channels)
#                    new_kernel = kernel * scale.reshape((1, 1, 1, -1))
#                else:
#                    # For DepthwiseConv2D, kernel shape: (kernel_h, kernel_w, in_channels, channel_multiplier)
#                    in_channels = kernel.shape[2]
#                    channel_multiplier = kernel.shape[3]
#                    new_kernel = kernel * scale.reshape((1, 1, in_channels, channel_multiplier))
#
#                # Fold the bias.
#                new_bias = beta - (gamma * moving_mean) / np.sqrt(moving_variance + epsilon) + bias * scale
#
#                # Prepare the new convolution configuration.
#                new_config = conv_layer.get_config()
#                # When folding BN, we now need a bias.
#                new_config['use_bias'] = True
#                # By default, do not fuse any activation.
#                new_config['activation'] = None
#
#                skip = 2  # Skip conv and BN layers.
#                # Optionally merge a following ReLU activation.
#                if merge_activation and (i + 2 < len(model.layers)) and is_relu_activation(model.layers[i+2]):
#                    new_config['activation'] = 'relu'
#                    skip = 3  # Also skip the activation layer.
#
#                # Create the new convolution (or depthwise convolution) layer.
#                if isinstance(conv_layer, tf.keras.layers.Conv2D):
#                    new_conv = tf.keras.layers.Conv2D(**new_config)
#                else:
#                    new_conv = tf.keras.layers.DepthwiseConv2D(**new_config)
#                new_layers.append(new_conv)
#                # Build and set folded weights.
#                new_conv.build(conv_layer.input_shape)
#                new_conv.set_weights([new_kernel, new_bias])
#                i += skip
#                continue
#
#            # Case 2: No BN to fold but optionally merge a following activation.
#            if merge_activation and (i + 1 < len(model.layers)) and is_relu_activation(model.layers[i+1]):
#                conv_layer = layer
#                new_config = conv_layer.get_config()
#                new_config['activation'] = 'relu'
#                if isinstance(conv_layer, tf.keras.layers.Conv2D):
#                    new_conv = tf.keras.layers.Conv2D(**new_config)
#                else:
#                    new_conv = tf.keras.layers.DepthwiseConv2D(**new_config)
#                new_layers.append(new_conv)
#                new_conv.build(conv_layer.input_shape)
#                new_conv.set_weights(conv_layer.get_weights())
#                i += 2  # Skip the activation layer.
#                continue
#
#            # Otherwise, leave the convolution layer unchanged.
#            new_layers.append(layer)
#        else:
#            new_layers.append(layer)
#        i += 1
#
#    new_model = tf.keras.Sequential(new_layers)
#    return new_model

def apply_bn_folding(model: tf.keras.Model, merge_activation: bool = False) -> tf.keras.Model:
    """
    Creates a new model where each Conv2D, DepthwiseConv2D, or Dense layer immediately
    followed by a BatchNormalization layer is fused into a single layer with updated weights.
    Additionally, if merge_activation is True, a ReLU activation (either via an Activation('relu')
    or a ReLU layer) immediately following the BN (or directly after the layer if no BN)
    is merged into the layer.
    
    When folding BN, we compute:
      new_kernel = kernel * (gamma / sqrt(moving_variance + epsilon))
      new_bias   = beta - (gamma * moving_mean) / sqrt(moving_variance + epsilon) + bias * (gamma / sqrt(moving_variance + epsilon))
    
    This implementation handles cases where the BN layer is configured with center=False or scale=False.
    
    Note:
      - The model should be in inference mode (training=False) so that BN uses its moving averages.
      - This function currently supports Sequential models.
    """

    def is_relu_activation(layer):
        # Check if the layer represents a ReLU activation.
        if isinstance(layer, tf.keras.layers.ReLU):
            return True
        if isinstance(layer, tf.keras.layers.Activation) and layer.get_config().get('activation') == 'relu':
            return True
        return False

    new_layers = []
    i = 0
    while i < len(model.layers):
        layer = model.layers[i]
        # Process Conv2D, DepthwiseConv2D, or Dense layers.
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Dense)):
            # Case 1: Layer followed by BatchNormalization.
            if (i + 1 < len(model.layers) and
                isinstance(model.layers[i+1], tf.keras.layers.BatchNormalization)):
                base_layer = layer
                bn_layer = model.layers[i+1]
                weights = base_layer.get_weights()
                # Determine bias: if not used, create zeros.
                if base_layer.use_bias:
                    bias = weights[1]
                else:
                    if isinstance(base_layer, tf.keras.layers.Dense):
                        bias = np.zeros(weights[0].shape[-1])
                    elif isinstance(base_layer, tf.keras.layers.Conv2D):
                        bias = np.zeros(weights[0].shape[-1])
                    elif isinstance(base_layer, tf.keras.layers.DepthwiseConv2D):
                        bias = np.zeros(weights[0].shape[2] * weights[0].shape[3])
                kernel = weights[0]

                # Get BN configuration and weights.
                bn_config = bn_layer.get_config()
                bn_weights = bn_layer.get_weights()
                # Handle scale (gamma) and center (beta) flags.
                if bn_config.get('scale', True):
                    gamma = bn_weights[0]
                else:
                    # When scale is False, gamma is effectively 1.
                    gamma = np.ones(kernel.shape[-1])
                if bn_config.get('center', True):
                    if bn_config.get('scale', True):
                        beta = bn_weights[1]
                    else:
                        beta = bn_weights[0]
                else:
                    beta = np.zeros(gamma.shape)

                # Get moving_mean and moving_variance.
                if bn_config.get('scale', True):
                    moving_mean = bn_weights[2]
                    moving_variance = bn_weights[3]
                else:
                    moving_mean = bn_weights[1]
                    moving_variance = bn_weights[2]

                epsilon = bn_layer.epsilon

                # Compute scaling factor.
                scale_factor = gamma / np.sqrt(moving_variance + epsilon)

                # Fold the kernel weights according to the layer type.
                if isinstance(base_layer, tf.keras.layers.Conv2D):
                    # Kernel shape: (kernel_h, kernel_w, in_channels, out_channels)
                    new_kernel = kernel * scale_factor.reshape((1, 1, 1, -1))
                elif isinstance(base_layer, tf.keras.layers.DepthwiseConv2D):
                    # Kernel shape: (kernel_h, kernel_w, in_channels, channel_multiplier)
                    in_channels = kernel.shape[2]
                    channel_multiplier = kernel.shape[3]
                    new_kernel = kernel * scale_factor.reshape((1, 1, in_channels, channel_multiplier))
                elif isinstance(base_layer, tf.keras.layers.Dense):
                    # Kernel shape: (input_dim, output_dim)
                    new_kernel = kernel * scale_factor.reshape((1, -1))

                # Fold the bias.
                new_bias = beta - (gamma * moving_mean) / np.sqrt(moving_variance + epsilon) + bias * scale_factor

                # Prepare new configuration for the folded layer.
                new_config = base_layer.get_config()
                new_config['use_bias'] = True  # Bias is now folded.
                new_config['activation'] = None  # Remove activation for now.
                skip = 2  # We'll skip the base layer and the BN layer.
                # Optionally merge a following ReLU activation.
                if merge_activation and (i + 2 < len(model.layers)) and is_relu_activation(model.layers[i+2]):
                    new_config['activation'] = 'relu'
                    skip = 3  # Also skip the activation layer.
                    
                # Create the new folded layer.
                if isinstance(base_layer, tf.keras.layers.Conv2D):
                    new_layer = tf.keras.layers.Conv2D(**new_config)
                elif isinstance(base_layer, tf.keras.layers.DepthwiseConv2D):
                    new_layer = tf.keras.layers.DepthwiseConv2D(**new_config)
                elif isinstance(base_layer, tf.keras.layers.Dense):
                    new_layer = tf.keras.layers.Dense(**new_config)
                    
                new_layers.append(new_layer)
                new_layer.build(base_layer.input_shape)
                new_layer.set_weights([new_kernel, new_bias])
                i += skip
                continue

            # Case 2: No BN to fold but optionally merge a following activation.
            if merge_activation and (i + 1 < len(model.layers)) and is_relu_activation(model.layers[i+1]):
                base_layer = layer
                new_config = base_layer.get_config()
                new_config['activation'] = 'relu'
                if isinstance(base_layer, tf.keras.layers.Conv2D):
                    new_layer = tf.keras.layers.Conv2D(**new_config)
                elif isinstance(base_layer, tf.keras.layers.DepthwiseConv2D):
                    new_layer = tf.keras.layers.DepthwiseConv2D(**new_config)
                elif isinstance(base_layer, tf.keras.layers.Dense):
                    new_layer = tf.keras.layers.Dense(**new_config)
                new_layers.append(new_layer)
                new_layer.build(base_layer.input_shape)
                new_layer.set_weights(base_layer.get_weights())
                i += 2  # Skip the activation layer.
                continue

        # For other layers, add them unchanged.
        new_layers.append(layer)
        i += 1

    new_model = tf.keras.Sequential(new_layers)
    return new_model




import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model



def print_model_weighs(model):
    for layer in model.layers:
        print(f"  layer={layer.name}")
        for v in layer.weights:
            print(f"    weight={v.name}")

#def compute_alpha_dict(fmodel, x_train, batch_size=128):
#
#    # Assume fmodel and x_train are already defined.
#    # ----------------------------
#    # 1. Compute weight alpha values (as before)
#    alpha_dict = {}
#    for layer in fmodel.layers:
#        for v in layer.weights:
#            # Expected format: "layer_name/weight_type:0"
#            name_parts = v.name.split("/")
#            if len(name_parts) == 1 or len(name_parts) == 2:
#                layer_key = layer.name
#                # Remove the trailing ":0" from the weight type.
#                weight_type = name_parts[0].split(":")[0]
#            if len(name_parts) == 2:
#                layer_key = name_parts[0]
#                # Remove the trailing ":0" from the weight type.
#                weight_type = name_parts[1].split(":")[0]
#            if len(name_parts) == 0 or len(name_parts) > 2:
#                assert(False, "Unexpected format: {v.name} -> len(parts)={len(name_parts)}")
#    
#            # Compute alpha_value as the maximum of the absolute min and max values.
#            v_array = v.numpy() if hasattr(v, "numpy") else v
#            alpha_value = np.max(np.abs([np.max(v_array), np.min(v_array)]))
#    
#            # Create the nested dictionary if necessary.
#            if layer_key not in alpha_dict:
#                alpha_dict[layer_key] = {}
#            alpha_dict[layer_key][weight_type] = alpha_value
#    
#    # ----------------------------
#    # 2. Compute activation alpha values using batches
#    num_samples = x_train.shape[0]
#    num_batches = int(np.ceil(num_samples / batch_size))
#    
#    # Initialize the activation entry for each layer in fmodel.
#    for layer in fmodel.layers:
#        layer_key = layer.name
#        if layer_key not in alpha_dict:
#            alpha_dict[layer_key] = {}
#        # Set an initial value of 0 to later update the maximum.
#        alpha_dict[layer_key]["activation"] = 0
#
#        #def has_activation(layer):
#        #    act = getattr(layer, "activation", None)
#        #    return act is not None and act != tf.keras.activations.linear
#        #alpha_dict[layer_key]["has_activation"] = has_activation(layer)
#    
#    # Iterate over batches of x_train.
#    for i in range(num_batches):
#        # Get a batch of inputs
#        batch_input = x_train[i * batch_size : (i + 1) * batch_size]
#        current_activation = batch_input
#        # Propagate the batch through each layer of the model.
#        for layer in fmodel.layers:
#            current_activation = layer(current_activation)
#            # Compute the maximum absolute activation in the current batch.
#            current_max = np.max(np.abs(current_activation))
#            layer_key = layer.name
#            # Update if this batch has a larger value.
#            if current_max > alpha_dict[layer_key]["activation"]:
#                alpha_dict[layer_key]["activation"] = current_max
#    
#
#    return alpha_dict


#def compute_alpha_dict(fmodel, x_train, batch_size=128):
#    """
#    Computes a nested dictionary of alpha values for weights and activations for each layer
#    of the given model.
#
#    The dictionary has the following structure:
#      {
#         'layer_name': {
#              'kernel': <alpha_value from weight>,
#              'bias': <alpha_value from bias>,
#              'activation': <alpha_value from activations>
#         },
#         ...
#      }
#
#    For activations, the alpha value is computed as the maximum of the absolute values
#    over the training set. The training data (x_train) can be either a NumPy array or a
#    tf.data.Dataset.
#
#    Parameters:
#      fmodel: A Keras model whose layers will be used to compute alpha values.
#      x_train: The training inputs. It can be a NumPy array or a tf.data.Dataset.
#      batch_size: Batch size to use if x_train is a NumPy array.
#
#    Returns:
#      alpha_dict: The nested dictionary containing computed alpha values.
#    """
#    # ----------------------------
#    # 1. Compute weight alpha values
#    alpha_dict = {}
#    for layer in tqdm(fmodel.layers, desc="Computing weight alpha values"):
#        for v in layer.weights:
#            # Expected format: "layer_name/weight_type:0" or just "weight_type:0"
#            name_parts = v.name.split("/")
#            if len(name_parts) == 1:
#                layer_key = layer.name
#                weight_type = name_parts[0].split(":")[0]
#            elif len(name_parts) == 2:
#                layer_key = name_parts[0]
#                weight_type = name_parts[1].split(":")[0]
#            else:
#                raise ValueError(f"Unexpected format: {v.name} -> len(parts)={len(name_parts)}")
#
#            # Compute alpha_value as the maximum of the absolute minimum and maximum values.
#            v_array = v.numpy() if hasattr(v, "numpy") else v
#            alpha_value = np.max(np.abs([np.max(v_array), np.min(v_array)]))
#
#            # Create the nested dictionary if necessary.
#            if layer_key not in alpha_dict:
#                alpha_dict[layer_key] = {}
#            alpha_dict[layer_key][weight_type] = alpha_value
#    
#    del v_array
#
#    # ----------------------------
#    # 2. Initialize activation entries
#    for layer in fmodel.layers:
#        layer_key = layer.name
#        if layer_key not in alpha_dict:
#            alpha_dict[layer_key] = {}
#        # Set an initial value of 0 to later update the maximum.
#        alpha_dict[layer_key]["activation"] = 0
#
#    # Define a helper function to process a batch.
#    def process_batch(batch_input):
#        # If the dataset yields a tuple (inputs, labels), take the inputs.
#        if isinstance(batch_input, (tuple, list)):
#            batch_input = batch_input[0]
#        current_activation = batch_input
#        for layer in fmodel.layers:
#            current_activation = layer(current_activation)
#            current_max = np.max(np.abs(current_activation))
#            layer_key = layer.name
#            if current_max > alpha_dict[layer_key]["activation"]:
#                alpha_dict[layer_key]["activation"] = current_max
#
#    # ----------------------------
#    # 3. Compute activation alpha values with progress bar
#    if isinstance(x_train, tf.data.Dataset):
#        for batch in tqdm(x_train, desc="Processing activations"):
#            process_batch(batch)
#    else:
#        # Assume x_train is a NumPy array.
#        num_samples = x_train.shape[0]
#        num_batches = int(math.ceil(num_samples / batch_size))
#        for i in tqdm(range(num_batches), desc="Processing activations"):
#            batch_input = x_train[i * batch_size : (i + 1) * batch_size]
#            process_batch(batch_input)
#
#    return alpha_dict
import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm

def compute_alpha_dict(fmodel, x_train, batch_size=128):
    """
    Computes a nested dictionary of alpha values for weights and activations for each layer
    of the given model. Alpha values for weights are computed using TensorFlow ops to avoid
    unnecessary tensor-to-numpy conversions. Activations are computed in a batched and
    memory-efficient manner via a compiled tf.function.
    """
    # ----------------------------
    # 1. Compute weight alpha values using TensorFlow operations
    alpha_dict = {}
    for layer in tqdm(fmodel.layers, desc="Computing weight alpha values", file=sys.stdout):
        for v in layer.weights:
            # Expected format: "layer_name/weight_type:0" or just "weight_type:0"
            name_parts = v.name.split("/")
            if len(name_parts) == 1:
                layer_key = layer.name
                weight_type = name_parts[0].split(":")[0]
            elif len(name_parts) == 2:
                layer_key = name_parts[0]
                weight_type = name_parts[1].split(":")[0]
            else:
                raise ValueError(f"Unexpected format: {v.name} -> len(parts)={len(name_parts)}")

            # Compute alpha_value using TF operations: maximum of the absolute maximum and minimum.
            abs_v = tf.abs(v)
            max_val = tf.reduce_max(abs_v)
            # Compute absolute of the minimum
            min_val = tf.abs(tf.reduce_min(v))
            alpha_value = max(max_val.numpy(), min_val.numpy())

            if layer_key not in alpha_dict:
                alpha_dict[layer_key] = {}
            alpha_dict[layer_key][weight_type] = alpha_value

    # ----------------------------
    # 2. Initialize activation entries for each layer.
    for layer in fmodel.layers:
        layer_key = layer.name
        if layer_key not in alpha_dict:
            alpha_dict[layer_key] = {}
        # Set an initial activation value of 0.
        alpha_dict[layer_key]["activation"] = 0

    # ----------------------------
    # 3. Compute activation alpha values using a compiled function for efficiency.
    @tf.function
    def get_batch_maxes(batch_input):
        current_activation = batch_input
        batch_maxes = []
        for layer in fmodel.layers:
            current_activation = layer(current_activation)
            # Compute maximum absolute value for this layer.
            batch_max = tf.reduce_max(tf.abs(current_activation))
            batch_maxes.append(batch_max)
        return batch_maxes

    # Process activations: If x_train is a tf.data.Dataset, iterate over it; otherwise, treat it as a NumPy array.
    if isinstance(x_train, tf.data.Dataset):
        for batch in tqdm(x_train, desc="Processing activations", file=sys.stdout):
            # If the dataset yields a tuple (inputs, labels), select the inputs.
            batch_input = batch[0] if isinstance(batch, (tuple, list)) else batch
            batch_maxes = get_batch_maxes(batch_input)
            for i, layer in enumerate(fmodel.layers):
                layer_key = layer.name
                current_max = batch_maxes[i].numpy()
                if current_max > alpha_dict[layer_key]["activation"]:
                    alpha_dict[layer_key]["activation"] = current_max
    else:
        num_samples = x_train.shape[0]
        num_batches = int(math.ceil(num_samples / batch_size))
        for i in tqdm(range(num_batches), desc="Processing activations", file=sys.stdout):
            batch_input = x_train[i * batch_size : (i + 1) * batch_size]
            batch_maxes = get_batch_maxes(batch_input)
            for j, layer in enumerate(fmodel.layers):
                layer_key = layer.name
                current_max = batch_maxes[j].numpy()
                if current_max > alpha_dict[layer_key]["activation"]:
                    alpha_dict[layer_key]["activation"] = current_max

    return alpha_dict







def apply_alpha_dict(qmodel, alpha_dict):
    for layer in qmodel.layers:
        #print(f"  Layer: {layer.name}")
        #for w in layer.weights:
        #    print(f"    Weight name: {w.name}")

        orig_layer_name = layer.name
        if orig_layer_name.startswith("quant_"):
            orig_layer_name = orig_layer_name[len("quant_"):]

        if orig_layer_name in alpha_dict:
            for alpha_type in ["kernel", "bias", "activation"]:
                new_alpha = alpha_dict[orig_layer_name].get(alpha_type, None)
                if new_alpha is not None:
                    for v in layer.weights:
                        if "alpha" in v.name and alpha_type in v.name:
                            v.assign(new_alpha)
                            #print(f"Updated {v.name} ({alpha_type}) with new alpha value {new_alpha}")
                        elif alpha_type == "activation" and "post_activation" in v.name and "alpha" in v.name:
                            v.assign(new_alpha)
                            #print(f"Updated {v.name} (activation) with new alpha value {new_alpha}")

    return qmodel


# TODO(Colo): COMPLETE this
def apply_initial_values(qmodel, initial_values_dict):
    """
    TODO(Colo): Implementation missing.
    Some functions heuristically define good initial values for levels and thresholds.
    """
    raise NotImplementedError(
        "apply_initial_values is not yet implemented: "
        "Some functions heuristically define good initial values for levels and thresholds."
    )









import tensorflow as tf

def report_memory_usage():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpu_name = gpus[0].name.split("/")[-1]  # ✅ get 'GPU:0' from '/physical_device:GPU:0'
        memory_info = tf.config.experimental.get_memory_info("GPU:0")
        print(f"  DEBUG: Current GPU memory usage: {memory_info['current'] / (1024 ** 2):.2f} MB")
        print(f"  DEBUG: Peak GPU memory usage: {memory_info['peak'] / (1024 ** 2):.2f} MB")
    else:
        print(f"  DEBUG: No GPU available.")

















from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_data(dataset):

    if dataset == "mnist":
        # Load dataset
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        input_shape = (None,) + x_train.shape[1:] + (1,)
        image_shape = input_shape[1:]
        categories = 10

        x_train = x_train.reshape(-1, *image_shape).astype("float32") / 255.0
        x_test = x_test.reshape(-1, *image_shape).astype("float32") / 255.0

        y_train = to_categorical(y_train, categories)
        y_test = to_categorical(y_test, categories)

        data = {
                'x_train' : x_train,
                'y_train' : y_train,
                'x_test'  : x_test,
                'y_test'  : y_test,
                # TODO(Colo): How to handle data generators
                # TODO(Colo): How to handle validation data for all cases
                }

        return data

    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")










from tensorflow.keras import layers, models

def model_create(params):

    input_shape = params['input_shape']
    categories = params['categories']
    model_name = params['model_name']

    if model_name == "lenet5_custom":
        model = models.Sequential()
        model.add(layers.Conv2D(6, kernel_size=5, activation='relu', padding='same'))
        model.add(layers.AveragePooling2D())
        model.add(layers.Conv2D(16, kernel_size=5, activation='relu'))
        model.add(layers.AveragePooling2D())
        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(84, activation='relu'))
        model.add(layers.Dense(categories, activation='softmax'))

        model.build(input_shape=input_shape)

        model.compile(
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")


def model_train(model, params):
    learning_rate    = params['learning_rate']
    epochs           = params['epochs']
    batch_size       = params['batch_size']
    validation_split = params['validation_split']
    dataset          = params['dataset']
    data             = load_data(dataset)
    x_train          = data['x_train']
    y_train          = data['y_train']
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    if epochs > 0:
        model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split
            # callbacks=[callback for callback, _ in callback_tuples],
        )
    return model


def model_transform_bnf(model, params):
    merge_activation = params['merge_activation']
    model = apply_bn_folding(model, merge_activation=merge_activation)
    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer
from configs.qmodel import apply_quantization

def model_quantize(model, params):

    kernel      = params['kernel']
    bias        = params['bias']
    activations = params['activations']

    # Layers initialization
    layers = list()
    supported = ("conv2d", "dense")
    for layer in model.layers:
        if any(kw in layer.name for kw in supported):
            layers.append(layer.name)

    # QConfig initialization
    qconfig = dict()
    for layer in layers:
        qconfig[layer] = dict()
    for layer in layers:
        for k in ('weights', 'activations'):
            qconfig[layer][k] = dict()

    for layer, k, b, a in zip(layers, kernel, bias, activations):
        # Kernel
        if k['type'] == "uniform":
            qconfig[layer]['weights']['kernel'] = UniformQuantizer(bits=k['bits'], signed=True)
        elif k['type'] == "flexible":
            qconfig[layer]['weights']['kernel'] = FlexQuantizer(bits=k['bits'], n_levels=k['n_levels'], signed=True)
        else:
            pass
        # Bias
        if b['type'] == "uniform":
            qconfig[layer]['weights']['bias'] = UniformQuantizer(bits=b['bits'], signed=True)
        elif b['type'] == "flexible":
            qconfig[layer]['weights']['bias'] = FlexQuantizer(bits=b['bits'], n_levels=b['n_levels'], signed=True)
        else:
            pass
        # Arctivations
        if a['type'] == "uniform":
            qconfig[layer]['activations']['activation'] = UniformQuantizer(bits=a['bits'], signed=False)
        elif a['type'] == "flexible":
            qconfig[layer]['activations']['activation'] = FlexQuantizer(bits=a['bits'], n_levels=a['n_levels'], signed=False)
        else:
            pass

    ## For debugging
    ##for k in qconfig:
    ##    print(f'DEBUG: {k} -> {qconfig[k]}')

    #qconfig_uniform = {
    #        "conv2d"    : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
    #        "conv2d_1"  : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
    #        "dense"     : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
    #        "dense_1"   : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
    #        "dense_2"   : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
    #        }
    #qconfig_flex = {
    #        "conv2d"    : { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
    #        "conv2d_1"  : { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
    #        "dense"     : { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
    #        "dense_1"   : { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
    #        "dense_2"   : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, },
    #        }
    #qconfig = qconfig_flex

    ## For debugging
    #print(f"DEBUG: QConfig")
    #for i, qc in enumerate(qconfig.values()):
    #    print(f"DEBUG: {i:04d}: {qc}")
    #print(f"DEBUG: ------------------")
    model = apply_quantization(model, qconfig)
    input_shape = params['input_shape']
    model.build(input_shape=input_shape)
    model.compile(
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def model_initialize_parameters(model, params, ref_model):

    if params['type'] == "alpha":
        dataset          = params['dataset']
        data             = load_data(dataset)
        x_train          = data['x_train']

        alpha_dict = compute_alpha_dict(ref_model, x_train)
        ## For debugging
        #print(f"DEBUG: QConfig")
        #for i, (layer_name, weights_dict) in enumerate(alpha_dict.items()):
        #    print(f"{i:04d}: {layer_name}")
        #    for key, alpha_value in weights_dict.items():
        #        print(f"    {key}: {alpha_value}")
        model = apply_alpha_dict(model, alpha_dict)

        return model

    else:
        raise ValueError(f"Unknown type: {type!r}")


from tensorflow.keras.optimizers import Adam

def run_stage(model, params, other, ref_model=None):
    stage_type = params['stage_type']


    if stage_type == 'model_creation':
        func_name   = params['stage_function']
        func_params = params['stage_parameters']
        func = globals().get(func_name)
        if func is None:
            raise NameError(f"Function '{func_name}' is not defined")
        return func(func_params)

    elif stage_type == 'training':
        func_name   = params['stage_function']
        func_params = params['stage_parameters']
        func = globals().get(func_name)
        if func is None:
            raise NameError(f"Function '{func_name}' is not defined")
        return func(model, func_params)

    elif stage_type == 'model_transformation':
        func_name   = params['stage_function']
        func_params = params['stage_parameters']
        func = globals().get(func_name)
        if func is None:
            raise NameError(f"Function '{func_name}' is not defined")
        model = func(model, func_params)
        return model

    elif stage_type == 'parameter_initialization':
        if ref_model is None:
            raise ValueError(f"A reference model (ref_model parameter) must be provided")
        func_name   = params['stage_function']
        func_params = params['stage_parameters']
        func = globals().get(func_name)
        if func is None:
            raise NameError(f"Function '{func_name}' is not defined")
        model = func(model, func_params, ref_model)
        return model

    else:
        raise ValueError(f"Unknown stage_type: {stage_type!r}")




import numpy as np

def get_layers_size(model):
    """
    For each layer in `model` returns three parallel lists:
      - kernel_sizes:   # of elements in layer.kernel (or first weight array)
      - bias_sizes:     # of elements in layer.bias   (or second weight array)
      - activation_sizes: product of layer.output_shape[1:]
    
    Layers without a kernel/bias get size 0 in the corresponding list.
    Layers with dynamic output shapes (None in dims) also yield 0 activation size.
    """
    kernel_sizes = []
    bias_sizes = []
    activation_sizes = []

    for layer in model.layers:
        # inspect weights
        weights = layer.get_weights()  # list of numpy arrays
        if len(weights) >= 1:
            kernel_sizes.append(int(np.prod(weights[0].shape)))
        else:
            kernel_sizes.append(0)

        if len(weights) >= 2:
            bias_sizes.append(int(np.prod(weights[1].shape)))
        else:
            bias_sizes.append(0)

        # inspect output shape (may be tuple or list)
        out_shape = layer.output_shape
        # out_shape[0] is batch dim; skip it
        if hasattr(out_shape, "__iter__") and len(out_shape) > 1:
            dims = out_shape[1:]
            # only count if all dims are fixed integers
            if all(isinstance(d, int) for d in dims):
                activation_sizes.append(int(np.prod(dims)))
            else:
                activation_sizes.append(0)
        else:
            activation_sizes.append(0)

    return kernel_sizes, bias_sizes, activation_sizes


