#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

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

def compute_alpha_dict(fmodel, x_train, batch_size=128):

    # Assume fmodel and x_train are already defined.
    # ----------------------------
    # 1. Compute weight alpha values (as before)
    alpha_dict = {}
    for layer in fmodel.layers:
        for v in layer.weights:
            # Expected format: "layer_name/weight_type:0"
            name_parts = v.name.split("/")
            if len(name_parts) == 1 or len(name_parts) == 2:
                layer_key = layer.name
                # Remove the trailing ":0" from the weight type.
                weight_type = name_parts[0].split(":")[0]
            if len(name_parts) == 2:
                layer_key = name_parts[0]
                # Remove the trailing ":0" from the weight type.
                weight_type = name_parts[1].split(":")[0]
            if len(name_parts) == 0 or len(name_parts) > 2:
                assert(False, "Unexpected format: {v.name} -> len(parts)={len(name_parts)}")
    
            # Compute alpha_value as the maximum of the absolute min and max values.
            v_array = v.numpy() if hasattr(v, "numpy") else v
            alpha_value = np.max(np.abs([np.max(v_array), np.min(v_array)]))
    
            # Create the nested dictionary if necessary.
            if layer_key not in alpha_dict:
                alpha_dict[layer_key] = {}
            alpha_dict[layer_key][weight_type] = alpha_value
    
    # ----------------------------
    # 2. Compute activation alpha values using batches
    num_samples = x_train.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    
    # Initialize the activation entry for each layer in fmodel.
    for layer in fmodel.layers:
        layer_key = layer.name
        if layer_key not in alpha_dict:
            alpha_dict[layer_key] = {}
        # Set an initial value of 0 to later update the maximum.
        alpha_dict[layer_key]["activation"] = 0

        #def has_activation(layer):
        #    act = getattr(layer, "activation", None)
        #    return act is not None and act != tf.keras.activations.linear
        #alpha_dict[layer_key]["has_activation"] = has_activation(layer)
    
    # Iterate over batches of x_train.
    for i in range(num_batches):
        # Get a batch of inputs
        batch_input = x_train[i * batch_size : (i + 1) * batch_size]
        current_activation = batch_input
        # Propagate the batch through each layer of the model.
        for layer in fmodel.layers:
            current_activation = layer(current_activation)
            # Compute the maximum absolute activation in the current batch.
            current_max = np.max(np.abs(current_activation))
            layer_key = layer.name
            # Update if this batch has a larger value.
            if current_max > alpha_dict[layer_key]["activation"]:
                alpha_dict[layer_key]["activation"] = current_max
    

    return alpha_dict




def apply_alpha_dict(qmodel, alpha_dict):
    print("#####################################################")
    print("Apply alpha values")

    for layer in qmodel.layers:

        print(f"  Layer: {layer.name}")
        for w in layer.weights:
            print(f"    Weight name: {w.name}")

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
                            print(f"Updated {v.name} ({alpha_type}) with new alpha value {new_alpha}")
                        elif alpha_type == "activation" and "post_activation" in v.name and "alpha" in v.name:
                            v.assign(new_alpha)
                            print(f"Updated {v.name} (activation) with new alpha value {new_alpha}")

    print("#####################################################\n")

    return qmodel


def apply_initial_values(qmodel, initial_values_dict):
    assert(False, "TODO(Colo): Implementation missing -> Some fuctions the heuristically define good initial values for levels and thresholds.")
