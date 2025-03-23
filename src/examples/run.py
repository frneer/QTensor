#!/usr/bin/env python3

import argparse
import importlib
import pickle

from pathlib import Path

from configs.qmodel import apply_quantization
from data_collection.callbacks import CaptureWeightCallback
from tensorflow.keras.optimizers import Adam


import tensorflow as tf
import numpy as np

def apply_bn_folding(model: tf.keras.Model, merge_activation: bool = False) -> tf.keras.Model:
    """
    Creates a new model where each Conv2D or DepthwiseConv2D layer immediately followed by a
    BatchNormalization layer is fused into a single layer with updated weights.
    Additionally, if merge_activation is True, a ReLU activation (either via an Activation('relu')
    or a ReLU layer) immediately following the BN (or directly after the convolution if no BN)
    is merged into the convolution layer.

    The BN folding is performed using the formulas:
      new_kernel = kernel * (gamma / sqrt(variance + epsilon))
      new_bias = beta - (gamma * moving_mean) / sqrt(variance + epsilon) + (bias if any) * (gamma / sqrt(variance + epsilon))

    Parameters:
      model: The original Keras Sequential model.
      merge_activation: If True, merge a subsequent ReLU activation layer into the convolution.

    Note:
      - This function currently supports Sequential models.
      - It assumes that the BN layer’s moving_mean and moving_variance are already updated (i.e. the model has been trained).
    """

    def is_relu_activation(layer):
        # Checks if the layer is a ReLU activation.
        if isinstance(layer, tf.keras.layers.ReLU):
            return True
        if isinstance(layer, tf.keras.layers.Activation) and layer.get_config().get('activation') == 'relu':
            return True
        return False

    new_layers = []
    i = 0
    while i < len(model.layers):
        layer = model.layers[i]
        # Check for convolutional layers (Conv2D or DepthwiseConv2D).
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            folded = False
            # Case 1: Convolution followed by BN folding.
            if i + 1 < len(model.layers) and isinstance(model.layers[i+1], tf.keras.layers.BatchNormalization):
                conv_layer = layer
                bn_layer = model.layers[i+1]

                # Get convolution weights and determine bias.
                conv_weights = conv_layer.get_weights()
                if conv_layer.use_bias:
                    bias = conv_weights[1]
                else:
                    # For Conv2D: shape (filters,), for DepthwiseConv2D: shape (in_channels * channel_multiplier)
                    if isinstance(conv_layer, tf.keras.layers.Conv2D):
                        bias = np.zeros(conv_weights[0].shape[-1])
                    else:  # DepthwiseConv2D
                        bias = np.zeros(conv_weights[0].shape[2] * conv_weights[0].shape[3])
                kernel = conv_weights[0]

                # Get BN parameters: gamma, beta, moving_mean, moving_variance.
                bn_weights = bn_layer.get_weights()
                gamma, beta, moving_mean, moving_variance = bn_weights
                epsilon = bn_layer.epsilon

                # Compute scaling factor.
                scale = gamma / np.sqrt(moving_variance + epsilon)

                # Fold the kernel weights.
                if isinstance(conv_layer, tf.keras.layers.Conv2D):
                    # Kernel shape: (kernel_h, kernel_w, in_channels, out_channels)
                    new_kernel = kernel * scale.reshape((1, 1, 1, -1))
                else:  # DepthwiseConv2D
                    # Kernel shape: (kernel_h, kernel_w, in_channels, channel_multiplier)
                    in_channels = kernel.shape[2]
                    channel_multiplier = kernel.shape[3]
                    new_kernel = kernel * scale.reshape((1, 1, in_channels, channel_multiplier))

                # Fold the bias.
                new_bias = beta - (gamma * moving_mean) / np.sqrt(moving_variance + epsilon) + bias * scale

                # Prepare new layer configuration.
                new_config = conv_layer.get_config()
                new_config['use_bias'] = True  # Bias is now folded.
                # By default, set activation to None.
                new_config['activation'] = None

                folded = True
                skip = 2  # We have consumed conv and BN.
                # Optionally merge an activation if present.
                if merge_activation and i + 2 < len(model.layers) and is_relu_activation(model.layers[i+2]):
                    # Merge the activation by setting the activation in the convolution.
                    new_config['activation'] = 'relu'
                    skip = 3  # Also skip the activation layer.
                # Create the new convolution layer.
                if isinstance(conv_layer, tf.keras.layers.Conv2D):
                    new_conv = tf.keras.layers.Conv2D(**new_config)
                else:
                    new_conv = tf.keras.layers.DepthwiseConv2D(**new_config)
                new_layers.append(new_conv)
                # Build and set weights.
                new_conv.build(conv_layer.input_shape)
                new_conv.set_weights([new_kernel, new_bias])
                i += skip
                continue

            # Case 2: No BN folding but merge activation if applicable.
            if merge_activation and i + 1 < len(model.layers) and is_relu_activation(model.layers[i+1]):
                conv_layer = layer
                new_config = conv_layer.get_config()
                # Merge the ReLU activation into the convolution.
                new_config['activation'] = 'relu'
                # Create a new convolution layer with the same weights.
                if isinstance(conv_layer, tf.keras.layers.Conv2D):
                    new_conv = tf.keras.layers.Conv2D(**new_config)
                else:
                    new_conv = tf.keras.layers.DepthwiseConv2D(**new_config)
                new_layers.append(new_conv)
                new_conv.build(conv_layer.input_shape)
                new_conv.set_weights(conv_layer.get_weights())
                i += 2  # Skip the activation layer.
                continue

            # If neither BN nor mergeable activation is found, just add the layer as is.
            new_layers.append(layer)
        else:
            # For non-convolutional layers, add them unchanged.
            new_layers.append(layer)
        i += 1

    # Create a new Sequential model with the updated layers.
    new_model = tf.keras.Sequential(new_layers)
    return new_model





def main(args):
    module_name = f"models.{args.model}"
    model_module = importlib.import_module(module_name)
    model = model_module.model

    print(f"Loaded model from {module_name}: {model}")


    qconfig = model_module.qconfigs[args.qconfig]

    print(f"Applying quantization configuration {args.qconfig}: {qconfig}")

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    dataset_module_name = f"datasets.{args.dataset}"
    dataset_module = importlib.import_module(dataset_module_name)
    train_dataset, validation_dataset, test_dataset = dataset_module.generate_dataset()

    input_shape = (None, 224, 224, 3)
    model.build(input_shape=input_shape)

    loss, accuracy = model.evaluate(test_dataset)
    print(f"PRE FIT Evaluation: loss={loss}, accuracy={accuracy}")

    # Pretrain the model to get a baseline
    if args.pre_training_epochs > 0:
        model.fit(
            train_dataset,
            epochs=args.pre_training_epochs,
            #batch_size=args.pre_training_batch_size,
            validation_data=validation_dataset
        )

    loss, accuracy = model.evaluate(test_dataset)
    print(f"POST FIT Evaluation: loss={loss}, accuracy={accuracy}")
    #model.summary(line_length=100)

    fmodel = apply_bn_folding(model, True)
    fmodel.build(input_shape=input_shape)
    fmodel.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    loss, accuracy = fmodel.evaluate(test_dataset)
    print(f"POST BN FOLDING Evaluation: loss={loss}, accuracy={accuracy}")
    #fmodel.summary(line_length=100)

    # Apply quantization
    qmodel = apply_quantization(fmodel, qconfig)
    qmodel.build(input_shape=input_shape)
    qmodel.compile(
        optimizer=Adam(learning_rate=0.001 / 20),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    loss, accuracy = qmodel.evaluate(test_dataset)
    print(f"POST QUANTIZATION Evaluation: loss={loss}, accuracy={accuracy}")


    callback_tuples = [(CaptureWeightCallback(qlayer), qconfig[layer.name]) for layer, qlayer in zip(model.layers, qmodel.layers) if layer.name in qconfig]

    if args.epochs > 0:
        hist = qmodel.fit(
            train_dataset,
            epochs=args.epochs,
            #batch_size=args.batch_size,
            validation_data=validation_dataset,
            callbacks=[callback for callback, _ in callback_tuples],
        )

    loss, accuracy = qmodel.evaluate(test_dataset)
    print(f"POST QAT Evaluation: loss={loss}, accuracy={accuracy}")
    #qmodel.summary(line_length=100)

    output_dict = {}
    for callback, qconfig in callback_tuples:
        output_dict[callback.layer.name] = {}
        output_dict[callback.layer.name]["history"] = callback.get_history()
        output_dict[callback.layer.name]["qconfig"] = qconfig
    with open(args.output_path / "output_dict.pkl", "wb") as f:
        pickle.dump(output_dict, f)

    #qmodel.evaluate(test_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Model to train",
        required=True
    )
    parser.add_argument(
        "--qconfig",
        type=str,
        help="Quantization configuration",
        required=True
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to use",
        required=True
    )

    parser.add_argument(
        "--pre-training-epochs",
        type=int,
        default=2,
        help="Number of epochs for pre-training"
    )
    parser.add_argument(
        "--pre-training-batch-size",
        type=int,
        default=1024 // 4,
        help="Batch size for pre-training"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024 // 4,
        help="Batch size for training"
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("snapshots"),
        help="Path to save snapshots"
    )

    args = parser.parse_args()
    main(args)
