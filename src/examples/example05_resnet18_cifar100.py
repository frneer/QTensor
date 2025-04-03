#!/usr/bin/env python3

import argparse
#import importlib
import pickle

from pathlib import Path

from configs.qmodel import apply_quantization
from data_collection.callbacks import CaptureWeightCallback
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical


import numpy as np
from pathlib import Path

from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer

from functions import apply_bn_folding, compute_alpha_dict, apply_alpha_dict, print_model_weighs

qconfig_uniform = {
        "conv2d"    : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "conv2d_1"  : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "dense"     : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "dense_1"   : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "dense_2"   : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        }
qconfig_flex = {
        "conv2d"    : { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "conv2d_1"  : { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "dense"     : { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "dense_1"   : { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "dense_2"   : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, },
        }
qconfig = qconfig_flex

output_path = Path("snapshots")

# Pre-training parameters
pre_training_epochs = 5
pre_training_batch_size = 128
pre_training_learning_rate = 0.001 * (pre_training_batch_size/256)

# BN parameters
post_bnf_epochs = 1
post_bnf_batch_size = 128
post_bnf_learning_rate = 0.0005 * (post_bnf_batch_size/256)

# QAT parameters
qat_epochs = 200
qat_batch_size = 32
qat_learning_rate = 0.0001 * (qat_batch_size/256)


if __name__ == "__main__":

    # Load Fashion CIFAR100 dataset
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    input_shape = (None,) + x_train.shape[1:]
    image_shape = input_shape[1:]
    categories = 100

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)

    def conv_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)

        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        return x

    inputs = tf.keras.Input(shape=image_shape)
    x = layers.Conv2D(64, 3, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = conv_block(x, 64)
    x = conv_block(x, 64)
    x = conv_block(x, 128, stride=2)
    x = conv_block(x, 128)
    x = conv_block(x, 256, stride=2)
    x = conv_block(x, 256)
    x = conv_block(x, 512, stride=2)
    x = conv_block(x, 512)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(categories, activation='softmax')(x)
    model = models.Model(inputs, outputs)

    model.build(input_shape=input_shape)

    print(f"#####################################################")
    print(f"Model summary")
    model.summary(line_length=100)
    print(f"#####################################################\n")

    model.compile(
        optimizer=Adam(learning_rate=pre_training_learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"#####################################################")
    print(f"Pre-Training")
    if pre_training_epochs > 0:
        model.fit(x_train, y_train,
                  batch_size=pre_training_batch_size,
                  epochs=pre_training_epochs,
                  validation_split=0.1
                  )
    print(f"#####################################################\n")

    print(f"#####################################################")
    print(f"POST FIT Evaluation")
    loss, accuracy = model.evaluate(x=x_test, y=y_test)
    print(f"#####################################################\n")

    fmodel = apply_bn_folding(model, merge_activation=True)
    fmodel.build(input_shape=input_shape)
    fmodel.compile(
        optimizer=Adam(learning_rate=post_bnf_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"#####################################################")
    print(f"Post-BNF-Training")
    if post_bnf_epochs > 0:
        fmodel.fit(x_train, y_train,
                   batch_size=post_bnf_batch_size,
                   epochs=post_bnf_epochs,
                   validation_split=0.1
                   )
    print(f"#####################################################\n")

    print(f"#####################################################")
    print(f"Folded model summary")
    fmodel.summary(line_length=100)
    print(f"#####################################################\n")

    print(f"#####################################################")
    print(f"POST BN FOLDING Evaluation")
    loss, accuracy = fmodel.evaluate(x=x_test, y=y_test)
    print(f"#####################################################\n")

    # Apply quantization
    print(f"#####################################################")
    print(f"QConfig")
    for i, qc in enumerate(qconfig.values()):
        print(f"{i:04d}: {qc}")
    print(f"#####################################################\n")
    qmodel = apply_quantization(fmodel, qconfig)
    qmodel.build(input_shape=input_shape)
    qmodel.compile(
        optimizer=Adam(learning_rate=qat_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Initialize alpha values
    alpha_dict = compute_alpha_dict(fmodel, x_train)
    print("#####################################################")
    print("Alpha Dictionary (Weights and Activations)")
    for i, (layer_name, weights_dict) in enumerate(alpha_dict.items()):
        print(f"{i:04d}: {layer_name}")
        for key, alpha_value in weights_dict.items():
            print(f"    {key}: {alpha_value}")
    print("#####################################################\n")
    qmodel = apply_alpha_dict(qmodel, alpha_dict)

    #print("#####################################################")
    #print("DEBUG: Model weights")
    #print_model_weighs(model)
    #print("--------------------------------")
    #print_model_weighs(fmodel)
    #print("--------------------------------")
    #print_model_weighs(qmodel)
    #print("#####################################################\n")

    print(f"#####################################################")
    print(f"POST QUANTIZATION Evaluation")
    loss, accuracy = qmodel.evaluate(x=x_test, y=y_test)
    print(f"#####################################################\n")

    callback_tuples = [(CaptureWeightCallback(qlayer), qconfig[layer.name]) for layer, qlayer in zip(model.layers, qmodel.layers) if layer.name in qconfig]

    print(f"#####################################################")
    print(f"QAT")
    if qat_epochs > 0:
        hist = qmodel.fit(
            x_train,
            y_train,
            epochs=qat_epochs,
            batch_size=qat_batch_size,
            validation_split=0.1,
            callbacks=[callback for callback, _ in callback_tuples],
        )
    print(f"#####################################################\n")

    print(f"#####################################################")
    print(f"POST QAT Evaluation")
    loss, accuracy = qmodel.evaluate(x=x_test, y=y_test)
    print(f"#####################################################\n")

    #model.summary(line_length=100)
    #fmodel.summary(line_length=100)
    #qmodel.summary(line_length=100)

    output_dict = {}
    for callback, qconfig in callback_tuples:
        output_dict[callback.layer.name] = {}
        output_dict[callback.layer.name]["history"] = callback.get_history()
        output_dict[callback.layer.name]["qconfig"] = qconfig
    with open(output_path / "output_dict.pkl", "wb") as f:
        pickle.dump(output_dict, f)



