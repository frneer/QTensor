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
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical


import tensorflow as tf
import numpy as np
from pathlib import Path

from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer

from functions import apply_bn_folding, compute_alpha_dict, apply_alpha_dict



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
pre_training_epochs = 15
pre_training_batch_size = 128
pre_training_learning_rate = 0.001 * (pre_training_batch_size/256)

# QAT parameters
epochs = 30
batch_size = 64
learning_rate = 0.0001 * (batch_size/256)

if __name__ == "__main__":

    # Load Fashion Mnist dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    input_shape = (None,) + x_train.shape[1:] + (1,)
    image_shape = input_shape[1:]
    categories = 10

    x_train = x_train.reshape(-1, *image_shape).astype("float32") / 255.0
    x_test = x_test.reshape(-1, *image_shape).astype("float32") / 255.0

    y_train = to_categorical(y_train, categories)
    y_test = to_categorical(y_test, categories)

    model = models.Sequential()
    model.add(layers.Conv2D(6, kernel_size=5, activation='relu', padding='same'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(16, kernel_size=5, activation='relu'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(220, activation='relu'))
    model.add(layers.Dense(110, activation='relu'))
    model.add(layers.Dense(categories, activation='softmax'))

    model.build(input_shape=input_shape)

    print(f"#####################################################")
    print(f"Summary")
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

    fmodel = apply_bn_folding(model, False)
    fmodel.build(input_shape=input_shape)
    fmodel.compile(
        optimizer=Adam(learning_rate=pre_training_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

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
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


    alpha_dict = compute_alpha_dict(fmodel, x_train)


    # ----------------------------
    # Print the complete alpha_dict with weights and activation alpha values.
    print("#####################################################")
    print("Alpha Dictionary (Weights and Activations)")
    for i, (layer_name, weights_dict) in enumerate(alpha_dict.items()):
        print(f"{i:04d}: {layer_name}")
        for key, alpha_value in weights_dict.items():
            print(f"    {key}: {alpha_value}")
    print("#####################################################\n")
    

    qmodel = apply_alpha_dict(qmodel, alpha_dict)
    


    print(f"#####################################################")
    print(f"POST QUANTIZATION Evaluation")
    loss, accuracy = qmodel.evaluate(x=x_test, y=y_test)
    print(f"#####################################################\n")


    callback_tuples = [(CaptureWeightCallback(qlayer), qconfig[layer.name]) for layer, qlayer in zip(model.layers, qmodel.layers) if layer.name in qconfig]

    print(f"#####################################################")
    print(f"QAT")
    if epochs > 0:
        hist = qmodel.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
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



