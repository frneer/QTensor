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
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


import tensorflow as tf
import numpy as np
from pathlib import Path

from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer

from batchnorm_folding import apply_bn_folding



qconfig_uniform = {
        "conv2d"              : { "weights": {"kernel": UniformQuantizer(bits=8 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        #"average_pooling2d"   : { "weights": {"kernel": UniformQuantizer(bits=8 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "conv2d_1"            : { "weights": {"kernel": UniformQuantizer(bits=8 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        #"average_pooling2d_1" : { "weights": {"kernel": UniformQuantizer(bits=8 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        #"flatten"             : { "weights": {"kernel": UniformQuantizer(bits=8 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "dense"               : { "weights": {"kernel": UniformQuantizer(bits=8 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "dense_1"             : { "weights": {"kernel": UniformQuantizer(bits=8 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        #"dense_2"             : { "weights": {"kernel": UniformQuantizer(bits=8 , signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        }
qconfig = qconfig_uniform

output_path = Path("snapshots")
pre_training_learning_rate = 0.001
pre_training_epochs = 1
pre_training_batch_size = 128
learning_rate = 0.001
epochs = 1
batch_size = 128

if __name__ == "__main__":


    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    input_shape=(28, 28, 1)

    model = models.Sequential()
    model.add(layers.Conv2D(6, kernel_size=5, activation='tanh', input_shape=input_shape, padding='same'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(16, kernel_size=5, activation='tanh'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='tanh'))
    model.add(layers.Dense(84, activation='tanh'))
    model.add(layers.Dense(10, activation='softmax'))

    model.summary(line_length=100)

    model.compile(
        optimizer=Adam(learning_rate=pre_training_learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=pre_training_epochs,
              validation_split=0.1
              )

    print(f"#####################################################")
    loss, accuracy = model.evaluate(x=x_test, y=y_test)
    print(f"POST FIT Evaluation: loss={loss}, accuracy={accuracy}")
    print(f"#####################################################\n")

    fmodel = apply_bn_folding(model, False)
    fmodel.build(input_shape=input_shape)
    fmodel.compile(
        optimizer=Adam(learning_rate=pre_training_learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    #fmodel.fit(
    #    train_dataset,
    #    epochs=args.pre_training_epochs,
    #    #batch_size=args.pre_training_batch_size,
    #    validation_data=validation_dataset
    #)

    # Code to define the initial alpha values for weights
    for i,l in enumerate(fmodel.layers):
        for j,v in enumerate(l.weights):
            print(f"{i:04d}: (max,min)=({np.max(v)},{np.min(v)}) {j+1}/{len(l.weights)} ({v.shape}) ({l.name})")


    ## Code to define the initial alpha values for activation
    ## After training, create a new model that outputs the activations of all layers.
    #min_val = +np.inf * np.ones(len(fmodel.layers))
    #max_val = -np.inf * np.ones(len(fmodel.layers))
    #for j, input_ in enumerate(x_train):
    #    print(f"{j:04d}/{len(x_train)}")
    #    input_ = input_[None,...]
    #    for i,layer in enumerate(fmodel.layers):
    #        output = layer(input_, training=False)

    #        min_aux = np.min(output)
    #        if min_aux < min_val[i]:
    #            min_val[i] = min_aux

    #        max_aux = np.max(output)
    #        if max_aux < max_val[i]:
    #            max_val[i] = max_aux

    #        input_ = output

    #for i, (min_, max_, layer) in enumerate(zip(min_val, max_val, fmodel.layers)):
    #    print(f"{i:04d}: (max,min)=(min_, max_) ({layer.name})")


    print(f"#####################################################")
    loss, accuracy = fmodel.evaluate(x=x_test, y=y_test)
    print(f"POST BN FOLDING Evaluation: loss={loss}, accuracy={accuracy}")
    print(f"#####################################################\n")

    # Apply quantization
    qmodel = apply_quantization(fmodel, qconfig)
    qmodel.build(input_shape=input_shape)
    qmodel.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"#####################################################")
    loss, accuracy = qmodel.evaluate(x=x_test, y=y_test)
    print(f"POST QUANTIZATION Evaluation: loss={loss}, accuracy={accuracy}")
    print(f"#####################################################\n")


    callback_tuples = [(CaptureWeightCallback(qlayer), qconfig[layer.name]) for layer, qlayer in zip(model.layers, qmodel.layers) if layer.name in qconfig]

    print(f"#####################################################")
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
    loss, accuracy = qmodel.evaluate(x=x_test, y=y_test)
    print(f"POST QAT Evaluation: loss={loss}, accuracy={accuracy}")
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

    #qmodel.evaluate(test_dataset)


