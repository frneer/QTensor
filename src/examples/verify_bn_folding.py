#!/usr/bin/env python3

import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

from functions import apply_bn_folding

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    input_shape = (None,) + x_train.shape[1:] + (1,)
    image_shape = input_shape[1:]
    categories = 10

    x_train = x_train.reshape(-1, *image_shape).astype("float32") / 255.0
    x_test = x_test.reshape(-1, *image_shape).astype("float32") / 255.0

    y_train = to_categorical(y_train, categories)
    y_test = to_categorical(y_test, categories)

    model = models.Sequential()
    model.add(layers.Conv2D(6, kernel_size=5, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Conv2D(16, kernel_size=5))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.AveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(220))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(110))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(categories, activation='softmax'))

    model.build(input_shape=input_shape)

    print(f"#####################################################")
    print(f"Model summary")
    model.summary(line_length=100)
    print(f"#####################################################\n")


    for merge in (True, False):
        # Create the folded model with activation merging enabled.
        folded_model = apply_bn_folding(model, merge_activation=merge)
        folded_model.build(input_shape=input_shape)

        print(f"#####################################################")
        print(f"Folded model summary (merge={merge})")
        folded_model.summary(line_length=100)
        print(f"#####################################################\n")
        
        # When comparing outputs, force inference mode:
        X = x_train[0][None,...]
        y = y_train[0][None,...]
        output_model = model(X, training=False)
        output_fmodel = folded_model(X, training=False)
        
        # You can compare using a tolerance or compare predicted classes.
        if not np.allclose(output_model, output_fmodel, atol=1e-6):
            print(f"NOT OK: Outputs differ (Merge={merge})")
        else:
            print(f"OK: Outputs are nearly identical (Merge={merge})")
        
        # For further debugging, you might compare argmax for classification.
        if not np.array_equal(np.argmax(output_model, axis=-1), np.argmax(output_fmodel, axis=-1)):
            print(f"NOT OK: Predicted classes differ (Merge={merge})")
        else:
            print(f"OK: Predicted classes are the same (Merge={merge})")
