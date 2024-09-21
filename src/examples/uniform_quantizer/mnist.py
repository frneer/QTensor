#!/usr/bin/env python3

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow_model_optimization.quantization.keras import (
    quantize_annotate_layer,
    quantize_annotate_model,
    quantize_apply,
    quantize_scope,
)

from configs.configs import UniformQuantizeConfig


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the data to a range of 0 to 1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Convert the labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Build a simple model with both Dense layers quantized
    layer_1 = Flatten(input_shape=(28, 28), name="input")
    layer_2 = quantize_annotate_layer(
        Dense(128, activation="relu", name="hidden"),
        UniformQuantizeConfig(bits=6, alpha=1),
    )
    layer_3 = quantize_annotate_layer(
        Dense(10, activation="softmax", name="output"),
        UniformQuantizeConfig(bits=4, alpha=1),
    )
    model = quantize_annotate_model(Sequential([layer_1, layer_2, layer_3]))

    with quantize_scope({"UniformQuantizeConfig": UniformQuantizeConfig}):
        quant_aware_model = quantize_apply(model)
    quant_aware_model.summary()

    # Train the model
    quant_aware_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    class VariableHistoryCallback(tf.keras.callbacks.Callback):
        def __init__(self, variable):
            super(VariableHistoryCallback, self).__init__()
            self.variable = variable
            self.variable_values = []

        def on_epoch_end(self, epoch, logs=None):
            # Record the variable's value at the end of each epoch
            self.variable_values.append(self.variable.numpy())

        def get_history(self):
            return self.variable_values

    vars = quant_aware_model.variables
    var_names = [v.name for v in vars]
    print(*var_names)
    callbacks = [VariableHistoryCallback(v) for v in vars if "alpha" in v.name]

    hist = quant_aware_model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(x_test, y_test),
        callbacks=[callbacks],
    )

    plt.figure()
    for callback in callbacks:
        plt.plot(callback.get_history(), label=callback.variable.name)
    plt.legend([callback.variable.name for callback in callbacks])
    plt.grid(which="both")
    plt.savefig("alpha_history.png")

    # Evaluate the model
    quant_aware_model.evaluate(x_test, y_test)


if __name__ == "__main__":
    main()
