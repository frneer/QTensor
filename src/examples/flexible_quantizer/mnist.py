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
import pandas as pd
from configs.configs import FlexibleQuantizeConfig
from typing import List


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
        FlexibleQuantizeConfig(bits=6, alpha=1.0, n_clusters=6),
    )
    # layer_2 = Dense(128, activation="relu", name="hidden")
    # layer_3 = quantize_annotate_layer(
    #     Dense(10, activation="softmax", name="output"),
    #     FlexibleQuantizeConfig(bits=8, alpha=1, n_clusters=16),
    # )
    layer_3 = Dense(10, activation="softmax", name="output")
    model = quantize_annotate_model(Sequential([layer_1, layer_2, layer_3]))

    with quantize_scope({"FlexibleQuantizeConfig": FlexibleQuantizeConfig}):
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
            self.variable_values.append(self.variable.numpy().flatten())

        def get_history(self):
            # Convert the list of values into a DataFrame
            history_df = pd.DataFrame(self.variable_values)
            history_df.columns = [self.variable.name + str(column_name) for column_name in history_df.columns]
            # Optionally, add epoch number as an index
            history_df.index.name = 'epoch'
            return history_df

    vars = quant_aware_model.variables
    var_names = [v.name for v in vars]
    # print(*var_names)
    callbacks_hidden = [VariableHistoryCallback(v) for v in vars if ("quant_hidden/kernel_cluster_centers" in v.name) or ("quant_hidden/kernel_cluster_limits" in v.name)]
    callbacks_output = [VariableHistoryCallback(v) for v in vars if ("quant_output/kernel_cluster_centers" in v.name) or ("quant_output/kernel_cluster_limits" in v.name)]
    hist = quant_aware_model.fit(
        x_train,
        y_train,
        epochs=3,
        batch_size=32,
        validation_data=(x_test, y_test),
        callbacks=callbacks_hidden + callbacks_output,
    )
    def plot_clusters(callbacks: List[VariableHistoryCallback], tag: str):
        fig, ax = plt.subplots(figsize=(20, 10))  # Create a single figure and axis
        for callback in callbacks:
            var_name = callback.variable.name
            history_df = callback.get_history()  # Get the history as a DataFrame
            linestyle="-"
            if "limits" in var_name:
                linestyle="-."

            # Plot each variable's history on the same axis
            history_df.plot(ax=ax, grid=True, title="Clusters", linestyle=linestyle)
        plt.savefig(f"cluster_history_{tag}.png")
    plot_clusters(callbacks_hidden, "hidden")
    plot_clusters(callbacks_output, "output")

    # Evaluate the model
    quant_aware_model.evaluate(x_test, y_test)


if __name__ == "__main__":
    main()
