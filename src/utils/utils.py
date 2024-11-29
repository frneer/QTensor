import tensorflow as tf


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
