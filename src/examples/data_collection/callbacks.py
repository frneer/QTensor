from typing import Optional

from tensorflow.keras.callbacks import Callback


class CaptureWeightCallback(Callback):
    def __init__(self, layer, exclude: Optional[list[str]] = None):
        super(CaptureWeightCallback, self).__init__()
        self.layer = layer
        self.exclude = exclude or []

        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for weight in self.layer.weights:
            if weight.name not in self.exclude:
                if weight.name not in self.history:
                    self.history[weight.name] = []
                self.history[weight.name].append(weight.numpy())

    def get_history(self):
        return self.history
