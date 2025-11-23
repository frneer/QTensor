from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential

categories = 10  # Number of classes (digits 0-9)
input_shape = [None, 32, 32, 3]  # Input shape for CIFAR-10 dataset
model = Sequential(
    [
        Conv2D(
            64,
            (3, 3),
            padding="same",
            activation="relu",
            input_shape=input_shape[1:],
            name="conv2d",
        ),
        MaxPooling2D((2, 2)),
        Conv2D(
            128, (3, 3), padding="same", activation="relu", name="conv2d_1"
        ),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation="relu", name="dense"),
        Dropout(0.5),
        Dense(categories, activation="softmax", name="dense_1"),
    ],
)

from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer

n_levels = 10  # Number of quantization levels
bits = 8  # Number of bits for quantization

qconfig = {
    "conv2d": {
        "weights": {
            "kernel": FlexQuantizer(bits=bits, n_levels=n_levels, signed=True),
            "bias": UniformQuantizer(bits=8, signed=True),
        },
        "activations": {"activation": UniformQuantizer(bits=16, signed=False)},
    },
    "conv2d_1": {
        "weights": {
            "kernel": FlexQuantizer(bits=bits, n_levels=n_levels, signed=True),
            "bias": UniformQuantizer(bits=8, signed=True),
        },
        "activations": {"activation": UniformQuantizer(bits=16, signed=False)},
    },
    "dense": {
        "weights": {
            "kernel": FlexQuantizer(bits=bits, n_levels=n_levels, signed=True),
            "bias": UniformQuantizer(bits=8, signed=True),
        },
        "activations": {"activation": UniformQuantizer(bits=16, signed=False)},
    },
    "dense_1": {
        "weights": {
            "kernel": FlexQuantizer(bits=bits, n_levels=n_levels, signed=True),
            "bias": UniformQuantizer(bits=8, signed=True),
        },
        "activations": {"activation": UniformQuantizer(bits=16, signed=False)},
    },
}

qconfigs = {"qconfig": qconfig}
