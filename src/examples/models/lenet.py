from tensorflow.keras.layers import AveragePooling2D, Conv2D, Dense, Flatten
from tensorflow.keras.models import Sequential

model = Sequential(
    [
        Conv2D(
            filters=6,
            kernel_size=(5, 5),
            activation="relu",
            padding="same",
            input_shape=(28, 28, 1),
        ),
        AveragePooling2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=16, kernel_size=(5, 5), activation="relu"),
        AveragePooling2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(120, activation="relu"),
        Dense(84, activation="relu"),
        Dense(10, activation="softmax"),  # 10 classes (digits 0-9)
    ]
)

from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer

simple_qconfig = {
    "conv2d": {
        "weights": {"kernel": FlexQuantizer(bits=4, n_levels=10, signed=True)},
        "activations": {"activation": UniformQuantizer(bits=4, signed=False)},
    },
    "conv2d_1": {
        "weights": {"kernel": FlexQuantizer(bits=4, n_levels=10, signed=True)},
        "activations": {"activation": UniformQuantizer(bits=4, signed=False)},
    },
    "dense": {
        "weights": {"kernel": FlexQuantizer(bits=4, n_levels=10, signed=True)},
        "activations": {"activation": UniformQuantizer(bits=4, signed=False)},
    },
    "dense_1": {
        "weights": {"kernel": FlexQuantizer(bits=4, n_levels=10, signed=True)},
        "activations": {"activation": UniformQuantizer(bits=4, signed=False)},
    },
    "dense_2": {
        "weights": {"kernel": FlexQuantizer(bits=4, n_levels=10, signed=True)},
        "activations": {"activation": UniformQuantizer(bits=4, signed=False)},
    },
}

qconfigs = {"qconfig": simple_qconfig}
