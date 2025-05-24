from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

layer_1 = Flatten(input_shape=(28, 28), name="input")
layer_2 = Dense(128, activation="relu", name="hidden")
layer_3 = Dense(10, activation="softmax", name="output")
model = Sequential([layer_1, layer_2, layer_3])


from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer

simple_qconfig = {
    "hidden": {
        "weights": {"kernel": FlexQuantizer(bits=4, n_levels=10, signed=True)},
        "activations": {"activation": UniformQuantizer(bits=4, signed=False)},
    }
}

uniform_qconfig = {
    "hidden": {
        "weights": {"kernel": UniformQuantizer(bits=4, signed=True)},
        "activations": {"activation": UniformQuantizer(bits=4, signed=False)},
    }
}

qconfigs = {
    "simple": simple_qconfig,
    "uniform": uniform_qconfig,
}
