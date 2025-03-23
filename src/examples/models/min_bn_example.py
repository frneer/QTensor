import tensorflow as tf

# Create a sample model.
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(name='input'),
    tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), name="conv2d_1", use_bias=False), 
    tf.keras.layers.BatchNormalization(name="bn_1"),
    tf.keras.layers.Activation('relu', name="activation_1"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, name="dense_2"),
])

from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer

simple_folded_merged_qconfig = {
        "conv2d_1": {
            "weights": {"kernel": FlexQuantizer(bits=4, n_levels=10 , signed=True)},
            "activations": {"activation": UniformQuantizer(bits=4, signed=False)},
            },
        }

qconfigs = {
    "folded_merged_qconfig": simple_folded_merged_qconfig
}
