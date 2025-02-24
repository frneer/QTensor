#!/usr/bin/env python3

import unittest

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Model, Sequential

from configs.qmodel import (
    apply_quantization,
    clone_layer,
    quantize_layer,
    quantize_model,
)
from quantizers.uniform_quantizer import UniformQuantizer


# TODO(Fran): Don't use specific layers, models or quantizers in the tests
# Create mocks for them instead
class TestQModel(unittest.TestCase):
    """Test class for the QModel class."""

    def setUp(self):
        input_tensor = Input(shape=(10,), name="input")
        output_tensor = Dense(5, name="dense_functional")(input_tensor)
        self.model = Model(
            inputs=input_tensor, outputs=output_tensor, name="functional_model"
        )
        self.quantizer = UniformQuantizer(8)
        self.layer = Dense(10, input_shape=(10,), name="layer_1")

        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(10, input_shape=(10, 1), name="lstm_1"))
        self.lstm_model.add(Dense(5, name="dense_functional"))

    def test_can_clone_layer(self):
        """Test that verifies that the layer can be cloned."""
        clone = clone_layer(self.layer)
        self.assertEqual(clone.name, self.layer.name)
        self.assertEqual(clone.get_weights(), self.layer.get_weights())

    def test_can_quantize_layer(self):
        """Test that verifies that the layer can be quantized."""
        qlayer = quantize_layer(
            self.layer,
            {
                "weights": {"kernel": self.quantizer, "bias": self.quantizer},
                "activations": {"activation": self.quantizer},
            },
        )
        self.assertEqual(qlayer.get_weights(), self.layer.get_weights())
        self.assertIn("quantize_annotate", qlayer.name)

    def test_can_quantize_model(self):
        """Test that verifies that the model can be quantized."""
        qmodel = quantize_model(
            self.model,
            {
                "dense_functional": {
                    "weights": {
                        "kernel": self.quantizer,
                        "bias": self.quantizer,
                    },
                    "activations": {"activation": self.quantizer},
                }
            },
        )
        self.assertEqual(qmodel.name, "functional_model")

    def test_can_apply_quantization(self):
        """Test that verifies that the model can be quantized."""
        qmodel = apply_quantization(
            self.model,
            {
                "dense_functional": {
                    "weights": {
                        "kernel": self.quantizer,
                        "bias": self.quantizer,
                    },
                    "activations": {"activation": self.quantizer},
                }
            },
        )
        qmodel.summary()
        self.assertEqual(qmodel.name, "functional_model")
        self.assertEqual(qmodel.layers[0].name, "input")
        self.assertIn("quantize_layer", qmodel.layers[1].name)
        self.assertIn("quant_dense_functional", qmodel.layers[2].name)

    def test_can_apply_lstm_quantization(self):
        """Test that verifies that the model can be quantized."""
        qmodel = apply_quantization(
            self.lstm_model,
            {
                "lstm_1": {
                    "weights": {
                        "cell": {
                            "kernel": self.quantizer,
                            "recurrent_kernel": self.quantizer,
                            "bias": self.quantizer,
                        }
                    },
                    "activations": {"cell": {"activation": self.quantizer}},
                }
            },
        )

    def test_can_apply_multi_layer(self):
        """Test that verifies that the model can be quantized."""
        qconfig = {
            "lstm_1": {
                "weights": {
                    "cell": {
                        "kernel": self.quantizer,
                        "recurrent_kernel": self.quantizer,
                        "bias": self.quantizer,
                    },
                },
                "activations": {
                    "cell": {"activation": self.quantizer},
                },
            },
            "dense_functional": {
                "weights": {"kernel": self.quantizer, "bias": self.quantizer},
                "activations": {"activation": self.quantizer},
            },
        }
        apply_quantization(self.lstm_model, qconfig)


if __name__ == "__main__":
    unittest.main()
