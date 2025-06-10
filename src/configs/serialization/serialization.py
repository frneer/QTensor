from tensorflow.keras.models import load_model
from tensorflow_model_optimization.quantization.keras import quantize_scope

from configs.qmodel import custom_objects


def save_qmodel(model, model_dir):
    """Save the quantized model to the specified directory.

    Args:
        model: The quantized Keras model to save.
        model_dir: Directory where the model will be saved.
    """
    model.save(model_dir, save_format="tf")


def load_qmodel(model_dir):
    """Load a quantized Keras model from the specified directory.

    Args:
        model_dir: Directory from which to load the model.

    Returns:
        The loaded Keras model.
    """
    with quantize_scope(custom_objects):
        return load_model(model_dir, custom_objects=custom_objects)
