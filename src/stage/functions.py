# functions.py


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from configs.qmodel import apply_quantization
from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer

# --- Data Loading ---


def load_data(dataset_name: str) -> dict:
    """Loads and preprocesses the specified dataset."""
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Reshape and normalize images
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

        # One-hot encode labels
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
        }
    if dataset_name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.fashion_mnist.load_data()
        )

        # Reshape and normalize images
        x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

        # One-hot encode labels
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
        }
    if dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.cifar10.load_data()
        )

        # Normalize images
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        # One-hot encode labels
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
        }
    if dataset_name == "cifar100":
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.cifar100.load_data()
        )

        # Normalize images
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        # One-hot encode labels
        y_train = to_categorical(y_train, 100)
        y_test = to_categorical(y_test, 100)

        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")


# --- Core Model Operations ---


def model_create(model, **params: dict) -> tf.keras.Model:
    """Creates a new Keras model based on the specified architecture."""
    model_name = params["model_name"]
    input_shape = params["input_shape"]
    categories = params["categories"]
    if model_name == "lenet5_custom":
        new_model = models.Sequential(
            [
                layers.Conv2D(
                    6,
                    kernel_size=5,
                    activation="relu",
                    padding="same",
                    input_shape=input_shape[1:],
                ),
                layers.AveragePooling2D(),
                layers.Conv2D(16, kernel_size=5, activation="relu"),
                layers.AveragePooling2D(),
                layers.Flatten(),
                layers.Dense(120, activation="relu"),
                layers.Dense(84, activation="relu"),
                layers.Dense(categories, activation="softmax"),
            ],
            name=model_name,
        )

        new_model.compile(
            optimizer=Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return new_model

    if model_name == "lenet5_custom_v2":
        new_model = models.Sequential(
            [
                layers.Conv2D(
                    32,
                    kernel_size=5,
                    activation="relu",
                    padding="same",
                    input_shape=input_shape[1:],
                ),
                layers.AveragePooling2D(),
                layers.Conv2D(64, kernel_size=5, activation="relu"),
                layers.AveragePooling2D(),
                layers.Conv2D(64, kernel_size=5, activation="relu"),
                layers.AveragePooling2D(),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(categories, activation="softmax"),
            ],
            name=model_name,
        )

        new_model.compile(
            optimizer=Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return new_model
    if model_name == "vgg16":
        new_model = tf.keras.applications.VGG16(
            include_top=True,
            weights=None,
            input_shape=input_shape[1:],
            classes=categories,
        )
        new_model.compile(
            optimizer=Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return new_model
    else:
        raise ValueError(f"Unknown model_name: {model_name!r}")


def model_train(model: tf.keras.Model, **params: dict) -> tf.keras.Model:
    """Trains the model with the given parameters."""
    if model is None:
        raise ValueError("model_train received an empty model.")

    data = load_data(params["dataset"])

    model.compile(
        optimizer=Adam(learning_rate=params["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    if params.get("epochs", 0) > 0:
        model.fit(
            data["x_train"],
            data["y_train"],
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            validation_split=params["validation_split"],
            verbose=1,  # Set to 1 to see progress
        )
    return model


# --- Model Transformation and Quantization ---


def apply_bn_folding(
    model: tf.keras.Model, merge_activation: bool = False
) -> tf.keras.Model:
    """Fuses Conv/Dense layers with subsequent BatchNormalization layers."""
    if not isinstance(model, tf.keras.Sequential):
        raise TypeError(
            "BN folding currently only supports Sequential models."
        )

    def is_relu(layer):
        return isinstance(layer, layers.ReLU) or (
            isinstance(layer, layers.Activation)
            and layer.get_config().get("activation") == "relu"
        )

    new_layers = []
    i = 0
    while i < len(model.layers):
        layer = model.layers[i]

        # Check if we can fold this layer with the next one
        if (
            isinstance(layer, (layers.Conv2D, layers.Dense))
            and i + 1 < len(model.layers)
            and isinstance(model.layers[i + 1], layers.BatchNormalization)
        ):

            conv_layer = layer
            bn_layer = model.layers[i + 1]

            # Get weights
            conv_weights = conv_layer.get_weights()
            bn_weights = bn_layer.get_weights()

            kernel = conv_weights[0]
            bias = (
                conv_weights[1]
                if conv_layer.use_bias
                else np.zeros(kernel.shape[-1])
            )

            gamma, beta, moving_mean, moving_variance = bn_weights
            epsilon = bn_layer.epsilon

            # Calculate new weights and biases
            scale = gamma / np.sqrt(moving_variance + epsilon)
            new_bias = beta + (bias - moving_mean) * scale

            if isinstance(conv_layer, layers.Dense):
                new_kernel = kernel * scale
            else:  # Conv2D
                new_kernel = kernel * scale.reshape((1, 1, 1, -1))

            # Create new layer configuration
            new_config = conv_layer.get_config()
            new_config["use_bias"] = True
            new_config["activation"] = conv_layer.activation

            i += 2  # Skip original conv and BN

            # Check for merging activation
            if (
                merge_activation
                and i < len(model.layers)
                and is_relu(model.layers[i])
            ):
                new_config["activation"] = "relu"
                i += 1

            # Create the new fused layer
            fused_layer = type(conv_layer).from_config(new_config)
            new_layers.append(fused_layer)

            # Build and set weights for the new layer
            fused_layer.build(conv_layer.input_shape)
            fused_layer.set_weights([new_kernel, new_bias])

        else:
            new_layers.append(layer)
            i += 1

    return models.Sequential(new_layers)


def model_transform_bnf(
    model: tf.keras.Model, **params: dict
) -> tf.keras.Model:
    """Applies Batch Normalization Folding to the model."""
    print("Function: model_transform_bnf called")
    if model is None:
        raise ValueError("model_transform_bnf received an empty model.")
    new_model = apply_bn_folding(
        model, merge_activation=params.get("merge_activation", False)
    )
    new_model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return new_model


def model_quantize(model: tf.keras.Model, **params) -> tf.keras.Model:
    """Applies quantization to the model."""
    print("Function: model_quantize called")
    if model is None:
        raise ValueError("model_quantize received an empty model.")
    # (Your quantization logic here)
    kernel = params["kernel"]
    bias = params["bias"]
    activations = params["activations"]

    # Layers initialization
    layers = list()
    supported = ("conv2d", "dense")
    for layer in model.layers:
        if any(kw in layer.name for kw in supported):
            layers.append(layer.name)

    # QConfig initialization
    qconfig = dict()
    for layer in layers:
        qconfig[layer] = dict()
    for layer in layers:
        for k in ("weights", "activations"):
            qconfig[layer][k] = dict()

    for layer, k, b, a in zip(layers, kernel, bias, activations):
        # Kernel
        if k["type"] == "uniform":
            qconfig[layer]["weights"]["kernel"] = UniformQuantizer(
                bits=k["bits"], signed=True
            )
        elif k["type"] == "flexible":
            qconfig[layer]["weights"]["kernel"] = FlexQuantizer(
                bits=k["bits"], n_levels=k["n_levels"], signed=True
            )
        else:
            pass
        # Bias
        if b["type"] == "uniform":
            qconfig[layer]["weights"]["bias"] = UniformQuantizer(
                bits=b["bits"], signed=True
            )
        elif b["type"] == "flexible":
            qconfig[layer]["weights"]["bias"] = FlexQuantizer(
                bits=b["bits"], n_levels=b["n_levels"], signed=True
            )
        else:
            pass
        # Arctivations
        if a["type"] == "uniform":
            qconfig[layer]["activations"]["activation"] = UniformQuantizer(
                bits=a["bits"], signed=False
            )
        elif a["type"] == "flexible":
            qconfig[layer]["activations"]["activation"] = FlexQuantizer(
                bits=a["bits"], n_levels=a["n_levels"], signed=False
            )
        else:
            pass
    # End logic
    quantized_model = apply_quantization(model, qconfig)
    quantized_model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return quantized_model


# --- Alpha Initialization for QAT ---


def get_activations_output(model, x_train, batch_size=128):
    """Gets the activations of the model for the training data."""
    intermediate_model = models.Model(
        inputs=model.input, outputs=[layer.output for layer in model.layers]
    )
    activations = intermediate_model.predict(
        x_train, batch_size=batch_size, verbose=0
    )
    return activations


def compute_alpha_dict(model, x_train, batch_size=128):
    """Computes alpha values for weights and activations in a single
    comprehension."""
    activations = get_activations_output(model, x_train, batch_size)

    alpha_dict = {
        layer.name: {
            **{
                weight.name: np.max(np.abs(weight.numpy()))
                for weight in layer.weights
            },
            "activation": np.max(np.abs(activation_data)),
        }
        for layer, activation_data in zip(model.layers, activations)
    }

    return alpha_dict


# def compute_flex_dict(model, x_train, batch_size=128):


def apply_alpha_dict(model, alpha_dict):
    """Applies pre-computed alpha values to a quantized model."""
    for layer in model.layers:
        original_layer_name = layer.name.replace("quant_", "")

        if original_layer_name not in alpha_dict:
            continue

        for weight in layer.weights:
            if (
                not weight.name.endswith("_alpha")
                or weight.name not in alpha_dict[original_layer_name]
            ):
                continue

            # See the quantizers weight naming convention
            # No name_suffix for now
            weight.assign(alpha_dict[original_layer_name][weight.name])
            print(
                f"Updated {weight.name} with alpha: {alpha_dict[original_layer_name][weight.name]:.4f}"
            )
    return model


def model_initialize_parameters(model, ref_model, **params) -> tf.keras.Model:
    """Initializes quantization parameters (alphas) using a reference model."""
    print("Function: model_initialize_parameters called")
    if ref_model is None:
        raise ValueError(
            "model_initialize_parameters requires a ref_model, but none was provided."
        )
    if params["type"] == "alpha":
        data = load_data(params["dataset"])
        alpha_dict = compute_alpha_dict(ref_model, data["x_train"])
        model = apply_alpha_dict(model, alpha_dict)
        return model
    raise ValueError(
        f"Unknown parameter initialization type: {params['type']!r}"
    )


def model_evaluate(model, **params):
    """Evaluates the model on the test dataset."""
    if model is None:
        raise ValueError("model_evaluate received an empty model.")
    data = load_data(params["dataset"])
    loss, accuracy = model.evaluate(data["x_test"], data["y_test"], verbose=0)
    print(f"Evaluation results - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return model


# --- Function Map ---
FUNCTION_MAP = {
    "model_create": model_create,
    "model_evaluate": model_evaluate,
    "model_train": model_train,
    "model_transform_bnf": model_transform_bnf,  # Assuming you will add this
    "model_quantize": model_quantize,
    "model_initialize_parameters": model_initialize_parameters,
}
