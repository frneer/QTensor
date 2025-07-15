# functions.py


import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from configs.generate_config import GenerateConfig
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
    if model_name == "custom_cnn1_for_cifar10":
        new_model = models.Sequential(
            [
                layers.Conv2D(
                    64,
                    (3, 3),
                    padding="same",
                    activation="relu",
                    input_shape=input_shape[1:],
                ),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.5),
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

    callbacks = []
    if params.get("early_stopping", False):
        callbacks.append(
            EarlyStopping(
                monitor=params.get("monitor", "val_loss"),
                patience=params.get("patience", 3),
                restore_best_weights=True,
            )
        )

    if params.get("epochs", 0) > 0:
        model.fit(
            data["x_train"],
            data["y_train"],
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            validation_split=params["validation_split"],
            verbose=1,  # Set to 1 to see progress
            callbacks=callbacks if callbacks else None,
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


# --- Quantizer Weight Initialization for QAT ---


def activation_output_generator(model, x_train, batch_size=128):
    """Generator to yield activations of the model for the training data."""
    # run an inference
    model(x_train[:1])  # Ensure the model is built
    layers_to_inspect = [
        layer for layer in model.layers if hasattr(layer, "quantize_config")
    ]
    intermediate_model = models.Model(
        inputs=model.input,
        outputs=[layer.output for layer in layers_to_inspect],
    )
    num_samples = x_train.shape[0]
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        yield intermediate_model.predict_on_batch(
            x_train[start:end],
        )


def compute_max_abs_activations(model, x_train, batch_size=128):
    """Computes the maximum absolute activations for each layer."""
    activations = activation_output_generator(model, x_train, batch_size)
    max_abs_activations = {}

    for layer_outputs in activations:
        for layer, output in zip(model.layers, layer_outputs):
            if layer.name not in max_abs_activations:
                max_abs_activations[layer.name] = np.max(np.abs(output))
            else:
                max_abs_activations[layer.name] = max(
                    max_abs_activations[layer.name], np.max(np.abs(output))
                )

    return max_abs_activations


from quantizers.common import max_value, min_value


def get_max_weight_value(weight):
    max_value = np.max(np.abs(weight))
    return max_value if max_value != 0 else 0.1


def get_uniform_levels(alpha, signed, n_levels):
    start = min_value(alpha, signed)
    end = max_value(alpha, n_levels, signed)
    return np.linspace(start, end, n_levels)


def get_uniform_thresholds(alpha, signed, n_levels):
    # Thresholds include the start and end points by design.
    start = min_value(alpha, signed)
    end = alpha
    return np.linspace(start, end, n_levels + 1)


def initialize_quantizer_weights(model, **params):
    """Initializes quantizer weights for the model."""
    data = load_data(params["dataset"])
    x_train = data["x_train"]

    model.compile(
        optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.predict(x_train[:1], verbose=0)

    batch_size = params.get("batch_size", 128)
    max_activations = compute_max_abs_activations(model, x_train, batch_size)
    print("Max absolute activations:", max_activations)
    for layer in model.layers:
        print(f"Layer: {layer.name}")
        if hasattr(layer, "quantize_config") and isinstance(
            layer.quantize_config, (GenerateConfig)
        ):
            layer_config = layer.quantize_config
            print(f"layer weights: {[w.name for w in layer.weights]}s")
            weights_dict = layer_config.weights
            activations_dict = layer_config.activations
            for weight_name, quantize_config in weights_dict.items():

                # Get layer and weight info
                print("layer name:", layer.name)
                original_layer_name = layer.name.replace("quant_", "")
                original_weight_name = f"{original_layer_name}/{weight_name}:0"
                filtered_weights = [
                    w for w in layer.weights if w.name == original_weight_name
                ]
                if not filtered_weights:
                    print(
                        f"Warning: No weight found for {original_weight_name} in layer {layer.name}"
                    )
                    continue
                weight = filtered_weights[0]

                # Compute and assign alpha
                alpha = get_max_weight_value(weight)
                alpha_weight = [
                    w
                    for w in layer.weights
                    if w.name == f"{layer.name}/{weight_name}_alpha:0"
                ]
                if not alpha_weight:
                    print(
                        f"Warning: No alpha weight found for {weight_name} in layer {layer.name}"
                    )
                    continue
                alpha_weight = alpha_weight[0]
                alpha_weight.assign(alpha)
                print(f"Alpha: {alpha_weight}")

                # If flex, we need to do the same with levels and thresholds
                if isinstance(quantize_config, FlexQuantizer):

                    # Compute and assign levels
                    levels = get_uniform_levels(
                        alpha, quantize_config.signed, quantize_config.n_levels
                    )
                    levels_weight = [
                        w
                        for w in layer.weights
                        if w.name == f"{layer.name}/{weight_name}_levels:0"
                    ]
                    if not levels_weight:
                        print(
                            f"Warning: No levels weight found for {weight_name} in layer {layer.name}"
                        )
                        continue
                    levels_weight = levels_weight[0]
                    levels_weight.assign(levels)
                    print(f"Levels: {levels_weight}")

                    # Compute and assign thresholds
                    thresholds = get_uniform_thresholds(
                        alpha, quantize_config.signed, quantize_config.n_levels
                    )
                    thresholds_weight = [
                        w
                        for w in layer.weights
                        if w.name == f"{layer.name}/{weight_name}_thresholds:0"
                    ]
                    if not thresholds_weight:
                        print(
                            f"Warning: No thresholds weight found for {weight_name} in layer {layer.name}"
                        )
                        continue
                    thresholds_weight = thresholds_weight[0]
                    thresholds_weight.assign(thresholds)
                    print(f"Thresholds: {thresholds_weight}")

            for activation_name, quantize_config in activations_dict.items():
                alpha = max_activations.get(layer.name, 0)
                alpha_weight = [
                    w
                    for w in layer.weights
                    if w.name == f"{layer.name}/post_activation_alpha:0"
                ]
                if not alpha_weight:
                    print(
                        f"Warning: No alpha weight found for activation {activation_name} in layer {layer.name}"
                    )
                    continue
                alpha_weight = alpha_weight[0]
                alpha_weight.assign(alpha)
                print(f"Activation Alpha: {alpha_weight}")
                if isinstance(quantize_config, FlexQuantizer):
                    levels = get_uniform_levels(
                        alpha, quantize_config.signed, quantize_config.n_levels
                    )
                    thresholds = get_uniform_thresholds(
                        alpha, quantize_config.signed, quantize_config.n_levels
                    )
                    levels_weight = [
                        w
                        for w in layer.weights
                        if w.name == f"{layer.name}/post_activation_levels:0"
                    ]
                    if not levels_weight:
                        print(
                            f"Warning: No levels weight found for activation {activation_name} in layer {layer.name}"
                        )
                        continue
                    levels_weight = levels_weight[0]
                    levels_weight.assign(levels)
                    # print
                    thresholds_weight = [
                        w
                        for w in layer.weights
                        if w.name
                        == f"{layer.name}/post_activation_thresholds:0"
                    ]
                    if not thresholds_weight:
                        print(
                            f"Warning: No thresholds weight found for activation {activation_name} in layer {layer.name}"
                        )
                        continue
                    thresholds_weight = thresholds_weight[0]
                    thresholds_weight.assign(thresholds)
    return model


def model_freeze(model: tf.keras.Model, **params: dict) -> tf.keras.Model:
    """Freeze subsets of weights in the model according to the four flags in
    params, and skip any weights related to optimizer state."""
    print("Function: model_freeze called")
    if model is None:
        raise ValueError("model_freeze received an empty model.")

    # read the four flags (defaults to False)
    freeze_conv_params = params.get("conv_parqameters", False)
    freeze_conv_qparams = params.get("conv_qparqameters", False)
    freeze_dense_params = params.get("dense_parqameters", False)
    freeze_dense_qparams = params.get("dense_qparqameters", False)

    # regex to detect quantization variables
    qparam_pattern = re.compile(r"(alpha|levels|thresholds)")

    for weight in model.weights:
        name = weight.name

        # **new**: skip any optimizer-related variables entirely
        if "optimizer" in name:
            print(f"  → skipping optimizer var: {name}")
            continue

        is_qparam = bool(qparam_pattern.search(name))
        is_conv = "conv2d" in name
        is_dense = "dense" in name

        # DEBUG:
        # print(f"{name}: is_qparam={is_qparam}, is_conv={is_conv}, is_dense={is_dense}")

        # decide whether this weight belongs to a group the user asked to freeze
        should_freeze = (
            (is_conv and not is_qparam and freeze_conv_params)
            or (is_conv and is_qparam and freeze_conv_qparams)
            or (is_dense and not is_qparam and freeze_dense_params)
            or (is_dense and is_qparam and freeze_dense_qparams)
        )

        if should_freeze:
            weight._trainable = False
            print(f"  → freezing: {name}")
        else:
            print(f"  → not freezing: {name}")

    return model


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
    "initialize_quantizer_weights": initialize_quantizer_weights,
    "model_freeze": model_freeze,
}
