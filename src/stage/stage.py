# stage

from __future__ import annotations

import functools
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import tensorflow as tf

# This shouldnt be here.
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from configs.serialization.serialization import load_qmodel, save_qmodel
from utils.metrics import compute_space_complexity_model


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
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")


@dataclass(frozen=True)
class StageConfig:
    """Holds all parameters that uniquely define a stage's output.

    This entire object is hashed to create a unique ID for the stage's result.
    """

    name: str
    function: str
    seed: int
    parameters: Dict[str, Any]
    previous_hash: Optional[str] = None

    def to_hash(self) -> str:
        """Generates a unique hash for this configuration."""
        # asdict converts the dataclass to a dictionary.
        # sort_keys ensures the hash is consistent.
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()


class Stage:
    def __init__(
        self,
        function: Callable,
        initial_config: Dict[str, Any],  # We'll start with a dict
    ):
        self.function = function
        self.initial_config = initial_config
        self.config: StageConfig = None  # Will be set at runtime
        self.hash: str = None
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.is_quantized = initial_config.get("is_quantized", False)

    def _save_metadata(self):
        """Saves the current stage configuration to a JSON file.

        This is useful for debugging and traceability.
        """
        if self.config is None:
            raise ValueError("StageConfig is not set. Run the stage first.")

        metadata_path = self.checkpoint_dir / f"{self.hash}.json"
        with metadata_path.open("w") as f:
            json.dump(asdict(self.config), f, indent=2)
        print(f"Configuration saved to '{metadata_path}'")

    def _save_model(self, model: tf.keras.Model):
        """Saves the model to a file using the unique hash as the filename.

        This is useful for traceability and caching.
        """
        if self.hash is None:
            raise ValueError("Hash is not set. Run the stage first.")

        # if self.is_quantized:
        #     model_path = self.checkpoint_dir / f"{self.hash}"
        #     save_qmodel(model, model_path)
        #     print(f"Quantized model saved to '{model_path}'")
        # else:
        #     model_path = self.checkpoint_dir / f"{self.hash}.keras"
        #     model.save(model_path)
        #     print(f"Model saved to '{model_path}'")
        model_path = self.checkpoint_dir / f"{self.hash}"
        save_qmodel(model, model_path)
        print(f"Model saved to '{model_path}'")

    def save(self, model: tf.keras.Model):
        """Saves the model and its configuration to disk.

        This is useful for traceability and caching.
        """
        if self.config is None:
            raise ValueError("StageConfig is not set. Run the stage first.")

        self._save_metadata()
        self._save_model(model)

    def run(
        self,
        input_model: Optional[tf.keras.Model],
        previous_hash: Optional[str] = None,
    ) -> Tuple[tf.keras.Model, str]:
        """Runs the stage with full traceability and caching.

        Returns the resulting model AND its unique hash.
        """
        start_time = time.time()

        # 1. Create the final, traceable config for this run
        self.config = StageConfig(
            name=self.initial_config["name"],
            seed=self.initial_config.get("seed", int(time.time())),
            function=(
                self.function.__name__
                if not isinstance(self.function, functools.partial)
                else self.function.func.__name__
            ),
            parameters=self.initial_config["kwargs"],
            previous_hash=previous_hash,
        )

        # 2. Generate the unique hash for this specific configuration
        self.hash = self.config.to_hash()

        print(f"--- Running Stage({self.config.name}) ---")
        print(f"    Hash: {self.hash}")
        print(f"    Depends on: {self.config.previous_hash}")

        model_path = self.checkpoint_dir / f"{self.hash}"
        # 3. Checkpoint logic: If a model with this exact history exists, load it.
        if model_path.exists():
            print(f"    Checkpoint FOUND. Loading model from '{model_path}'")
            output_model = load_qmodel(model_path)
        else:
            print("    Checkpoint NOT FOUND. Executing function...")
            output_model = self.function(
                model=input_model, **self.config.parameters
            )

            self.save(output_model)

        print(f"--- Stage finished in {time.time() - start_time:.2f}s ---\n")

        # 5. Return both the model and its hash to the orchestrator
        return output_model, self.hash

    def evaluate(self, model):
        if model is None:
            raise ValueError("No model to evaluate. Run the stage first.")
        # After loading it is not compiled I think....
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        if "dataset" in self.config.parameters.keys():
            data = load_data(self.config.parameters["dataset"])
            loss, accuracy = model.evaluate(
                data["x_test"], data["y_test"], verbose=0
            )
            print("Evaluation results:")
            print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    def compute_complexity(self, model):
        complexity = compute_space_complexity_model(model)
        print("Space complexity of the model:")
        print(complexity)
