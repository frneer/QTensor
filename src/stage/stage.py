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
from functions import load_data

from configs.serialization.serialization import load_qmodel, save_qmodel
from utils.metrics import compute_space_complexity_model

# This shouldnt be here.


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
        checkpoint_path: Optional[Path] = None,
        metadata_path: str = "metadata",
    ):
        self.function = function
        self.initial_config = initial_config
        self.config: StageConfig = None  # Will be set at runtime
        self.hash: str = None  # Will be set at runtime
        self.loss = None  # The loss after running the stage
        self.accuracy = None  # The accuracy after running the stage
        self.complexity = None  # The complexity after running the stage
        checkpoint_path = checkpoint_path or Path("checkpoints")
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.artifacts_path = checkpoint_path / "artifacts"
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.config_path = checkpoint_path / metadata_path
        self.config_path.mkdir(parents=True, exist_ok=True)
        self.model = None  # The model after running the stage

    def _save_metadata(self):
        """Saves the current stage configuration to a JSON file.

        This is useful for debugging and traceability.
        """
        if self.config is None:
            raise ValueError("StageConfig is not set. Run the stage first.")

        metadata_path = self.config_path / f"{self.config.name}.json"
        config_dict = asdict(self.config)
        config_dict["accuracy"] = self.accuracy
        config_dict["loss"] = self.loss
        config_dict["complexity"] = self.complexity
        config_dict["hash"] = self.hash
        with metadata_path.open("w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to '{metadata_path}'")

    def _save_model(self):
        """Saves the model to a file using the unique hash as the filename.

        This is useful for traceability and caching.
        """
        if self.hash is None:
            raise ValueError("Hash is not set. Run the stage first.")

        model_path = self.artifacts_path / f"{self.hash}"
        save_qmodel(self.model, model_path)
        print(f"Model saved to '{model_path}'")

    def save(self):
        """Saves the model and its configuration to disk.

        This is useful for traceability and caching.
        """
        if self.config is None:
            raise ValueError("StageConfig is not set. Run the stage first.")

        self._save_metadata()
        self._save_model()

    def load(self, hash: str):
        """Loads the model and its configuration from disk."""
        model_path = self.artifacts_path / hash
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = load_qmodel(model_path)

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

        try:
            self.load(self.hash)
        except FileNotFoundError as e:
            print(f"    Checkpoint NOT FOUND. {e} Executing function...")
            self.model = self.function(
                model=input_model, **self.config.parameters
            )
            self._save_model()
        # Evaluate the model if a dataset is provided in the parameters
        dataset = self.config.parameters.get("dataset", None)
        if dataset is not None:
            self.loss, self.accuracy = self.evaluate(load_data(dataset))
        # Compute the complexity of the model
        self.complexity = self.compute_complexity()

        self._save_metadata()

        print(f"--- Stage finished in {time.time() - start_time:.2f}s ---\n")

        # 5. Return both the model and its hash to the orchestrator
        return self.model, self.hash

    def evaluate(self, data):
        # After loading it is not compiled I think...
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        loss, accuracy = self.model.evaluate(
            data["x_test"], data["y_test"], verbose=0
        )
        print("Evaluation results:")
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, accuracy

    def compute_complexity(self):
        complexity = compute_space_complexity_model(self.model)
        print("Space complexity of the model:")
        print(complexity)
        return complexity


class Pipeline:
    def __init__(self, stages: list[Stage]):
        self.stages = stages

    def add(self, stage: Stage):
        """Adds a new stage to the pipeline."""
        self.stages.append(stage)

    def remove(self, stages_names: list[str] | str):
        """Removes stages by their names."""
        if isinstance(stages_names, str):
            stages_names = [stages_names]
        self.stages = [
            stage
            for stage in self.stages
            if stage.config.name not in stages_names
        ]

    def run(self, input_model: Optional[tf.keras.Model] = None):
        """Runs the entire pipeline, passing the model from one stage to the
        next."""
        previous_hash = None

        for stage in self.stages:
            current_model, previous_hash = stage.run(
                input_model=input_model, previous_hash=previous_hash
            )

        return current_model
