from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import tensorflow as tf
from functions import load_data

from configs.serialization.serialization import load_qmodel, save_qmodel
from utils.metrics import compute_space_complexity_model


@dataclass
class StageMetadata:
    name: str
    seed: int
    function: Callable
    parameters: Dict[str, Any]
    previous_hash: Optional[str] = None

    def to_hash(self) -> str:
        stage_metadata_dict = asdict(self)
        stage_metadata_dict["function"] = (
            self.function.__name__
            if not isinstance(self.function, partial)
            else self.function.func.__name__
        )
        config_str = json.dumps(stage_metadata_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def save(self, directory_path: Path):
        file_path = directory_path / f"{self.to_hash()}.json"
        stage_metadata_dict = asdict(self)
        stage_metadata_dict["function"] = (
            self.function.__name__
            if not isinstance(self.function, partial)
            else self.function.func.__name__
        )
        with open(file_path, "w") as f:
            json.dump(stage_metadata_dict, f, indent=4)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StageMetadata:
        return cls(
            name=data["name"],
            seed=data["seed"],
            function=data["function"],
            parameters=data["parameters"],
            previous_hash=data.get("previous_hash"),
        )


class Stage:
    def __init__(
        self, metadata: StageMetadata, store_path: Optional[Path] = None
    ):
        self.metadata = metadata
        self.hash = metadata.to_hash()

        self.store_path = store_path
        self.artifacts_path = store_path / "artifacts"
        self.model_path = self.artifacts_path / self.hash
        self.metadata_directory_path = store_path / "metadata"
        self.results_path = store_path / f"results/{self.hash}.json"
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.metadata_directory_path.mkdir(parents=True, exist_ok=True)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)

    def is_model_stored(self) -> bool:
        """Check if the model for this stage is already stored."""
        return self.model_path.exists()

    def load_model(self) -> tf.keras.Model:
        return load_qmodel(self.model_path)

    def save_model(self, model: tf.keras.Model) -> None:
        """Saves the model to the artifacts path."""
        save_qmodel(model, self.model_path)

    def evaluate_model(
        self, model: tf.keras.Model, dataset: str
    ) -> Tuple[float, float]:
        """Evaluates the model on the given dataset."""
        data = load_data(dataset)
        loss, accuracy = model.evaluate(
            data["x_test"], data["y_test"], verbose=0
        )
        return loss, accuracy

    def compute_complexity(self, model: tf.keras.Model) -> Dict[str, Any]:
        return compute_space_complexity_model(model)

    def generate_results(
        self, model: tf.keras.Model, dataset: Optional[str]
    ) -> Dict[str, Any]:
        """Generates results for the model, including loss, accuracy, and
        complexity."""
        results = {}
        if dataset is not None:
            loss, accuracy = self.evaluate_model(model, dataset)
            results["loss"] = loss
            results["accuracy"] = accuracy
        complexity = self.compute_complexity(model)
        results["complexity"] = complexity
        return results

    def run(
        self, input_model: Optional[tf.keras.Model] = None
    ) -> Tuple[tf.keras.Model, str]:
        if self.metadata.seed is not None:
            tf.random.set_seed(self.metadata.seed)
        self.metadata.save(self.metadata_directory_path)

        if self.is_model_stored():
            print(f"Loading model from {self.model_path}")
            model = self.load_model()
        else:
            print(
                f"Model not found at {self.model_path}, running stage function."
            )
            model = self.metadata.function(
                model=input_model, **self.metadata.parameters
            )
            self.save_model(model)

        # Save results if they don't already exist
        if not (self.results_path).exists():
            results = self.generate_results(
                model, self.metadata.parameters.get("dataset")
            )
            with (self.results_path).open("w") as f:
                json.dump(results, f, indent=4)

        return model, self.hash


class Pipeline:
    def __init__(
        self,
        name: str,
        stage_definitions: List[StageMetadata],
        store_path: Optional[Path] = None,
    ):
        self.name = name
        self.stage_definitions = stage_definitions
        self.store_path = store_path or Path("checkpoints")
        self.pipeline_metadata_path = self.store_path / "pipelines"
        self.pipeline_metadata_path.mkdir(parents=True, exist_ok=True)
        self.hash_history = []

    def run(self) -> Tuple[tf.keras.Model, str]:
        previous_stage_hash: Optional[str] = None
        current_model: Optional[tf.keras.Model] = None

        for i, stage_def in enumerate(self.stage_definitions):
            # Workaround for a particular case
            print("--- Running stage:", stage_def.name, "----")
            if stage_def.name == "alpha_initialization":
                assert (
                    ref_model is not None
                ), "Reference model for alpha initialization is not set."
                stage_def.function = partial(
                    stage_def.function, ref_model=ref_model
                )
            final_metadata = StageMetadata(
                name=stage_def.name,
                seed=stage_def.seed,
                function=stage_def.function,
                parameters=stage_def.parameters,
                previous_hash=previous_stage_hash,
            )
            current_stage = Stage(
                metadata=final_metadata, store_path=self.store_path
            )

            current_model, stage_hash = current_stage.run(
                input_model=current_model
            )
            self.hash_history.append(stage_hash)
            # Workaround for a particular case
            if stage_def.name == "initial_training":
                ref_model = (
                    current_model  # pyright: ignore[reportUnboundVariable]
                )

            previous_stage_hash = stage_hash

        with open(self.pipeline_metadata_path / f"{self.name}.json", "w") as f:
            json.dump({"history": self.hash_history}, f)
