from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Optional

import tensorflow as tf


class Stage:
    def __init__(
        self,
        id: int,
        name: str,
        function: Callable,  # Function to run for this stage
        function_kwargs: dict,  # Arguments for the function
        previous_stage: Optional[Stage] = None,
    ):
        self.id = id
        self.name = name
        self.previous_stage = previous_stage
        self.function = function
        self.function_kwargs = function_kwargs
        self.model_path = Path(
            f"output/{name.lower().replace(' ', '_')}_stage_{id}"
        )

        # Ensure the model path exists
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def to_pickle(self, file: Path) -> bytes:
        """Serialize the stage to a pickle byte string."""
        with file.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def from_pickle(file: Path):
        """Deserialize the stage from a pickle byte string."""
        with file.open("rb") as f:
            return pickle.load(f)

    def save_model(self, model: tf.keras.Model):
        """Save the model to a specified output path."""
        model.save(model.path / "model.h5", save_format="h5")
        print(f"Model saved to {model.path / 'model.h5'}")

    def load_model(self) -> tf.keras.Model:
        """Load the model from a specified output path."""
        if self.model_path.exists():
            model = tf.keras.models.load_model(self.model_path / "model.h5")
            print(f"Model loaded from {self.model_path}")
            return model

    def run(self, force_run: bool = False):
        """Run the stage function and save the model."""
        print(f"Running stage {self.id}: {self.name}")

        if self.model_path.exists() and not force_run:
            print(f"Loading model from {self.model_path}")
            model = self.load_model()
        else:
            # Run the function associated with the stage
            model = self.function(model=model, **self.function_kwargs)

            # Save the model to the specified path
            self.save_model(model)
            print(f"Model saved to {self.model_path}")

            # Save the stage to a pickle file
            self.to_pickle(self.model_path.with_suffix(".pkl"))
            print(
                f"Stage {self.id} saved to {self.model_path.with_suffix('.pkl')}"
            )
