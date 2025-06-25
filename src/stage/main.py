#!/usr/bin/env python3

from functools import partial
from pathlib import Path

import tensorflow as tf
from functions import FUNCTION_MAP

from stage import Stage

# --- Configuration for All 7 Pipeline Stages ---
# This list defines the blueprint for our pipeline. Each dictionary
# will be used to initialize a Stage object.

stages_hyperparams = [
    # Stage 0: Model creation
    {
        "name": "model_creation",
        "seed": 12345,
        "function": "model_create",
        "kwargs": {
            "dataset": "cifar10",
            "input_shape": [None, 32, 32, 3],
            "categories": 10,
            "model_name": "colo_custom_cnn1_for_cifar10",
        },
    },
    # Stage 1: Initial training
    {
        "name": "initial_training",
        "seed": 12345,
        "function": "model_train",
        "kwargs": {
            "dataset": "cifar10",
            "input_shape": [None, 32, 32, 3],
            "categories": 10,
            "epochs": 100,
            "batch_size": 512,
            "learning_rate": 0.0005,
            "validation_split": 0.1,
        },
    },
    # Stage 2: Model quantization
    {
        "name": "quantization",
        "seed": 12345,
        "function": "model_quantize",
        "kwargs": {
            "input_shape": [None, 32, 32, 3],
            # 'kernel' is set to None because it will be dynamically
            # updated inside the experimental loop below.
            "kernel": None,
            "bias": [
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
            ],
            "activations": [
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
            ],
        },
    },
    # Stage 3: Alpha initialization
    {
        "name": "alpha_initialization",  # Fixed typo from original "initialiation"
        "seed": 12345,
        "function": "model_initialize_parameters",
        "kwargs": {
            "dataset": "cifar10",
            "input_shape": [None, 32, 32, 3],
            "categories": 10,
            "type": "alpha",
        },
    },
    # Stage 4: QAT
    {
        "name": "qat",
        "seed": 12345,
        "function": "model_train",
        "kwargs": {
            "dataset": "cifar10",
            "input_shape": [None, 32, 32, 3],
            "categories": 10,
            "epochs": 25,
            "batch_size": 512,
            "learning_rate": 0.0001,
            "validation_split": 0.1,
        },
    },
    # Stage 5: Model quantization
    {
        "name": "quantization",
        "seed": 12345,
        "function": "model_quantize",
        "kwargs": {
            "input_shape": [None, 32, 32, 3],
            "activations": [
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
            ],
        },
    },
    # Stage 6: Final Evaluation
    {
        "name": "final_evaluation",
        "seed": 12345,
        "function": "model_evaluate",
        "kwargs": {
            "dataset": "cifar10",
            "input_shape": [None, 32, 32, 3],
            "categories": 10,
        },
    },
]

if __name__ == "__main__":

    # This is the main experimental loop from your coworker's script.
    # It runs the entire 7-stage pipeline multiple times.
    # for bits in range(1, 25):
    for bits in [1, 2, 3, 4, 5, 6, 8, 10, 16, 24]:
        print(
            f"\n{'='*20} STARTING EXPERIMENT: UNIFORM BITS = {bits} {'='*20}\n"
        )

        # --- Configure the Experiment ---
        # Dynamically set the 'kernel' quantization parameter for this specific run.
        kernel_config = [{"type": "uniform", "bits": bits} for _ in range(5)]
        stages_hyperparams[2]["kwargs"]["kernel"] = kernel_config

        # Create the list of Stage objects from the (now updated) configurations
        dataset = stages_hyperparams[0]["kwargs"].get("dataset")
        model_name = stages_hyperparams[0]["kwargs"].get("model_name")
        pipeline = [
            Stage(
                function=FUNCTION_MAP[config["function"]],
                initial_config=config,
                checkpoint_path=Path("checkpoints")
                / f"{model_name}-{dataset}",
                metadata_path=Path(f"{bits}_bit"),
            )
            for config in stages_hyperparams
        ]

        # --- The Orchestrator ---
        # It tracks both the model object and the hash of the last operation
        model: tf.keras.Model | None = None
        previous_hash: str | None = None

        # The loop's responsibility is to pass the state (model & hash) between stages
        for stage in pipeline:
            # We need to set the ref model
            if stage.initial_config["name"] == "alpha_initialization":
                assert (
                    ref_model is not None
                ), "Reference model for alpha initialization is not set."
                stage.function = partial(stage.function, ref_model=ref_model)
            model, previous_hash = stage.run(
                input_model=model, previous_hash=previous_hash
            )

            # Save the ref model after the last stage we dont quantize
            if stage.initial_config["name"] == "initial_training":
                ref_model = tf.keras.models.clone_model(model)

        print(
            f"\n{'='*20} FINISHED EXPERIMENT: UNIFORM BITS = {bits} {'='*20}\n"
        )
        print(
            f"Final model for {bits}-bit experiment corresponds to hash: {previous_hash}"
        )
