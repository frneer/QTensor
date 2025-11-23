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
            "model_name": "lenet5_custom_v2",
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
            "epochs": 200,
            "batch_size": 1024,
            "learning_rate": 0.001,
            "validation_split": 0.1,
        },
    },
    # Stage 2: BN folding
    {
        "name": "bnf",
        "seed": 12345,
        "function": "model_transform_bnf",
        "kwargs": {
            "merge_activation": True,
        },
    },
    # Stage 3: Post BN folding training
    {
        "name": "pbnf_training",
        "seed": 12345,
        "function": "model_train",
        "kwargs": {
            "dataset": "cifar10",
            "input_shape": [None, 32, 32, 3],
            "categories": 10,
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.0005,
            "validation_split": 0.1,
        },
    },
    # Stage 4: Model quantization
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
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
            ],
            "activations": [
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
            ],
        },
    },
    # Stage 5: Alpha initialization
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
    # Stage 6: QAT
    {
        "name": "qat",
        "seed": 12345,
        "function": "model_train",
        "kwargs": {
            "dataset": "cifar10",
            "input_shape": [None, 32, 32, 3],
            "categories": 10,
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "validation_split": 0.1,
        },
    },
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

vgg_stages_hyperparams = [
    # Stage 0: Model creation
    {
        "name": "model_creation",
        "seed": 12345,
        "function": "model_create",
        "kwargs": {
            "dataset": "cifar100",
            "input_shape": [None, 32, 32, 3],
            "categories": 100,
            "model_name": "vgg16",
        },
    },
    # Stage 1: Initial training
    {
        "name": "initial_training",
        "seed": 12345,
        "function": "model_train",
        "kwargs": {
            "dataset": "cifar100",
            "input_shape": [None, 32, 32, 3],
            "categories": 100,
            "epochs": 20,
            "batch_size": 1024,
            "learning_rate": 0.001,
            "validation_split": 0.1,
        },
    },
    # Stage 2: BN folding
    {
        "name": "bnf",
        "seed": 12345,
        "function": "model_transform_bnf",
        "kwargs": {
            "merge_activation": True,
        },
    },
    # Stage 3: Post BN folding training
    {
        "name": "pbnf_training",
        "seed": 12345,
        "function": "model_train",
        "kwargs": {
            "dataset": "cifar100",
            "input_shape": [None, 32, 32, 3],
            "categories": 100,
            "epochs": 1,
            "batch_size": 128,
            "learning_rate": 0.0005,
            "validation_split": 0.1,
        },
    },
    # Stage 4: Model quantization
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
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
                {"type": None},
            ],
            "activations": [
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
                {"type": "uniform", "bits": 8},
            ],
        },
    },
    # Stage 5: Alpha initialization
    {
        "name": "alpha_initialization",  # Fixed typo from original "initialiation"
        "seed": 12345,
        "function": "model_initialize_parameters",
        "kwargs": {
            "dataset": "cifar100",
            "input_shape": [None, 32, 32, 3],
            "categories": 100,
            "type": "alpha",
        },
    },
    # Stage 6: QAT
    {
        "name": "qat",
        "seed": 12345,
        "function": "model_train",
        "kwargs": {
            "dataset": "cifar100",
            "input_shape": [None, 32, 32, 3],
            "categories": 100,
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "validation_split": 0.1,
        },
    },
    {
        "name": "final_evaluation",
        "seed": 12345,
        "function": "model_evaluate",
        "kwargs": {
            "dataset": "cifar100",
            "input_shape": [None, 32, 32, 3],
            "categories": 100,
        },
    },
]

if __name__ == "__main__":

    # This is the main experimental loop from your coworker's script.
    # It runs the entire 7-stage pipeline multiple times.
    for bits in range(1, 11):
        print(
            f"\n{'='*20} STARTING EXPERIMENT: UNIFORM BITS = {bits} {'='*20}\n"
        )

        # --- Configure the Experiment ---
        # Dynamically set the 'kernel' quantization parameter for this specific run.
        kernel_config = [{"type": "uniform", "bits": bits} for _ in range(5)]
        stages_hyperparams[4]["kwargs"]["kernel"] = kernel_config

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
            if stage.initial_config["name"] == "pbnf_training":
                ref_model = tf.keras.models.clone_model(model)

        print(
            f"\n{'='*20} FINISHED EXPERIMENT: UNIFORM BITS = {bits} {'='*20}\n"
        )
        print(
            f"Final model for {bits}-bit experiment corresponds to hash: {previous_hash}"
        )
