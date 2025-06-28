#!/usr/bin/env python3

from functions import (
    model_create,
    model_initialize_parameters,
    model_quantize,
    model_train,
)
from stagev2 import Pipeline, StageMetadata

# --- Configuration for All 7 Pipeline Stages ---
# This list defines the blueprint for our pipeline. Each dictionary
# will be used to initialize a Stage object.

stages_hyperparams = [
    # Stage 0: Model creation
    {
        "name": "model_creation",
        "seed": 12345,
        "function": model_create,
        "parameters": {
            "dataset": "cifar10",
            "input_shape": [None, 32, 32, 3],
            "categories": 10,
            "model_name": "custom_cnn1_for_cifar10",
        },
    },
    # Stage 1: Initial training
    {
        "name": "initial_training",
        "seed": 12345,
        "function": model_train,
        "parameters": {
            "dataset": "cifar10",
            "input_shape": [None, 32, 32, 3],
            "categories": 10,
            "epochs": 100,
            "batch_size": 512,
            "learning_rate": 0.001,
            "validation_split": 0.1,
        },
    },
    # Stage 2: Model quantization
    {
        "name": "quantization",
        "seed": 12345,
        "function": model_quantize,
        "parameters": {
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
                {"type": "uniform", "bits": 16},
                {"type": "uniform", "bits": 16},
                {"type": "uniform", "bits": 16},
                {"type": "uniform", "bits": 16},
            ],
        },
    },
    # Stage 5: Alpha initialization
    {
        "name": "alpha_initialization",  # Fixed typo from original "initialiation"
        "seed": 12345,
        "function": model_initialize_parameters,
        "parameters": {
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
        "function": model_train,
        "parameters": {
            "dataset": "cifar10",
            "input_shape": [None, 32, 32, 3],
            "categories": 10,
            "epochs": 25,
            "batch_size": 512,
            "learning_rate": 0.0001,
            "validation_split": 0.1,
        },
    },
]


if __name__ == "__main__":
    for bits in reversed([1, 2, 3, 4, 5, 6, 8, 10, 16, 24]):
        print(
            f"\n{'='*20} STARTING EXPERIMENT: UNIFORM BITS = {bits} {'='*20}\n"
        )

        # --- Configure the Experiment ---
        # Dynamically set the 'kernel' quantization parameter for this specific run.
        kernel_config = [{"type": "uniform", "bits": bits} for _ in range(4)]
        stages_hyperparams[2]["parameters"]["kernel"] = kernel_config
        stages_metadata = [
            StageMetadata.from_dict(stage_dict)
            for stage_dict in stages_hyperparams
        ]
        pipeline = Pipeline(
            name=f"experiment_uniform_bits_{bits}",
            stage_definitions=stages_metadata,
        )
        pipeline.run()
