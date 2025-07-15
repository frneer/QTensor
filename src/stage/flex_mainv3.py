#!/usr/bin/env python3

from functions import model_create, model_quantize, model_train
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
            "epochs": 20,
            "batch_size": 128,
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
                {"type": None},  # "uniform", "bits": 16},
            ],
        },
    },
    # Stage 5: Alpha initialization
    # {
    #     "name": "alpha_initialization",
    #     "seed": 12345,
    #     "function": initialize_quantizer_weights,
    #     "parameters": {
    #         "dataset": "cifar10",
    #         "batch_size": 512,
    #         "input_shape": [None, 32, 32, 3],
    #         "categories": 10,
    #         "type": "alpha",
    #     },
    # },
    # Stage 6: QAT
    {
        "name": "qat",
        "seed": 12345,
        "function": model_train,
        "parameters": {
            "dataset": "cifar10",
            "input_shape": [None, 32, 32, 3],
            "categories": 10,
            "epochs": 200,
            "batch_size": 16,
            "learning_rate": 0.001 / 100,
            "validation_split": 0.2,
            "early_stopping": True,
        },
    },
]


if __name__ == "__main__":
    # Check that there are no repeated names.
    names = [stage["name"] for stage in stages_hyperparams]
    cnt = Counter(names)
    dups = [name for name, c in cnt.items() if c > 1]
    assert not dups, f"Duplicate stage names detected: {dups}"

    seeds = [12345, 123456, 1234567, 12345678, 123456789]
    bits = [4, 6, 8]
    n_levels = [2, 3, 4, 6, 8, 10, 16]
    combinations = [
        (seed, b, n)
        for seed in seeds
        for b in reversed(bits)
        for n in reversed(n_levels)
        if n < 2**b
        # (b, n) for b in bits for n in n_levels if n <= 2**b
    ]
    for seed, bits, n_levels in combinations:
        print(
            f"\n{'='*20} STARTING EXPERIMENT: SEED = {seed}, FLEX BITS = {bits}, N_LEVELS = {n_levels} {'='*20}\n"
        )

        for i in range(len(stages_hyperparams)):
            stages_hyperparams[i]["seed"] = seed

        # --- Configure the Experiment ---
        # Dynamically set the 'kernel' quantization parameter for this specific run.
        # kernel_config = [
        #     {"type": "flexible", "bits": bits, "n_levels": n_levels}
        #     for _ in range(4)
        # ]
        kernel_config = [
            {"type": "flexible", "bits": 8, "n_levels": 16},
            {"type": "flexible", "bits": bits, "n_levels": n_levels},
            {"type": "flexible", "bits": bits, "n_levels": n_levels},
            {"type": "flexible", "bits": 8, "n_levels": 16},
        ]
        stages_hyperparams[2]["parameters"]["kernel"] = kernel_config
        stages_metadata = [
            StageMetadata.from_dict(stage_dict)
            for stage_dict in stages_hyperparams
        ]
        pipeline = Pipeline(
            name=f"experiment_flex_bits_{bits}_nlevels_{n_levels}",
            stage_definitions=stages_metadata,
        )
        pipeline.run()
