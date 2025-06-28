#!/usr/bin/env python3

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_pipeline(pipeline_metadata_file, results_path, metadata_path):
    df = pd.DataFrame()
    stages_results = []
    stages_names = []
    with open(pipeline_metadata_file, "r") as f:
        pipeline_metadata = json.load(f)
        for stage_hash in pipeline_metadata["history"]:
            with open(results_path / f"{stage_hash}.json", "r") as stage_file:
                stage_results = json.load(stage_file)
                stages_results.append(stage_results)
            with open(
                metadata_path / f"{stage_hash}.json", "r"
            ) as metadata_file:
                stage_metadata = json.load(metadata_file)
                stages_names.append(stage_metadata["name"])
    df = pd.DataFrame(stages_results)
    df["stage"] = stages_names
    return df


if __name__ == "__main__":
    checkpoint_path = Path("checkpoints")
    pipeline_path = checkpoint_path / "pipelines"
    metadata_path = checkpoint_path / "metadata"
    results_path = checkpoint_path / "results"

    combined_df = pd.DataFrame()

    for pipeline_metadata_file in pipeline_path.glob("*.json"):
        print(f"Loading pipeline metadata from {pipeline_metadata_file}")
        df = load_pipeline(pipeline_metadata_file, results_path, metadata_path)
        # expermient_[TYPE]_bits_[NBITS]_n_levels_[NLEVELS]
        pipeline_name = pipeline_metadata_file.stem
        pipeline_name_parts = pipeline_name.split("_")
        pipeline_quantizer_type = pipeline_name_parts[1]
        pipeline_bits = pipeline_name_parts[3]
        pipeline_levels = (
            pipeline_name_parts[5] if len(pipeline_name_parts) > 5 else None
        )
        df["pipeline_name"] = pipeline_name
        df["quantizer_type"] = pipeline_quantizer_type
        df["bits"] = pipeline_bits
        df["levels"] = pipeline_levels
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    combined_df["complexity"] = (
        combined_df["complexity"] / 1024
    )  # Convert to Kbits
    original_accuracy = combined_df[
        combined_df["stage"] == "initial_training"
    ]["accuracy"].mean()
    original_complexity = combined_df[
        combined_df["stage"] == "initial_training"
    ]["complexity"].mean()
    combined_df = combined_df[combined_df["stage"] == "qat"]
    combined_df.sort_values(by=["complexity"], inplace=True)
    all_accuracies = combined_df["accuracy"].tolist()
    all_complexities = combined_df["complexity"].tolist()
    all_accuracies.append(original_accuracy)
    all_complexities.append(original_complexity)
    print(combined_df)
    plt.figure()
    combined_df.plot(
        x="complexity",
        y=["accuracy"],
        kind="scatter",
        title="Complexity vs Accuracy",
        xlabel="Complexity (Kbits)",
        ylabel="Accuracy",
        zorder=3,
        color="blue",
        label="Quantized Model",
    )
    plt.scatter(
        original_complexity,
        original_accuracy,
        color="red",
        label="Original Model",
        zorder=3,
    )
    plt.axhline(
        original_accuracy, color="red", linestyle=":", alpha=0.3, zorder=1
    )
    plt.plot(
        all_complexities,
        all_accuracies,
        color="gray",
        linestyle="--",
        zorder=1,
    )
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig("complexity_vs_quantized.png")
