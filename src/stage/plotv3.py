#!/usr/bin/env python3

import fnmatch
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)


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


def load_file(hash_, results_path: Path, metadata_path: Path) -> dict:
    """Lee los dos JSON correspondientes a `hash_` y devuelve un dict unificado
    con todas las claves de ambos."""
    # Cargar resultados
    with open(results_path / f"{hash_}.json", "r") as f_res:
        res = json.load(f_res)
    # Cargar metadata
    with open(metadata_path / f"{hash_}.json", "r") as f_meta:
        meta = json.load(f_meta)
    # Combinar, dándole preferencia a `res` en caso de colisión de claves
    combined = {**meta, **res}
    # Añadimos el hash como columna
    combined["hash"] = hash_
    return combined


def flatten_json(obj, parent_key: str = "", sep: str = "_"):
    """Recursively flattens dicts and lists into a single dict mapping
    flattened_key -> value.

    List items get their index injected into the key.
    """
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_json(v, new_key, sep=sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_json(v, new_key, sep=sep))
    else:
        # reached a leaf
        items[parent_key] = obj
    return items


if __name__ == "__main__":
    checkpoint_path = Path("checkpoints")
    # pipeline_path = checkpoint_path / "pipelines"
    metadata_path = checkpoint_path / "metadata"
    results_path = checkpoint_path / "results"

    # Patrón: sólo caracteres hexadecimales en el nombre del fichero
    hash_pattern = re.compile(r"^[0-9a-fA-F]+$")

    # Listado de hashes (sin la extensión .json)
    hashes = [
        p.stem
        for p in results_path.glob("*.json")
        if hash_pattern.match(p.stem)
    ]
    flat_records = []
    for h in hashes:
        with open(results_path / f"{h}.json", "r") as fres:
            res = json.load(fres)
        with open(metadata_path / f"{h}.json", "r") as fmeta:
            meta = json.load(fmeta)

        merged = {**meta, **res, "hash": h}
        flat = flatten_json(merged)
        flat_records.append(flat)
    combined_df = pd.DataFrame(flat_records)

    # 1) Build adjacency list: parent_hash → [child_hashes]
    tree = defaultdict(list)
    for _, row in combined_df.iterrows():
        h = row["hash"]
        prev = row.get("previous_hash")
        if pd.notna(prev):
            tree[prev].append(h)

    # 2) Find root nodes: those with no valid previous_hash
    all_hashes = set(combined_df["hash"])
    roots = [
        h
        for _, row in combined_df.iterrows()
        for h in [row["hash"]]
        if pd.isna(row.get("previous_hash"))
        or row["previous_hash"] not in all_hashes
    ]

    # 3) DFS to collect every root→leaf path
    experiments = []

    def _collect_paths(node, path):
        children = tree.get(node, [])
        if not children:
            experiments.append(path)
        else:
            for child in children:
                _collect_paths(child, path + [child])

    for root in roots:
        _collect_paths(root, [root])

    print("Found", len(experiments), "experiments:")
    for exp in experiments:
        print(" → ".join(exp))

    # 1) build a lookup: hash → flat dict
    rec_map = {rec["hash"]: rec for rec in flat_records}

    # 2) for each experiment, merge the stages side-by-side, prefixing with the stage name
    experiments_data = []
    for exp in experiments:
        combined = {}
        for h in exp:
            rec = rec_map[h]
            stage = rec["name"]
            # prefix every field (including hash, loss, accuracy, nested flattened keys…)
            for col, val in rec.items():
                combined[f"{stage}_{col}"] = val
        experiments_data.append(combined)

    # 3) make a DataFrame
    experiments_df = pd.DataFrame(experiments_data)

    print(experiments_df.head())

    to_drop = [
        "model_creation*",
        "freeze*",
        "*function",
        "*activations_*",
        "*bias*",
        "*input_shape*",
        "*_0_*",
        "*_2_*",
        "*_3_*",
        "*_name",
        "*_hash",
        "model_creation_seed",
        "quantization_seed",
        "qat_seed",
        "*dataset",
        "*_categories",
        "*_epochs",
        "*_batch_size",
        "*_learning_rate",
        "*_validation_split",
        "quantization_complexity",
        "qat_parameters_early_stopping",
    ]
    # find all columns matching any pattern
    cols_to_drop = [
        col
        for col in experiments_df.columns
        if any(fnmatch.fnmatch(col, pat) for pat in to_drop)
    ]
    # print(cols_to_drop)
    experiments_df = experiments_df.drop(columns=cols_to_drop)

    rename_map = {
        "name": "stage",
        "quantization_parameters_kernel_1_type": "type",
        "quantization_parameters_kernel_1_bits": "bits",
        "quantization_parameters_kernel_1_n_levels": "n_levels",
        "initial_training_seed": "seed",
    }
    experiments_df = experiments_df.rename(columns=rename_map)
    print(experiments_df)
    # for k in experiments_df.columns:
    #     print(k)
    # print(experiments_df.head())
    experiments_df = experiments_df.dropna()
    print(experiments_df)

    experiments_df["qat_complexity"] = (
        experiments_df["qat_complexity"] / 1024
    )  # Convert to Kbits
    # print(
    #     experiments_df.sort_values(by=["qat_accuracy_mean", "qat_complexity_mean"], ascending=False)
    # )
    # print(experiments_df.sort_values(by=["qat_complexity_mean"], ascending=False))

    original_accuracy_mean = experiments_df["initial_training_accuracy"].mean()
    original_complexity_mean = (
        experiments_df["initial_training_complexity"].mean() / 1024
    )  # in Kbits
    original_accuracy_var = experiments_df["initial_training_accuracy"].var()
    original_complexity_var = experiments_df[
        "initial_training_complexity"
    ].var()
    original_accuracy_sd = np.sqrt(original_accuracy_var)
    original_complexity_var_kbits = original_complexity_var / (1024**2)
    original_complexity_sd = np.sqrt(original_complexity_var_kbits)
    n = experiments_df["initial_training_accuracy"].count()
    original_accuracy_se = original_accuracy_sd / np.sqrt(n)
    original_complexity_se = original_complexity_sd / np.sqrt(n)

    # 1) Identify the metrics and the hyperparam columns to group by
    metrics = ["qat_loss", "qat_accuracy", "qat_complexity"]
    hyperparam_cols = [
        col
        for col in experiments_df.columns
        if col in ("type", "bits", "n_levels")
    ]
    print("Hyperparameter columns:", hyperparam_cols)
    print("All columns in df:", experiments_df.columns.tolist())

    # 2) Group by those hyperparams, compute mean & var of the metrics
    stats = (
        experiments_df.groupby(hyperparam_cols)[metrics]
        .agg(["mean", "var"])
        .reset_index()
    )

    # 3) Flatten the resulting MultiIndex columns
    stats.columns = [
        f"{lvl0}_{lvl1}" if lvl1 else lvl0 for lvl0, lvl1 in stats.columns
    ]
    counts = (
        experiments_df.groupby(hyperparam_cols)[metrics].count().reset_index()
    )
    # rename columns back to match stats
    counts.columns = hyperparam_cols + [f"{m}_count" for m in metrics]

    # 2) Merge counts into stats
    stats = stats.merge(counts, on=hyperparam_cols)

    # 3) Compute standard deviations and errors
    stats["se_accuracy"] = np.sqrt(stats["qat_accuracy_var"]) / np.sqrt(
        stats["qat_accuracy_count"]
    )
    stats["se_complexity"] = np.sqrt(stats["qat_complexity_var"]) / np.sqrt(
        stats["qat_complexity_count"]
    )
    stats["sd_accuracy"] = np.sqrt(stats["qat_accuracy_var"])
    stats["sd_complexity"] = np.sqrt(stats["qat_complexity_var"])

    print(
        stats.sort_values(
            by=["qat_accuracy_mean", "qat_complexity_mean"], ascending=False
        )
    )
    print(stats.sort_values(by=["qat_complexity_mean"], ascending=False))

    cmap = mpl.colormaps["tab10"]

    plt.figure(figsize=(6, 4))
    plt.scatter(
        original_complexity_mean,
        original_accuracy_mean,
        color="red",
        label="Original Model",
        zorder=3,
    )
    plt.axhline(
        original_accuracy_mean, color="red", linestyle=":", alpha=0.3, zorder=1
    )
    bits_values = experiments_df["bits"].unique()
    for i, bits in enumerate(sorted(bits_values)):
        subset = experiments_df[experiments_df["bits"] == bits].sort_values(
            "qat_complexity"
        )
        plt.semilogx(
            subset["qat_complexity"],
            subset["qat_accuracy"],
            label=f"{bits} bits",
            color=cmap(i % 10),
            zorder=3,
            marker=".",
            linestyle="--",
        )
    plt.xlabel("Quantized Complexity (Kbits)")
    plt.ylabel("Quantized Accuracy")
    plt.ylim([0.6, 0.75])
    plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("complexity_vs_quantized_flex.png", dpi=150)

    # 2) sort bits for consistent coloring
    plt.figure(figsize=(6, 4))
    plt.scatter(
        original_complexity_mean,
        original_accuracy_mean,
        color="red",
        label="Original Model",
        zorder=3,
    )
    plt.axhline(
        original_accuracy_mean, color="red", linestyle=":", alpha=0.3, zorder=1
    )
    bits_list = sorted(stats["bits"].unique())
    for i, bits in enumerate(bits_list):
        grp = stats[stats["bits"] == bits].sort_values("qat_complexity_mean")

        plt.errorbar(
            grp["qat_complexity_mean"],
            grp["qat_accuracy_mean"],
            xerr=grp["se_complexity"],
            yerr=grp["se_accuracy"],
            label=f"{bits} bits",
            color=cmap(i % 10),
            zorder=3,
            marker=".",
            linestyle="--",
        )
    plt.xscale("log")
    plt.xlabel("Quantized Complexity (Kbits)")
    plt.ylabel("Quantized Accuracy")
    plt.ylim([0.6, 0.75])
    plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("complexity_vs_quantized_flex_stats.png", dpi=150)
