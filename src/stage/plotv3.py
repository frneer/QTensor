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
pd.set_option("display.width", None)


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


def add_horizontal_se_band(
    ax,
    mean,
    se,
    *,
    color="red",
    label="Original Model",
    alpha_line=0.8,
    alpha_band=0.2,
    z_mean=3,
    z_line=2,
    z_band=1,
):
    """On ax, draw a dotted line at `mean`, dashed lines at ±se, and fill
    between them."""
    # mean line
    ax.axhline(
        mean,
        color=color,
        linestyle=":",
        linewidth=0.3,
        alpha=alpha_line,
        zorder=z_mean,
        # label=f"{label}"
    )
    # ±1 SE lines
    ax.axhline(
        mean + se,
        color=color,
        linestyle="-",
        linewidth=0.3,
        alpha=0.6,
        zorder=z_line,
    )
    ax.axhline(
        mean - se,
        color=color,
        linestyle="-",
        linewidth=0.3,
        alpha=0.6,
        zorder=z_line,
    )
    # filled band
    x0, x1 = ax.get_xlim()
    ax.fill_between(
        [x0, x1],
        [mean - se, mean - se],
        [mean + se, mean + se],
        color=color,
        alpha=alpha_band,
        zorder=z_band,
    )


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

    blacklist = {
        "fe398b8e7dfdc1cb8c71d2183a84de07",
        "c8a4f191f560ea10931775d0a8320c08",
        "04ccc689dc66e77aa8ecd339281c5ff0",
    }
    experiments = [
        exp for exp in experiments if not any(h in blacklist for h in exp)
    ]

    print("After blacklisting, keeping", len(experiments), "experiments:")
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
    print(
        experiments_df[
            [
                "qat_accuracy",
                "qat_complexity",
                "qat_hash",
            ]
        ].sort_values(by=["qat_accuracy"], ascending=False)
    )

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
    experiments_df = experiments_df.drop(columns=cols_to_drop)

    rename_map = {
        "name": "stage",
        "quantization_parameters_kernel_1_type": "type",
        "quantization_parameters_kernel_1_bits": "bits",
        "quantization_parameters_kernel_1_n_levels": "n_levels",
        "initial_training_seed": "seed",
    }
    experiments_df = experiments_df.rename(columns=rename_map)
    experiments_df = experiments_df.dropna()

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
    original_accuracy_min = experiments_df["initial_training_accuracy"].min()
    original_complexity_min = experiments_df[
        "initial_training_complexity"
    ].min()
    original_accuracy_max = experiments_df["initial_training_accuracy"].max()
    original_complexity_max = experiments_df[
        "initial_training_complexity"
    ].max()

    # 1) Identify the metrics and the hyperparam columns to group by
    metrics = ["qat_loss", "qat_accuracy", "qat_complexity"]
    hyperparam_cols = [
        col
        for col in experiments_df.columns
        if col in ("type", "bits", "n_levels")
    ]

    # 2) Group by those hyperparams, compute mean & var of the metrics
    stats = (
        experiments_df.groupby(hyperparam_cols)[metrics]
        .agg(["mean", "var", "max", "min"])
        .reset_index()
    )

    # 3) Flatten the resulting MultiIndex columns
    stats.columns = [
        f"{lvl0}_{lvl1}" if lvl1 else lvl0 for lvl0, lvl1 in stats.columns
    ]

    counts = (
        experiments_df.groupby(hyperparam_cols)[metrics].count().reset_index()
    )
    count_cols = [f"{m}_count" for m in metrics]
    counts.columns = hyperparam_cols + count_cols
    row_uniques = counts[count_cols].nunique(axis=1)
    assert (
        row_uniques == 1
    ).all(), "❌ Not all metric‐counts agree per group!"
    counts["count"] = counts[count_cols[0]]
    counts = counts[hyperparam_cols + ["count"]]
    stats = stats.merge(counts, on=hyperparam_cols, how="left")

    # 3) Compute standard deviations and errors
    stats["se_accuracy"] = np.sqrt(stats["qat_accuracy_var"]) / np.sqrt(
        stats["count"]
    )
    stats["se_complexity"] = np.sqrt(stats["qat_complexity_var"]) / np.sqrt(
        stats["count"]
    )

    print(
        stats.sort_values(
            by=["qat_accuracy_mean", "qat_complexity_mean"], ascending=False
        )
    )
    print(stats.sort_values(by=["qat_complexity_mean"], ascending=False))

    cmap = mpl.colormaps["tab10"]
    xmin = stats["qat_complexity_mean"].min() * 0.9
    xmax = stats["qat_complexity_mean"].max() * 1.1
    xmax = original_complexity_mean * 1.1

    plt.figure(figsize=(6, 4))
    plt.scatter(
        original_complexity_mean,
        original_accuracy_mean,
        color="red",
        label="Original Model",
        zorder=3,
    )
    bits_values = experiments_df["bits"].unique()
    for i, bits in enumerate(sorted(bits_values)):
        subset = experiments_df[experiments_df["bits"] == bits].sort_values(
            "qat_complexity"
        )
        # 1) dashed line only, semi-transparent
        plt.semilogx(
            subset["qat_complexity"],
            subset["qat_accuracy"],
            linestyle="--",
            color=cmap(i % 10),
            alpha=0.3,
            zorder=2,
            label=None,  # we’ll label in the marker call
        )
        # 2) opaque markers on top, with label
        plt.scatter(
            subset["qat_complexity"],
            subset["qat_accuracy"],
            marker=".",
            color=cmap(i % 10),
            zorder=3,
            label=f"{bits} bits",
        )
    add_horizontal_se_band(
        plt.gca(),
        original_accuracy_mean,
        original_accuracy_se,
        color="red",
    )
    plt.xlabel("Quantized Complexity (Kbits)")
    plt.ylabel("Quantized Accuracy")
    plt.ylim([0.6, 0.75])
    plt.xlim([xmin, xmax])
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
    bits_list = sorted(stats["bits"].unique())
    for i, bits in enumerate(bits_list):
        grp = stats[stats["bits"] == bits].sort_values("qat_complexity_mean")
        # 1) draw only the dashed line + errorbars (no markers)
        plt.errorbar(
            grp["qat_complexity_mean"],
            grp["qat_accuracy_mean"],
            xerr=grp["se_complexity"],
            yerr=grp["se_accuracy"],
            fmt="--",  # just the line
            color=cmap(i % 10),
            ecolor=cmap(i % 10),
            alpha=0.3,
            capsize=3,
            zorder=2,
        )
        # 2) draw the opaque markers on top
        plt.scatter(
            grp["qat_complexity_mean"],
            grp["qat_accuracy_mean"],
            marker=".",
            s=30,
            color=cmap(i % 10),
            label=f"{bits} bits",
            zorder=3,
        )
    add_horizontal_se_band(
        plt.gca(),
        original_accuracy_mean,
        original_accuracy_se,
        color="red",
    )
    plt.xscale("log")
    plt.xlabel("Quantized Complexity (Kbits)")
    plt.ylabel("Quantized Accuracy")
    plt.ylim([0.6, 0.75])
    plt.xlim([xmin, xmax])
    plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("complexity_vs_quantized_flex_stats.png", dpi=150)
