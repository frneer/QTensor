from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# path =  Path("checkpoints/lenet5_custom-fashion_mnist")
path = Path("checkpoints/lenet5_custom_v2-cifar10")

experiment_paths = [
    experiment_path
    for experiment_path in path.iterdir()
    if experiment_path.is_dir() and experiment_path.name != "artifacts"
]

df = pd.DataFrame()
for experiment_path in experiment_paths:
    if not experiment_path.exists():
        print(f"Metadata path '{experiment_path}' does not exist. Skipping.")
        continue
    for metadata_file in experiment_path.glob("*.json"):
        with metadata_file.open("r") as f:
            metadata = pd.read_json(f, orient="index").T
            metadata["experiment"] = experiment_path.name
            # Select only the desired columns
            selected_columns = [
                "experiment",
                "name",
                "loss",
                "accuracy",
                "complexity",
            ]
            metadata = metadata[
                [col for col in selected_columns if col in metadata.columns]
            ]
            df = pd.concat([df, metadata], ignore_index=True)

simple_df = df[df["name"].isin(["pbnf_training", "final_evaluation"])].copy()

simple_df["name"] = simple_df["name"].replace(
    {
        "pbnf_training": "Original",
        "final_evaluation": "Quantized",
    }
)

# Convert complexity from bits to Kbits
simple_df["complexity"] = simple_df["complexity"] / 1024

pivoted_df = simple_df.pivot_table(
    index="experiment", columns="name", values=["accuracy", "complexity"]
)


# The accuracy of the common points doesn't match, fix that.

df = pivoted_df.copy()
# Sort the DataFrame by this new column
df = df.sort_values(by=("complexity", "Quantized"), ascending=True)
pd.set_option("display.max_rows", None)
print(df)

# df.sort_index(inplace=True, sort_by=['complexity'])
# --- 2. Create the Plot (using tuple access) ---

# Get the data for the original model from the first row
# Note the use of tuples to access the columns
original_accuracy = df[("accuracy", "Original")].iloc[0]
original_size = df[("complexity", "Original")].iloc[0]

# Set up the plot size and style
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 7))

# --- 3. Plot Each Point ---

# Plot the single point for the Original Model
ax.scatter(
    x=original_size,
    y=original_accuracy,
    marker="*",
    s=250,
    color="red",
    label="Original Model",
    zorder=5,
)

# Plot the points for ALL of your Quantized Models
# We use tuples to get the correct columns for the X and Y axes
ax.scatter(
    x=df[("complexity", "Quantized")],
    y=df[("accuracy", "Quantized")],
    s=60,
    color="royalblue",
    label="Quantized Models",
)

# Plot the line connecting quantized models and the original model as the final item
quantized_sizes = df[("complexity", "Quantized")].tolist()
quantized_accuracies = df[("accuracy", "Quantized")].tolist()

# Append the original model as the final item
quantized_sizes.append(original_size)
quantized_accuracies.append(original_accuracy)

ax.plot(
    quantized_sizes,
    quantized_accuracies,
    color="royalblue",
    linestyle="--",
    linewidth=1,
    zorder=1,
)

# --- 4. Add Labels to make the plot readable ---

# Loop through the DataFrame index (e.g., '1_bit', '2_bit')
for experiment_name in df.index:
    ax.annotate(
        experiment_name.rstrip("_bit"),
        (
            df.loc[experiment_name, ("complexity", "Quantized")],
            df.loc[experiment_name, ("accuracy", "Quantized")],
        ),
        textcoords="offset points",
        xytext=(-5, 15),  # Shift right and down
        ha="left",
        va="top",
    )

# Add titles and labels for the axes
ax.set_title("Model Accuracy vs. Size Trade-off", fontsize=16)
ax.set_xlabel("Model Size (Complexity in KB)", fontsize=12)
ax.set_ylabel("Model Accuracy", fontsize=12)
ax.legend(fontsize=11)

plt.savefig("model_accuracy_vs_size.png", dpi=300, bbox_inches="tight")
