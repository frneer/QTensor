import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from quantizers.common import delta, min_value, quantize


class VariableHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, variable):
        super(VariableHistoryCallback, self).__init__()
        self.variable = variable
        self.variable_values = []
        self.variable_values_before_epoch = []

    def on_epoch_begin(self, epoch, logs=None):
        self.variable_values_before_epoch.append(self.variable.numpy())

    def on_epoch_end(self, epoch, logs=None):
        # Record the variable's value at the end of each epoch
        self.variable_values.append(self.variable.numpy())

    def get_history(self):
        return self.variable_values

    def get_pre_epoch_history(self):
        return self.variable_values_before_epoch


def plot_uniform_snapshot(
    alpha_hist, accuracy_hist, bits, signed, output_path
):
    m_levels = 2**bits

    def generate_levels_from_alpha(alpha, m_levels, signed):
        print(
            f"Generating levels for alpha={alpha}, m_levels={m_levels}, signed={signed}"
        )
        delta_v = delta(alpha, m_levels, signed)
        print(delta_v)
        min_val = min_value(alpha, signed)
        return np.array([min_val + i * delta_v for i in range(m_levels)])

    level_hist = []
    for alpha in alpha_hist:
        levels = generate_levels_from_alpha(alpha, m_levels, signed)
        level_hist.append(levels)
    for epoch, (alpha, level, acc) in enumerate(
        zip(alpha_hist, level_hist, accuracy_hist)
    ):
        threshold = level
        level = np.append(level, level[-1])
        threshold = np.append(threshold, alpha)
        plt.figure(figsize=(8, 8 / 1.2))

        # Set fixed axis limits based on first alpha value (for consistency across epochs)
        first_alpha = alpha_hist[0]
        ax = plt.gca()
        # ax.set_xlim(min_value(first_alpha, signed), first_alpha)
        # ax.set_ylim(min_value(first_alpha, signed), first_alpha)

        # Ticks and grid configuration with fixed positions
        tick_positions = np.linspace(
            min_value(first_alpha, signed), first_alpha, m_levels + 1
        )
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.tick_params(axis="x", rotation=45)

        # Actual levels
        plt.step(threshold, level, where="post", label=r"$q_u(x;\alpha)$")

        # Domain of the quantizer (current epoch's alpha)
        plt.axvline(
            x=min_value(alpha, signed),
            color="red",
            linestyle="--",
            alpha=0.25,
            label=r"$\pm\alpha$",
        )
        plt.axhline(
            y=min_value(alpha, signed), color="red", linestyle="--", alpha=0.25
        )
        plt.axvline(x=alpha, color="red", linestyle="--", alpha=0.25)
        plt.axhline(y=alpha, color="red", linestyle="--", alpha=0.25)

        # plt.axvline(x=0, color="black", linestyle=":", alpha=0.3)
        plt.axhline(y=0, color="black", linestyle=":", alpha=0.3)

        # Annotate the plot
        plt.text(
            0.05,
            0.95,
            f"Época {epoch}" if epoch > 0 else "Estado inicial",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
        )
        plt.text(
            0.05,
            0.90,
            f"Precisión: {acc:.3f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
        )
        plt.legend(loc="lower right")

        plt.grid(which="both", alpha=0.5, linestyle=":")
        plt.savefig(f"{output_path}/uniform/snapshot_{epoch:05d}.png", dpi=300)


def plot_flex_snapshot(
    alpha_hist,
    level_hist,
    threshold_hist,
    accuracy_hist,
    bits,
    signed,
    output_path,
):
    m_levels = 2**bits
    for epoch, (alpha, level, threshold, acc) in enumerate(
        zip(alpha_hist, level_hist, threshold_hist, accuracy_hist)
    ):
        plt.figure(figsize=(8, 8 / 1.2))

        # Set fixed axis limits and ticks based on first alpha value (for consistency)
        first_alpha = alpha_hist[0]
        ax = plt.gca()

        # Ticks and grid configuration with fixed positions
        tick_positions = np.linspace(
            min_value(first_alpha, signed), first_alpha, m_levels + 1
        )
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.tick_params(axis="x", rotation=45)

        # Possible values of quantization
        quantized_levels = quantize(level, alpha, m_levels, signed)
        plt.step(
            threshold,
            np.concatenate((quantized_levels, [quantized_levels[-1]])),
            label=r"$q_f(x;\alpha, \mathbf{l}, \mathbf{t})$",
            where="post",
        )
        # Actual levels
        plt.step(
            threshold,
            np.concatenate((level, [level[-1]])),
            label=r"$q_f(x;\alpha, \mathbf{l}, \mathbf{t})$ (niveles sin cuantizar)",
            linestyle="--",
            where="post",
        )

        # Domain of the quantizer (current epoch's alpha)
        plt.axvline(
            x=min_value(alpha, signed),
            color="red",
            linestyle="--",
            alpha=0.25,
            label=r"$\pm\alpha$",
        )
        plt.axhline(
            y=min_value(alpha, signed), color="red", linestyle="--", alpha=0.25
        )
        plt.axvline(x=alpha, color="red", linestyle="--", alpha=0.25)
        plt.axhline(y=alpha, color="red", linestyle="--", alpha=0.25)

        # plt.axvline(x=0, color="black", linestyle=":", alpha=0.3)
        plt.axhline(y=0, color="black", linestyle=":", alpha=0.3)

        # Annotate the plot
        plt.text(
            0.05,
            0.95,
            f"Época {epoch}" if epoch > 0 else "Estado inicial",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
        )
        plt.text(
            0.05,
            0.90,
            f"Precisión: {acc:.3f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
        )
        plt.legend(loc="lower right")

        plt.grid(which="both", alpha=0.5, linestyle=":")
        plt.savefig(f"{output_path}/flex/snapshot_{epoch:05d}.png", dpi=300)
        # ffmpeg -framerate 2 -i snapshots/snapshot_%05d.png -c:v libx264 -pix_fmt yuv420p snapshots/output.mp4 -y


def plot_training_history(vars: dict, output_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    def plot_scalar_variable(ax, history, label):
        if label == "alpha":
            label = r"$\alpha$"
            # Plot alpha with higher zorder so it appears on top
            ax.plot(history, label=label, linestyle="--", zorder=10)
        else:
            ax.plot(history, label=label)

    def plot_multi_variable(ax, history, label):
        # unwrap each index as a scalar variable
        # [[var1_epoch1, var2_epoch1, ...], [var1_epoch2, var2_epoch2, ...], ...
        # Get a color from the cycle
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
        color_idx = len(ax.lines) % len(colors)
        var_color = colors[color_idx]

        for idx in range(len(history[0])):
            var_history = [epoch_vals[idx] for epoch_vals in history]
            # Only add label to first index
            line_label = label if idx == 0 else None
            # Use lower zorder for multi-variables so alpha appears on top
            ax.plot(var_history, color=var_color, label=line_label, zorder=1)

    # Plot loss in the first subplot with a distinct color
    if "loss" in vars:
        loss_data = vars["loss"]
        ax1.plot(loss_data, label=r"$\mathcal{L}$", color="red")
        # ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(which="both", alpha=0.5)

    # Plot other variables in the second subplot
    for var_name, var in vars.items():
        if var_name == "loss":
            continue
        if isinstance(var, list) and all(
            isinstance(v, (list, np.ndarray)) for v in var
        ):
            plot_multi_variable(ax2, var, var_name)
        else:
            plot_scalar_variable(ax2, var, var_name)

    # position legend lower left
    ax2.legend(loc="lower left")
    ax2.grid(which="both", alpha=0.5)

    # Add common x-label for both subplots
    fig.text(0.5, 0.04, "Época", ha="center", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for the common xlabel
    plt.savefig(output_path, dpi=300)
