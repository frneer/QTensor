import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from quantizers.common import max_value, min_value, quantize


class VariableHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, variable):
        super(VariableHistoryCallback, self).__init__()
        self.variable = variable
        self.variable_values = []

    def on_epoch_end(self, epoch, logs=None):
        # Record the variable's value at the end of each epoch
        self.variable_values.append(self.variable.numpy())

    def get_history(self):
        return self.variable_values


def plot_snapshot(
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
        plt.figure()
        # Actual levels
        plt.step(threshold, np.concatenate((level, [level[-1]])), where="post")

        # Possible values of quantization
        quantized_levels = quantize(level, alpha, m_levels, signed)
        plt.step(
            threshold,
            np.concatenate((quantized_levels, [quantized_levels[-1]])),
            label=f"Quantized levels",
            linestyle="--",
            where="post",
        )

        # Domain of the quantizer
        plt.axvline(x=min_value(alpha, signed), color="red", linestyle="--", alpha=0.5)
        plt.axhline(y=min_value(alpha, signed), color="red", linestyle="--", alpha=0.5)
        plt.axvline(x=alpha, color="red", linestyle="--", alpha=0.5)
        plt.axhline(y=alpha, color="red", linestyle="--", alpha=0.5)

        plt.axvline(x=0, color="black", linestyle=":", alpha=0.3)
        plt.axhline(y=0, color="black", linestyle=":", alpha=0.3)

        # Annotate the plot
        plt.text(
            0.05,
            0.95,
            f"Epoch {epoch + 1}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
        )
        plt.text(
            0.05,
            0.90,
            f"Accuracy: {acc:.3f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
        )
        plt.legend(loc="lower right")

        # Ticks and grid configuration
        plt.yticks(
            np.linspace(level[0], max_value(alpha, m_levels, signed), m_levels),
            labels=[f"" for _ in range(m_levels)],
        )
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=0)
        plt.grid(which="both", alpha=0.3, linestyle=":")
        plt.savefig(f"{output_path}/snapshot_{epoch:05d}.png")
        plt.close()

        # ffmpeg -framerate 2 -i snapshots/snapshot_%05d.png -c:v libx264 -pix_fmt yuv420p snapshots/output.mp4 -y


def plot_training_history(history, callbacks):
    """Plot the training history including loss, accuracy, and alpha
    variables."""
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Plot training & validation loss values
    axs[0].plot(history.history["loss"])
    axs[0].plot(history.history["val_loss"])
    axs[0].set_title("Model loss")
    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].legend(["Train", "Validation"], loc="upper left")
    axs[0].grid(which="both")

    # Plot training & validation accuracy values
    axs[1].plot(history.history["accuracy"])
    axs[1].plot(history.history["val_accuracy"])
    axs[1].set_title("Model accuracy")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(["Train", "Validation"], loc="upper left")
    axs[1].grid(which="both")

    # Plot alpha history
    for callback in callbacks:
        axs[2].plot(callback.get_history(), label=callback.variable.name)
    axs[2].set_title("Alpha history")
    axs[2].set_ylabel("Alpha")
    axs[2].set_xlabel("Epoch")
    axs[2].legend([callback.variable.name for callback in callbacks])
    axs[2].grid(which="both")

    plt.tight_layout()
    plt.savefig("training_history.png")
