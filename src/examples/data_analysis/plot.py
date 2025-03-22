import numpy as np
import matplotlib.pyplot as plt

def plot_flex_snapshot(history, accuracy_history, output_path):
    """
    :param accuracy_history: history of the accuracy
    :param history: history of a single layer weights
    """
    alpha_hist = history["alpha"]
    level_hist = history["level"]
    threshold_hist = history["threshold"]
    accuracy_hist = history["accuracy"]

    m_levels = 2 ** bits
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
        plt.axvline(
            x=min_value(alpha, signed), color="red", linestyle="--", alpha=0.5
        )
        plt.axhline(
            y=min_value(alpha, signed), color="red", linestyle="--", alpha=0.5
        )
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
        )

        plt.savefig(output_path / f"epoch_{epoch}.png")
        plt.close()
