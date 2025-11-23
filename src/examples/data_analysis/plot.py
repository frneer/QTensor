from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quantizers.common import max_value, min_value, quantize


def plot_flex_snapshot(
    layer_name, layer_history, quantizer, accuracy_history, output_path
):
    """
    :param accuracy_history: history of the accuracy
    :param history: history of a single layer weights
    """
    # NOTE(Fran): Big assumption here that the keys are always the same
    # TODO(Fran): Also it seems activations aren't being stored as model weights
    # ...
    print(layer_history.keys())
    alpha_history = layer_history[f"{layer_name}/kernel_alpha:0"]
    level_history = layer_history[f"{layer_name}/kernel_levels:0"]
    threshold_history = layer_history[f"{layer_name}/kernel_thresholds:0"]
    bits = quantizer.bits
    signed = quantizer.signed

    m_levels = 2**bits
    for epoch, (alpha, level, threshold, acc) in enumerate(
        zip(alpha_history, level_history, threshold_history, accuracy_history)
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
            0.85,
            f"Accuracy: {acc:.3f}",
            transform=plt.gca().transAxes,
            fontsize=12,
        )
        plt.legend(loc="lower right")

        # Ticks and grid configuration
        plt.yticks(
            np.linspace(
                level[0], max_value(alpha, m_levels, signed), m_levels
            ),
            labels=[f"" for _ in range(m_levels)],
        )
        ax = plt.gca()
        ax.yaxis.set_tick_params(width=0)
        plt.grid(which="both", alpha=0.3, linestyle=":")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / f"epoch_{epoch}.png")
        plt.close()
