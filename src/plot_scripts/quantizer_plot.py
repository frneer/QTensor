#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from quantizers.common import delta, max_value, min_value


def plot_levels(ax, thresholds, levels, **kwargs):
    ax.step(
        thresholds,
        levels,
        color=kwargs.get("color", "blue"),
        linestyle=kwargs.get("linestyle", "-"),
        lw=1,
        where="post",
        label=kwargs.get("label", "función de cuantización"),
    )


def plot_ste(ax, min_limit, max_limit):
    dq_dx = np.linspace(min_limit, max_limit, 100)
    ax.plot(
        dq_dx,
        dq_dx,
        color="purple",
        linestyle="-.",
        lw=1,
        alpha=0.5,
        label="ste",
    )


def plot_limits(ax, min_limit, max_limit):
    ax.axvline(x=min_limit, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=min_limit, color="red", linestyle="--", alpha=0.5)
    ax.axvline(x=max_limit, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=max_limit, color="red", linestyle="--", alpha=0.5)


def generate_uniform(bits, alpha, signed):
    m_levels = 2**bits
    min_val = min_value(alpha, signed)
    max_val = max_value(alpha, m_levels, signed)
    delta_val = delta(alpha, m_levels, signed)

    levels = np.arange(min_val, max_val + delta_val, delta_val)
    thresholds = levels

    # Add last level and threshold for plotting
    levels = np.append(levels, max_val)
    thresholds = np.append(thresholds, max_val + delta_val)

    return thresholds, levels


def generate_non_uniform(alpha, signed, m_levels):
    np.random.seed(42)
    # Randomly sample levels from a uniform distribution
    levels = np.random.uniform(
        low=min_value(alpha, signed),
        high=max_value(alpha, m_levels, signed),
        size=m_levels,
    )
    levels = np.sort(levels)

    # Randomly sample thresholds from a uniform distribution
    thresholds = np.random.uniform(
        low=min_value(alpha, signed),
        high=max_value(alpha, m_levels, signed),
        size=m_levels + 1,
    )
    thresholds = np.sort(thresholds)

    levels = np.append(levels, levels[-1])

    return thresholds, levels


def plot_uniform(
    bits, alpha, signed, fig_path, do_plot_ste=False, alpha_generic=False
):
    fig, ax = plt.subplots()

    # Generate quantization levels and thresholds
    thresholds, levels = generate_uniform(bits, alpha, signed)

    min_limit = thresholds[0]
    max_limit = thresholds[-1]

    # Plots
    plot_limits(ax, min_limit, max_limit)
    plot_levels(ax, thresholds, levels)
    if do_plot_ste:
        plot_ste(ax, min_limit, max_limit)

    thresholds_ticks = thresholds
    levels_ticks = levels  # Exclude the last level for ticks
    levels_ticks[-1] = max_limit  # Ensure the last tick is at max_limit
    thresholds_labels = [f"{t:.3f}" for t in thresholds]
    levels_labels = [f"{l:.3f}" for l in levels]
    rotation = 45

    if alpha_generic and bits == 3 and signed:
        m_levels = 2**bits
        # double de step to generate major/minor ticks
        step = delta(alpha, m_levels, signed)

        levels_ticks = np.arange(min_limit, max_limit + step, step)
        # Set major ticks
        labels = [
            r"$-\alpha$",
            "",
            r"$-\frac{\alpha}{2}$",
            "",
            "0",
            "",
            r"$\frac{\alpha}{2}$",
            "",
            r"$\alpha$",
        ]
        thresholds_labels = labels
        levels_labels = labels
        rotation = 0

    plt.xticks(
        ticks=thresholds_ticks, labels=thresholds_labels, rotation=rotation
    )
    plt.yticks(ticks=levels_ticks, labels=levels_labels, rotation=rotation)
    # Annotations
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$q$")
    ax.grid(which="both", alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=10)
    plt.savefig(f"{fig_path}.png")


def plot_sym_non_sym_uniform(bits, alpha, signed, fig_path, do_plot_ste=False):
    fig, ax = plt.subplots()

    thresholds, levels = generate_uniform(bits, alpha, signed)
    plot_levels(ax, thresholds, levels, color="orange", linestyle="-")

    thresholds, levels = generate_uniform(bits, alpha, not signed)
    plot_levels(ax, thresholds, levels, color="blue", linestyle="--")

    ax.axhline(y=0, color="black", linestyle="-", lw=1, alpha=0.5)
    ax.axvline(x=0, color="black", linestyle="-", lw=1, alpha=0.5)
    ax.legend([r"Simétrica", r"Asimétrica"], loc="upper left", fontsize=10)

    plt.xticks([0])
    plt.yticks([0])

    # Annotations
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$q$")
    # ax.grid(which="both", alpha=0.3, linestyle=":")
    plt.savefig(f"{fig_path}.png")


def plot_uniform_vs_non_uniform(
    bits, alpha, signed, fig_path, do_plot_ste=False
):
    fig, ax = plt.subplots()

    thresholds, levels = generate_uniform(bits, alpha, signed)
    plot_levels(ax, thresholds, levels, color="orange", linestyle="-")

    thresholds, levels = generate_non_uniform(alpha, signed, 2**bits)
    plot_levels(ax, thresholds, levels, color="blue", linestyle="--")

    ax.axhline(y=0, color="black", linestyle="-", lw=1, alpha=0.5)
    ax.axvline(x=0, color="black", linestyle="-", lw=1, alpha=0.5)
    ax.legend([r"Uniforme", r"No uniforme"], loc="upper left", fontsize=10)

    plt.xticks([0])
    plt.yticks([0])

    # Annotations
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$q$")
    # ax.grid(which="both", alpha=0.3, linestyle=":")
    plt.savefig(f"{fig_path}.png")


def generate_flex_quantizer_fixed(bits, alpha, signed):
    # Data taken from training a model with flex quantization
    pass
    # levels += [levels[-1]]
    qlevels = [-0.15165123, -0.11373842, -0.03791281, 0.0, 0.07582562]
    qlevels = qlevels + [qlevels[-1]]
    # qlevels += [qlevels[-1]]
    thresholds = [
        -0.15165123,
        -0.01363241,
        0.0057151,
        0.0173785,
        0.08350082,
        0.15165123,
    ]

    return thresholds, qlevels


def generate_non_uniform_fixed():
    levels = [-0.15165123, -0.08514632, -0.02999737, 0.02491043, 0.07670338]
    levels = levels + [levels[-1]]
    thresholds = [
        -0.15165123,
        -0.01363241,
        0.0057151,
        0.0173785,
        0.08350082,
        0.15165123,
    ]

    return thresholds, levels


def plot_flex_quantizer(fig_path, do_plot_prequantized, do_plot_ste=False):
    thresholds, levels = generate_flex_quantizer_fixed(
        bits=3, alpha=1, signed=True
    )
    thresholds_nu, levels_nu = generate_non_uniform_fixed()

    fig, ax = plt.subplots()

    plot_limits(ax, thresholds[0], thresholds[-1])

    plot_levels(
        ax,
        thresholds,
        levels,
        color="blue",
        linestyle="-",
        label="función de cuantización",
    )

    if do_plot_ste:
        plot_ste(ax, thresholds[0], thresholds[-1])

    if do_plot_prequantized:
        plot_levels(
            ax,
            thresholds_nu,
            levels_nu,
            color="orange",
            linestyle=":",
            label="función con niveles sin cuantizar",
        )

    ax.legend(loc="upper left", fontsize=10)
    # Annotations
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$q$")
    ax.grid(which="both", alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=10)

    x_labels = [rf"$t_0 = -\alpha$"]
    x_labels += [rf"$t_{i}$" for i in range(1, len(thresholds) - 1)]
    x_labels += [rf"$t_{len(thresholds) - 1} = \alpha$"]

    plt.xticks(thresholds, labels=x_labels)
    plt.yticks(levels, labels=[rf"$l_{i}$" for i in range(len(levels))])

    plt.savefig(f"{fig_path}.png")


def plot_non_uniform_quantizer(fig_path):
    thresholds, levels = generate_non_uniform_fixed()

    fig, ax = plt.subplots()

    plot_limits(ax, thresholds[0], thresholds[-1])

    plot_levels(
        ax,
        thresholds,
        levels,
        color="blue",
        linestyle="-",
        label="función de cuantización",
    )

    ax.legend(loc="upper left", fontsize=10)
    # Annotations
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$q$")
    ax.grid(which="both", alpha=0.3, linestyle=":")
    ax.legend(loc="upper left", fontsize=10)

    x_labels = [rf"$t_0 = -\alpha$"]
    x_labels += [rf"$t_{i}$" for i in range(1, len(thresholds) - 1)]
    x_labels += [rf"$t_{len(thresholds) - 1} = \alpha$"]

    plt.xticks(thresholds, labels=x_labels)
    plt.yticks(levels, labels=[rf"$l_{i}$" for i in range(len(levels))])

    plt.savefig(f"{fig_path}.png")


if __name__ == "__main__":
    plot_uniform(bits=3, alpha=1, signed=True, fig_path="signed_3bits")
    plot_uniform(bits=3, alpha=1, signed=False, fig_path="unsigned_3bits")
    plot_sym_non_sym_uniform(
        bits=3, alpha=1, signed=True, fig_path="sym_vs_asym_3bits"
    )
    plot_uniform_vs_non_uniform(
        bits=3, alpha=1, signed=True, fig_path="uniform_vs_non_uniform_3bits"
    )
    plot_uniform(bits=3, alpha=4, signed=True, fig_path="signed_3bits_alpha4")
    plot_uniform(
        bits=3,
        alpha=1,
        signed=True,
        fig_path="signed_generic_3bits",
        alpha_generic=True,
    )
    plot_uniform(
        bits=3,
        alpha=1,
        signed=True,
        fig_path="signed_generic_3bits_ste",
        alpha_generic=True,
        do_plot_ste=True,
    )

    plot_flex_quantizer(
        fig_path="flex_quantizer_3bits",
        do_plot_prequantized=True,
        do_plot_ste=True,
    )
    plot_flex_quantizer(
        fig_path="flex_quantizer_3bits_no_prequantized",
        do_plot_prequantized=False,
        do_plot_ste=True,
    )
    plot_flex_quantizer(
        fig_path="flex_quantizer_3bits_no_prequantized_no_ste",
        do_plot_prequantized=False,
        do_plot_ste=False,
    )
    plot_flex_quantizer(
        fig_path="flex_quantizer_3bits_no_ste",
        do_plot_prequantized=True,
        do_plot_ste=False,
    )

    plot_non_uniform_quantizer(fig_path="non_uniform_quantizer_3bits")
