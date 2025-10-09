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
    )


def plot_ste(ax, min_limit, max_limit):
    dq_dx = np.linspace(min_limit, max_limit, 100)
    ax.plot(dq_dx, dq_dx, color="purple", linestyle="-", lw=2, alpha=0.5)


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


def plot_uniform(bits, alpha, signed, fig_path, plot_ste=False):
    fig, ax = plt.subplots()

    # Generate quantization levels and thresholds
    thresholds, levels = generate_uniform(bits, alpha, signed)

    min_limit = thresholds[0]
    max_limit = thresholds[-1]

    # Plots
    plot_limits(ax, min_limit, max_limit)
    plot_levels(ax, thresholds, levels)
    if plot_ste:
        plot_ste(ax, min_limit, max_limit)

    plt.xticks(
        ticks=thresholds, labels=[f"{t:.3f}" for t in thresholds], rotation=45
    )
    plt.yticks(
        ticks=levels[:-1],
        labels=[f"{l:.3f}" for l in levels[:-1]],
        rotation=45,
    )
    # # alpha annotated ticks
    # step = 2 * delta(alpha, m_levels, signed) # double de step to generate major/minor ticks
    # ## Set major ticks
    # labels = [
    #     r"$-\alpha$",
    #     r"$-\frac{\alpha}{2}$",
    #     "0",
    #     r"$\frac{\alpha}{2}$",
    #     r"$\alpha$",
    # ]
    # plt.xticks(ticks=np.arange(min_limit, max_limit + step, step), labels=labels)
    # plt.yticks(ticks=np.arange(min_limit, max_limit + step, step), labels=labels)
    # ## Set minor ticks
    # ax = plt.gca()
    # ax.set_xticks(np.arange(min_limit + step, max_limit, step), minor=True)
    # ax.set_yticks(np.arange(min_limit + step, max_limit, step), minor=True)

    # Annotations
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$q(x; \alpha)$")
    ax.grid(which="both", alpha=0.3, linestyle=":")
    plt.savefig(f"{fig_path}.png")


def plot_sym_non_sym_uniform(bits, alpha, signed, fig_path, plot_ste=False):
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
    ax.set_ylabel(r"$q(x; \alpha)$")
    # ax.grid(which="both", alpha=0.3, linestyle=":")
    plt.savefig(f"{fig_path}.png")


def plot_uniform_vs_non_uniform(bits, alpha, signed, fig_path, plot_ste=False):
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
    ax.set_ylabel(r"$q(x; \alpha)$")
    # ax.grid(which="both", alpha=0.3, linestyle=":")
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
