#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def main():
    plot_ste = False
    name = "q_ste" if plot_ste else "q"
    bits = 3
    alpha = 1
    n_levels = 2 ** (bits - 1)
    step = alpha / n_levels

    levels = np.arange(-alpha, alpha, step)
    input_levels = levels.copy() + step
    dq_dx = levels.copy()

    levels = np.insert(levels, 0, levels[0])
    levels = np.insert(levels, -1, levels[-1])

    input_levels = np.insert(input_levels, 0, -(alpha + step))
    input_levels = np.insert(input_levels, -1, alpha + step)

    dq_dx = np.insert(dq_dx, alpha, alpha)
    dq_dx_neg = [-1.25, -1]
    dq_dx_pos = [1, 1.25]

    plt.figure()
    # Generate the box of lenght alpha)
    plt.axvline(x=-alpha, color="red", linestyle="--", alpha=0.5)
    plt.axhline(y=-alpha, color="red", linestyle="--", alpha=0.5)
    plt.axvline(x=alpha, color="red", linestyle="--", alpha=0.5)
    plt.axhline(y=alpha, color="red", linestyle="--", alpha=0.5)

    # Generate the quantization funcion q(x; alpha)
    plt.step(
        input_levels,
        levels,
        label=r"$q\left(x; \alpha\right)$",
        color="blue",
        linestyle="-",
        lw=1,
    )

    # Generate STE function
    ## -alpha < x < alpha
    if plot_ste:
        plt.plot(dq_dx, dq_dx, color="purple", linestyle="-", lw=2, alpha=0.5)
        ## x < -alpha
        plt.plot(
            dq_dx_neg,
            -np.ones(len(dq_dx_neg)),
            color="purple",
            linestyle="-",
            lw=2,
            alpha=0.5,
        )
        ## x > alpha
        plt.plot(
            dq_dx_pos,
            3 / 4 * np.ones(len(dq_dx_pos)),
            label=r"STE",
            color="purple",
            linestyle="-",
            lw=2,
            alpha=0.5,
        )

    # Plot labels
    plt.xlabel(r"$x$")
    plt.ylabel(r"$q(x; \alpha)$")

    # Plot visual aids
    plt.xlim(-alpha - step, alpha + step)
    plt.ylim(-alpha - step, alpha + step)

    step = 2 * step  # double de step to generate major/minor ticks
    ## Set major ticks
    labels = [
        r"$-\alpha$",
        r"$-\frac{\alpha}{2}$",
        "0",
        r"$\frac{\alpha}{2}$",
        r"$\alpha$",
    ]
    plt.xticks(ticks=np.arange(-1, 1 + step, step), labels=labels)
    plt.yticks(ticks=np.arange(-1, 1 + step, step), labels=labels)
    ## Set minor ticks
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.75, 0.75 + step, step), minor=True)
    ax.set_yticks(np.arange(-0.75, 0.75 + step, step), minor=True)
    plt.grid(which="both", alpha=0.3, linestyle=":")
    # plt.legend(loc="upper left", fontsize=10)

    plt.savefig(f"../docs/pics/quantizers/uniform/{name}.png")


if __name__ == "__main__":
    main()
