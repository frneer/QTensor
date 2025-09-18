#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


# TODO(Fran): Migrate this plots to src and leverage the quantizers.common module
def min_value(alpha, signed):
    """Return the minimum possible value of the quantization range."""
    return -alpha if signed else 0


def span(alpha, signed):
    """Return the span of the quantizer."""
    return 2 * alpha if signed else alpha


def max_value(alpha, m_levels, signed):
    """Return the maximum possible value of the quantization range."""
    return (
        min_value(alpha, signed)
        + span(alpha, signed) * (m_levels - 1) / m_levels
    )


def delta(alpha, m_levels, signed):
    """Returns the quantization step."""
    return span(alpha, signed) / m_levels


def quantize(x, alpha, m_levels, signed):
    """Simple uniform quantization function."""
    delta_v = delta(alpha, m_levels, signed)
    return delta_v * tf.math.floor(x / delta_v)


def main():
    plot_ste = True
    name = "q_ste" if plot_ste else "q"
    bits = 3
    alpha = 1
    n_levels = 2 ** (bits - 1)
    alpha / n_levels
    m_levels = 2**bits
    signed = True

    alpha = 0.15165123343467712
    levels = [-0.15165123, -0.08514632, -0.02999737, 0.02491043, 0.07670338]
    # levels += [levels[-1]]
    qlevels = [-0.15165123, -0.11373842, -0.03791281, 0.0, 0.07582562]
    # qlevels += [qlevels[-1]]
    thresholds = [
        -0.15165123,
        -0.01363241,
        0.0057151,
        0.0173785,
        0.08350082,
        0.15165123,
    ]

    # levels = np.insert(levels, 0, levels[0])
    # levels = np.insert(levels, -1, levels[-1])

    # thresholds = np.insert(thresholds, 0, -(alpha + step))
    # thresholds = np.insert(thresholds, -1, alpha + step)

    # dq_dx = np.insert(dq_dx, alpha, alpha)
    # dq_dx_neg = [-1.25, -1]
    # dq_dx_pos = [1, 1.25]

    plt.figure()
    # Domain of the quantizer
    plt.axvline(
        x=min_value(alpha, signed),
        color="red",
        linestyle="--",
        alpha=0.5,
    )
    plt.axhline(
        y=min_value(alpha, signed), color="red", linestyle="--", alpha=0.5
    )
    plt.axvline(x=alpha, color="red", linestyle="--", alpha=0.5)
    plt.axhline(y=alpha, color="red", linestyle="--", alpha=0.5)

    # Generate the levels without quantization
    plt.step(
        thresholds,
        levels + [levels[-1]],
        label=r"$levels without quantization$",
        color="orange",
        linestyle="--",
        lw=1,
        where="post",
    )

    # Generate the quantized levels
    plt.step(
        thresholds,
        qlevels + [qlevels[-1]],
        label=r"$q\left(x; \alpha\right)$",
        color="blue",
        lw=1,
        # linestyle="--",
        where="post",
    )

    xticks = thresholds
    plt.xticks(
        xticks,
        labels=[rf"$t_0 = -\alpha$"]
        + [rf"$t_{i}$" for i in range(1, len(thresholds) - 1)]
        + [rf"$t_{len(thresholds) - 1} = \alpha$"],
    )

    # Ticks and grid configuration
    # Estos yticks son con todos los bits
    # yticks = np.linspace(
    #         levels[0], max_value(alpha, m_levels, signed), m_levels
    # )
    # yticks = np.round(yticks, 8)
    # plt.yticks(
    #     yticks,
    #     labels=[fr"$l_{i}$" if yticks[i] in qlevels else "" for i in range(m_levels)],
    # )

    bit_levels = np.linspace(
        levels[0], max_value(alpha, m_levels, signed), m_levels
    )
    print("bit_levels", bit_levels)
    print("qlevels", qlevels)
    plt.yticks(
        bit_levels,
        labels=[
            (
                rf"$l_{i}$"
                if np.isin(np.round(bit_levels[i], 5), np.round(qlevels, 5))
                else ""
            )
            for i in range(m_levels)
        ],
    )
    # Plot labels
    plt.xlabel(r"$x$")
    plt.ylabel(r"$q(x; \alpha, \overline{t}, \overline{l})$")
    # plt.legend(loc="upper left", fontsize=8)
    plt.gca()
    plt.grid(which="both", alpha=0.3, linestyle=":")
    plt.savefig(f"../pics/quantizers/flex/{name}_q.png")


if __name__ == "__main__":
    main()
