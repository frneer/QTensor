#!/usr/bin/env python3
import matplotlib.pyplot as plt


def main():
    thresholds = [
        -0.15165123,
        -0.01363241,
        0.0057151,
        0.0173785,
        0.08350082,
        0.15165123,
    ]
    qlevels = [-0.15165123, -0.11373842, -0.03791281, 0.0, 0.07582562]

    plt.figure()
    plt.step(thresholds, qlevels)
    plt.savefig(f"dq_dt.png")


if __name__ == "__main__":
    main()
