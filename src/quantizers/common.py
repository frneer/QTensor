"""Common quantizer functions."""

import tensorflow as tf


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
