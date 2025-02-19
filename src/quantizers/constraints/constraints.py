import numpy as np
import tensorflow as tf

from quantizers.common import max_value, min_value


class PositiveConstraint(tf.keras.constraints.Constraint):
    """Constrains the values to be positive."""

    def __call__(self, w):
        return tf.clip_by_value(w, tf.keras.backend.epsilon(), np.inf)


class LevelConstraint(tf.keras.constraints.Constraint):
    """Constrains the values to:
    1. Be clipped between min_value and max_value of the quantization levels
    2. Be ordered
    3. Have the first value to be fixed at the min_value
    """

    def __init__(self, alpha: tf.Variable, m_levels: int, signed: bool = True):
        assert isinstance(alpha, tf.Variable), "alpha must be a tf.Variable"
        self.alpha = alpha
        self.m_levels = m_levels
        self.signed = signed

    def __call__(self, w):
        # Compute the min and max value in each step, to evaluate alpha along the way
        min_level = min_value(self.alpha, self.signed)
        max_level = max_value(self.alpha, self.m_levels, self.signed)

        w = tf.clip_by_value(
            w, min_level, max_level + tf.keras.backend.epsilon()
        )
        w = tf.sort(w)
        w = tf.tensor_scatter_nd_update(w, [[0]], [min_level])
        return w

    def get_config(self):
        return {
            "alpha": self.alpha,
            "m_levels": self.m_levels,
            "signed": self.signed,
        }


class ThresholdConstraint(tf.keras.constraints.Constraint):
    """Constrains the values to:

    1. Be clipped between -alpha and alpha
    2. Be ordered
    3. Have the first value to be fixed at -alpha
    4. Have the last value to be fixed at alpha
    """

    def __init__(self, alpha, signed):
        self.alpha = alpha
        self.signed = signed

    def __call__(self, w):
        # Compute the min value in each step, to evaluate alpha along the way
        min_level = min_value(self.alpha, self.signed)

        w = tf.clip_by_value(w, -self.alpha, self.alpha)
        w = tf.sort(w)
        w = tf.tensor_scatter_nd_update(w, [[0]], [min_level])
        w = tf.tensor_scatter_nd_update(w, [[w.shape[0] - 1]], [self.alpha])
        return w

    def get_config(self):
        return {"alpha": self.alpha}
