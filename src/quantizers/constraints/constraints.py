import numpy as np
import tensorflow as tf

class CompositeConstraint(tf.keras.constraints.Constraint):
    """Applies multiple constraints in sequential order."""
    def __init__(self, *constraints):
        self.constraints = constraints

    def __call__(self, w):
        for constraint in self.constraints:
            print(constraint.__class__.__name__)
            w = constraint(w)
        return w

    def get_config(self):
        return {"constraints": [c.get_config() for c in self.constraints]}
class PositiveConstraint(ClippedConstraint):
    """Constrains the values to be positive."""
    def __init__(self):
        super().__init__(tf.keras.backend.epsilon(), np.inf)

class OrderedConstraint(tf.keras.constraints.Constraint):
    """Constrains the values to be ordered."""
    def __init__(self, axis=-1, ascending=True):
        self.axis = axis
        self.ascending = ascending

    def __call__(self, w):
        return tf.sort(w, self.axis, direction="ASCENDING" if self.ascending else "DESCENDING")

class ClippedConstraint(tf.keras.constraints.Constraint):
    """Constrains the values to be clipped."""
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


class FixedValueConstraint(tf.keras.constraints.Constraint):
    """Constrains certain values to be fixed defined by a list of indices."""
    def __init__(self, value, idx: int):
        self.value = value
        self.idx = idx

    def __call__(self, w):
        w = tf.tensor_scatter_nd_update(w, [[self.idx]], [self.value])
        return w

    def get_config(self):
        return {"value": self.value, "idx": self.idx}


class ClippedAndOrderedConstraint(tf.keras.constraints.Constraint):
    """Constrains the values to be ordered."""
    def __init__(self, alpha):
        self.alpha = alpha
    def __call__(self, w):
        ret = tf.clip_by_value(w, -self.alpha, self.alpha)
        return tf.sort(ret)

class LevelConstraint(ClippedAndOrderedConstraint):
    """Constrains the values to be ordered."""
    def __init__(self, alpha, bits):
        super().__init__(alpha)
        self.bits = bits
    def __call__(self, w):
        w = super().__call__(w)
        w = tf.tensor_scatter_nd_update(w, [[0]], [-self.alpha])
        max_res_value = 2**self.bits
        max_value = (max_res_value - 2) * self.alpha / max_res_value
        tf.clip_by_value(w, -self.alpha, max_value + tf.keras.backend.epsilon())
        return w

class ThresholdConstraint(ClippedAndOrderedConstraint):
    """Constrains the values to be ordered."""
    def __init__(self, alpha):
        super().__init__(alpha)
    def __call__(self, w):
        w = super().__call__(w)
        w = tf.tensor_scatter_nd_update(w, [[0]], [-self.alpha])
        w = tf.tensor_scatter_nd_update(w, [[w.shape[0] - 1]], [self.alpha])
        return w
