import numpy as np
import tensorflow as tf

class CompositeConstraint(tf.keras.constraints.Constraint):
    """Applies multiple constraints in sequential order."""
    def __init__(self, *constraints):
        self.constraints = constraints

    def __call__(self, w):
        for constraint in self.constraints:
            w = constraint(w)
        return w

    def get_config(self):
        return {"constraints": [c.get_config() for c in self.constraints]}

class ClippedConstraint(tf.keras.constraints.Constraint):
    """Constrains the values to be clipped."""
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}

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

class FixedValueConstraint(tf.keras.constraints.Constraint):
    """Constrains certain values to be fixed defined by a list of indices."""
    def __init__(self, value, idx: int):
        self.value = value
        self.idx = idx

    def __call__(self, w):
        w = tf.tensor_scatter_nd_update(w, [[self.idx]], [self.value])
        return w
