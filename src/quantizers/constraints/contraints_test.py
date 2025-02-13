#!/usr/bin/env python

import unittest

import tensorflow as tf
from quantizers.constraints.constraints import PositiveConstraint, ClippedConstraint, OrderedConstraint, FixedValueConstraint, CompositeConstraint


class TestConstraints(unittest.TestCase):
    def assertListAlmostEqual(self, a, b):
        self.assertEqual(len(a), len(b))
        for i in range(len(a)):
            self.assertAlmostEqual(a[i], b[i], delta=2*tf.keras.backend.epsilon())

    def test_positive_constraint(self):
        constraint = PositiveConstraint()
        w = tf.constant([-1.0, 0.0, 1.0], dtype=tf.float32)
        w = constraint(w)
        self.assertListAlmostEqual([0.0, 0.0, 1.0], w.numpy())

    def test_clipped_constraint(self):
        constraint = ClippedConstraint(-1.0, 1.0)
        w = tf.constant([-2.0, 0.0, 2.0], dtype=tf.float32)
        w = constraint(w)
        self.assertListAlmostEqual([-1.0, 0.0, 1.0], w.numpy())

    def test_ordered_constraint(self):
        constraint = OrderedConstraint()
        w = tf.constant([1.0, -1.0, 0.0, 2.0], dtype=tf.float32)
        w = constraint(w)
        self.assertListAlmostEqual([-1.0, 0.0, 1.0, 2.0], w.numpy())

    def test_fixed_value_constraint(self):
        constraint = FixedValueConstraint(value=0.0, idx=1)
        w = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
        w = constraint(w)
        self.assertListAlmostEqual([1.0, 0.0, 1.0], w.numpy())

    def test_fixed_value_constraint_last(self):
        w = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
        constraint = FixedValueConstraint(value=0.0, idx=w.shape[0] - 1)
        w = constraint(w)
        self.assertListAlmostEqual([1.0, 1.0, 0.0], w.numpy())

    def test_fixed_value_constraint_ref(self):
        alpha = tf.Variable(-3.0)
        constraint = FixedValueConstraint(value=-alpha, idx=0)
        w = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
        w = constraint(w)
        self.assertListAlmostEqual([3.0, 1.0, 1.0], w.numpy())
        alpha.assign(-1.0)
        w = constraint(w)
        self.assertListAlmostEqual([1.0, 1.0, 1.0], w.numpy())

    def test_fixed_value_constraint_ref_other(self):
        alpha = tf.Variable(-3.0)
        constraint = FixedValueConstraint(value=alpha, idx=0)
        w = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
        w = constraint(w)
        self.assertListAlmostEqual([-3.0, 1.0, 1.0], w.numpy())
        alpha.assign(-1.0)
        w = constraint(w)
        self.assertListAlmostEqual([-1.0, 1.0, 1.0], w.numpy())

    # def test_composite_contraint(self):
    #     self.alpha = -3.0
    #     constraint = CompositeConstraint(
    #         ClippedConstraint(-1.0, 1.0),
    #         OrderedConstraint(),
    #         FixedValueConstraint(self.alpha, 0),
    #         FixedValueConstraint(value=1.0, idx=2),
    #     )
    #     w = tf.constant([1.0, -1.0, 0.0, 2.0], dtype=tf.float32)
    #     # Order: [-1.0, 0.0, 1.0, 2.0]
    #     # Clipped: [-1.0, 0.0, 1.0, 1.0]
    #     # FixedValueConstraint: [0.0, 0.0, 1.0, 1.0]
    #     w = constraint(w)
    #     self.assertListAlmostEqual([-3.0, 0.0, 1.0, 1.0], w.numpy())

    #     self.alpha = -1.0
    #     w = constraint(w)
    #     self.assertListAlmostEqual([-1.0, 0.0, 1.0, 1.0], w.numpy())

# GPT 1
    # def test_composite_constraint(self):
    #     alpha = -3.0
    #     constraint = CompositeConstraint(
    #         FixedValueConstraint(value=alpha, idx=0),  # Apply fixed values first
    #         ClippedConstraint(-1.0, 1.0),
    #         OrderedConstraint(),
    #         FixedValueConstraint(value=1.0, idx=2),
    #     )
    #     w = tf.constant([1.0, -1.0, 0.0, 2.0], dtype=tf.float32)
    #     # Order: [-1.0, 0.0, 1.0, 2.0]
    #     # Clipped: [-1.0, 0.0, 1.0, 1.0]
    #     # FixedValueConstraint: [-3.0, 0.0, 1.0, 1.0]
    #     w = constraint(w)
    #     self.assertListAlmostEqual([-3.0, 0.0, 1.0, 1.0], w.numpy())

    #     alpha = -1.0
    #     w = constraint(w)
    #     self.assertListAlmostEqual([-1.0, 0.0, 1.0, 1.0], w.numpy())

if __name__ == "__main__":
    unittest.main()
