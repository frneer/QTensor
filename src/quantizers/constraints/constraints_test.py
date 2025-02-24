#!/usr/bin/env python3

import unittest

import tensorflow as tf

from quantizers.constraints.constraints import (
    LevelConstraint,
    PositiveConstraint,
    ThresholdConstraint,
)


def assertListAlmostEqual(self, list1, list2):
    self.assertEqual(len(list1), len(list2))
    for i in range(len(list1)):
        self.assertAlmostEqual(
            list1[i], list2[i], delta=2 * tf.keras.backend.epsilon()
        )


class TestPositiveContraint(unittest.TestCase):
    class TestPositiveConstraint(unittest.TestCase):
        def test_zero(self):
            pc = PositiveConstraint()
            self.assertEqual(pc(0), tf.keras.backend.epsilon())

        def test_negative(self):
            pc = PositiveConstraint()
            self.assertEqual(pc(-1), tf.keras.backend.epsilon())

        def test_positive(self):
            pc = PositiveConstraint()
            self.assertEqual(pc(1), 1)

        def test_small_positive(self):
            pc = PositiveConstraint()
            self.assertEqual(pc(1e-7), tf.keras.backend.epsilon())

        def test_small_negative(self):
            pc = PositiveConstraint()
            self.assertEqual(pc(-1e-7), tf.keras.backend.epsilon())

        def test_large_positive(self):
            pc = PositiveConstraint()
            self.assertEqual(pc(1e7), 1e7)

        def test_large_negative(self):
            pc = PositiveConstraint()
            self.assertEqual(pc(-1e7), tf.keras.backend.epsilon())


class TestLevelConstraint(unittest.TestCase):
    def test_clipping(self):
        alpha = tf.Variable(1.0)
        lc = LevelConstraint(alpha, m_levels=8, signed=True)

        assertListAlmostEqual(
            self,
            lc(tf.constant([-2, -1, 0, 1, 2], dtype=tf.float32))
            .numpy()
            .tolist(),
            [-1.0, -1.0, 0.0, 3 / 4, 3 / 4],
        )

    def test_ordering(self):
        alpha = tf.Variable(100.0)
        lc = LevelConstraint(alpha, m_levels=8, signed=True)

        assertListAlmostEqual(
            self,
            lc(tf.constant([2, 1, 0, -1, -2], dtype=tf.float32))
            .numpy()
            .tolist(),
            [-100, -1, 0, 1, 2],
        )

    def test_update_alpha(self):
        alpha = tf.Variable(1.0)
        lc = LevelConstraint(alpha, m_levels=8, signed=True)

        assertListAlmostEqual(
            self,
            lc(tf.constant([-2, -1, 0, 1, 2], dtype=tf.float32))
            .numpy()
            .tolist(),
            [-1.0, -1.0, 0.0, 3 / 4, 3 / 4],
        )

        alpha.assign(2.0)
        assertListAlmostEqual(
            self,
            lc(tf.constant([-2, -1, 0, 1, 2], dtype=tf.float32))
            .numpy()
            .tolist(),
            [-2.0, -1.0, 0.0, 1.0, 1.5],
        )


class TestThresholdConstraint(unittest.TestCase):
    def test_clipping(self):
        alpha = 1.0
        tc = ThresholdConstraint(alpha, signed=True)

        assertListAlmostEqual(
            self,
            tc(tf.constant([-2, -1, 0, 1, 2], dtype=tf.float32))
            .numpy()
            .tolist(),
            [-1.0, -1.0, 0.0, 1.0, 1.0],
        )

    def test_ordering(self):
        alpha = 100.0
        tc = ThresholdConstraint(alpha, signed=True)

        assertListAlmostEqual(
            self,
            tc(tf.constant([2, 1, 0, -1, -2], dtype=tf.float32))
            .numpy()
            .tolist(),
            [-100, -1, 0, 1, 100],
        )

    def test_update_alpha(self):
        alpha = tf.Variable(1.0)
        tc = ThresholdConstraint(alpha, signed=True)

        assertListAlmostEqual(
            self,
            tc(tf.constant([-2, -1, 0, 1, 2], dtype=tf.float32))
            .numpy()
            .tolist(),
            [-1.0, -1.0, 0.0, 1.0, 1.0],
        )

        alpha.assign(2.0)
        assertListAlmostEqual(
            self,
            tc(tf.constant([-2, -1, 0, 1, 2], dtype=tf.float32))
            .numpy()
            .tolist(),
            [-2.0, -1.0, 0.0, 1.0, 2.0],
        )


if __name__ == "__main__":
    unittest.main()
