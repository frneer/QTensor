#!/usr/bin/env python3

import unittest

import numpy as np
import tensorflow as tf

from utils.huffman import compute_huffman_nominal_complexity


class TestComputeHuffmanNominalComplexity(unittest.TestCase):
    """Unit tests for `compute_huffman_nominal_complexity`."""

    def test_all_identical(self):
        """Entropy should be zero when every symbol is the same."""
        q = tf.constant([7, 7, 7, 7], dtype=tf.int32)
        expected = 0.0  # N=4 => 0 bits
        self.assertAlmostEqual(
            compute_huffman_nominal_complexity(q), expected, places=6
        )

    def test_balanced_two_symbols(self):
        """50 % / 50 % distribution => entropy = 1 bit per symbol."""
        q = tf.constant([0, 1, 0, 1], dtype=tf.int32)
        expected = 4.0  # N=4 => 4 bits
        self.assertAlmostEqual(
            compute_huffman_nominal_complexity(q), expected, places=6
        )

    def test_three_to_one_ratio(self):
        """75 % / 25 % distribution => entropy ~0.811278 bits per symbol."""
        q = tf.constant([0, 0, 0, 1], dtype=tf.int32)
        entropy = -(
            0.75 * np.log2(0.75) + 0.25 * np.log2(0.25)
        )  # ~0.811278 bits
        expected = 4 * entropy  # N=4 => ~3.2451 bits
        self.assertAlmostEqual(
            compute_huffman_nominal_complexity(q), expected, places=6
        )

    def test_larger_vector_distribution(self):
        """100 elements: 50*0, 30*1, 20*2 =>"""
        # build the vector
        vals = [0] * 50 + [1] * 30 + [2] * 20
        q = tf.constant(vals, dtype=tf.int32)

        # compute expected: N=100, p0=0.5, p1=0.3, p2=0.2
        ps = np.array([0.5, 0.3, 0.2])
        entropy = -np.sum(ps * np.log2(ps))  # ~1.485475
        expected = 100 * entropy  # N=100 => ~148.5475

        self.assertAlmostEqual(
            compute_huffman_nominal_complexity(q), expected, places=6
        )


if __name__ == "__main__":
    unittest.main()
