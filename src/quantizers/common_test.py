#!/usr/bin/env python3

import unittest
from common import min_value, span, max_value, delta

class TestQuantizersCommon(unittest.TestCase):

    def test_min_value(self):
        self.assertEqual(min_value(alpha=10, signed=True), -10)
        self.assertEqual(min_value(alpha=10, signed=False), 0)

    def test_span(self):
        self.assertEqual(span(alpha=10, signed=True), 20)
        self.assertEqual(span(alpha=10, signed=False), 10)

    def test_delta(self):
        self.assertEqual(delta(alpha=1, signed=True, m_levels=8), 1/4)
        self.assertEqual(delta(alpha=1, signed=False, m_levels=8), 1/8)

    def test_max_value(self):
        self.assertEqual(max_value(alpha=1, m_levels=8, signed=True), 3/4)
        self.assertEqual(max_value(alpha=1, m_levels=8, signed=False), 7/8)


if __name__ == '__main__':
    unittest.main()
