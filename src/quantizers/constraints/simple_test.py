#!/usr/bin/env python3

import unittest

# def simple_function(alpha: int):
#     print(alpha)
import tensorflow as tf

class SimpleClass:
    def __init__(self, alpha: int):
        self.alpha = alpha
        # -> This I want to be a reference
        # assert alpha is ref
    def __call__(self):
        print(self.alpha)


class TestSimpleFunction(unittest.TestCase):
    # def test_simple_function(self):
    #     alpha = 3
    #     simple_function = SimpleClass(alpha)

    #     simple_function()

    #     alpha = 1
    #     simple_function()

    # def test_simple_function_with_tf_variable(self):
    #     alpha = tf.Variable(3)
    #     simple_function = SimpleClass(alpha)

    #     simple_function()

    #     alpha.assign(1)
    #     simple_function()

    def test_simple_function_with_tf_variable(self):
        alpha = tf.Variable(3)
        simple_function = SimpleClass(-alpha)

        simple_function()

        alpha.assign(1)
        simple_function()
# Algo que se comporte siempre como una referencia

if __name__ == "__main__":
    unittest.main()
