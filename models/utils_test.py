import unittest
import tensorflow as tf
import numpy as np
from .utils import *

class TestBinaryConversion(unittest.TestCase):

    def test_conversion(self):
        x = tf.random.uniform((1, 64, 128))
        tmp = tensor_to_binary(x)
        y = binary_to_tensor(tmp)
        y = tf.reshape(y, (1, 64, 128))

        x = tf.bitcast(x, tf.uint32).numpy()
        y = tf.bitcast(y, tf.uint32).numpy()

        # bit-level exact match
        np.testing.assert_array_equal(x, y)

    def test_conversion_v2(self):
        shape = (1, 64, 128)
        x = tf.random.uniform(shape)
        tmp = tensor_to_binary_v2(x)
        y = binary_to_tensor_v2(tmp)
        y = tf.reshape(y, shape)

        x = tf.bitcast(x, tf.uint32).numpy()
        y = tf.bitcast(y, tf.uint32).numpy()

        # bit-level exact match
        # np.testing.assert_almost_equal(x, y)
        np.testing.assert_array_equal(x, y)

    def test_tensor_to_binary(self):
        # IEEE754
        # f32: 6027202.5
        # hex: 0x4ab7ef85
        # bin: 01001010101101111110111110000101
        x = tf.constant([6027202.5])
        bin = tensor_to_binary(x)
        actual = bin.numpy().astype(np.int32).flatten()
        expected = np.array([
            0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 0, 0, 0, 0, 1, 0, 1
        ], dtype=np.int32)[::-1]
        self.assertTrue((actual == expected).all())
    
    def test_tensor_to_binary_v2(self):
        # IEEE754
        # f32: 6027202.5
        # hex: 0x4ab7ef85
        # bin: 01001010101101111110111110000101
        x = tf.constant([6027202.5])
        bin = tensor_to_binary_v2(x)
        actual = bin.numpy().astype(np.int32).flatten()
        expected = np.array([
            0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 0, 0, 0, 0, 1, 0, 1
        ], dtype=np.int32)[::-1]
        np.testing.assert_equal(actual, expected)


    def test_binary_to_tensor(self):
        # IEEE754
        # f32: 6027202.5
        # hex: 0x4ab7ef85
        # bin: 01001010101101111110111110000101
        x = np.array([
            0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 0, 0, 0, 0, 1, 0, 1
        ],
                     dtype=np.float32)[::-1]
        x = tf.constant(x)
        bin = binary_to_tensor(x)
        actual = bin.numpy()[0]
        expected = 6027202.5

        self.assertAlmostEqual(actual, expected)
    
    def test_binary_to_tensor_v2(self):
        # IEEE754
        # f32: 6027202.5
        # hex: 0x4ab7ef85
        # bin: 01001010101101111110111110000101
        x = np.array([
            0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 0, 0, 0, 0, 1, 0, 1
        ],
                     dtype=np.float32)[::-1]
        x = tf.constant(x)
        bin = binary_to_tensor_v2(x)
        actual = bin.numpy()[0]
        expected = 6027202.5
        self.assertAlmostEqual(actual, expected)

    def test_binary_to_tensor_with_tanh_v2(self):
        # f32: inf
        # hex: 0x7f800000
        # bin: 01111111100000000000000000000000
        x = np.array([0] + [1 for i in range(8)] + [0 for i in range(23)], 
                dtype=np.float32)[::-1]
        x = tf.constant(x)
        y = tf.math.tanh(binary_to_tensor_v2(x))
        actual = y.numpy()[0]
        expected = 1.0
        self.assertAlmostEqual(actual, expected)
        # f32: -inf
        # hex: 0xff800000
        # bin: 11111111100000000000000000000000
        x = np.array([1] + [1 for i in range(8)] + [0 for i in range(23)], 
                dtype=np.float32)[::-1]
        x = tf.constant(x)
        y = tf.math.tanh(binary_to_tensor_v2(x))
        actual = y.numpy()[0]
        expected = -1.0
        self.assertAlmostEqual(actual, expected)
        # f32: nan 
        # hex: 0x7fc00000
        # bin: 01111111110000000000000000000000
        x = np.array([0] + [1 for i in range(8)] + [1] + [0 for i in range(22)], 
                dtype=np.float32)[::-1]
        x = tf.constant(x)
        y = tf.math.tanh(binary_to_tensor_v2(x))
        actual = y.numpy()[0]
        expected = 1.0
        self.assertAlmostEqual(actual, expected)
        # f32: -nan 
        # hex: 0xffc00000
        # bin: 11111111110000000000000000000000
        x = np.array([1] + [1 for i in range(8)] + [1] + [0 for i in range(22)], 
                dtype=np.float32)[::-1]
        x = tf.constant(x)
        y = tf.math.tanh(binary_to_tensor_v2(x))
        actual = y.numpy()[0]
        expected = -1.0
        self.assertAlmostEqual(actual, expected)
        


    def test_replace_nan(self):
        x = np.array([
            [[1, 2, np.nan],
             [4, np.nan, 6],],
            [[7, 8, np.nan],
             [10, np.nan, 12],],
        ], dtype=np.float32)
        expected = np.array([
            [[1, 2, 0.0],
             [4, 0, 6],],
            [[7, 8, 0],
             [10, 0, 12],],
        ], dtype=np.float32)

        actual = replace_nan(tf.constant(x))
        actual = actual.numpy()

        np.testing.assert_almost_equal(actual, expected)

    def test_replace_nan_to_inf(self):
        x = np.array([
            [[1, 2, np.nan],
             [4, -np.nan, np.inf],],
            [[7, 8, -np.nan],
             [10, np.nan, -np.inf],],
        ], dtype=np.float32)
        expected = np.array([
            [[1, 2, np.inf],
             [4, -np.inf, np.inf],],
            [[7, 8, -np.inf],
             [10, np.inf, -np.inf],],
        ], dtype=np.float32)

        actual = replace_nan_to_inf(tf.constant(x))
        actual = actual.numpy()

        np.testing.assert_almost_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
