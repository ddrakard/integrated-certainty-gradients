import unittest

import tensorflow as tf


def sample_tensor(axes_count: int = 4, channels_count: int = 4) -> tf.Tensor:
    """
        Make a tensor of numbers where the digits correspond to their axes,
        convenient as test data.
    """
    digits = tf.convert_to_tensor(range(1, channels_count + 1))
    result = digits
    for _ in range(axes_count - 1):
        result = tf.expand_dims(result, -1) * 10 + tf.expand_dims(digits, 0)
    return result


class TensorflowTestCase(unittest.TestCase):
    """
        The unittest.TestCase class augmented for convenience with tensorflow.
    """

    def assertTensorEqual(self, first, second, message=None):
        shape_message = 'The provided tensors do not have the same shape'
        if message is None:
            shape_message += '.'
            message = 'The tensors are not equal.'
        else:
            shape_message += ': ' + message
        self.assertEqual(
            tf.convert_to_tensor(first).shape,
            tf.convert_to_tensor(second).shape,
            shape_message)
        self.assertTrue(tf.reduce_all(first == second), message)
