"""
    The MNIST dataset (http://yann.lecun.com/exdb/mnist/) of
    images of numerals 0 to 9 in 28 x 28 pixels greyscale (0.0 to 1.0).
"""

import keras
import numpy as np
import tensorflow as tf
import data.dataset as dataset


def mnist_data() -> dataset.Dataset:
    """
        :return: The MNIST dataset (http://yann.lecun.com/exdb/mnist/) of
            images of numerals 0 to 9 in 28 x 28 pixels greyscale (0.0 to 1.0).
    """
    number_of_classes = 10

    def transform(x, y):
        new_shape = list(x.shape)
        new_shape.append(1)
        return (
            tf.reshape(x, new_shape) / np.float32(255),
            tf.one_hot(y, number_of_classes)
        )

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    return (
        dataset.Dataset.from_tensors(x_train, y_train, x_test, y_test)
        .map(transform)
    )
