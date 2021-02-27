import typing

import keras
import numpy as np
import tensorflow as tf

import pixel_certainty


class Dataset:
    """
        A set of data for training and evaluating a neural network model.
    """

    def __init__(self, x_train, y_train, x_test, y_test):
        self._x_train = tf.convert_to_tensor(x_train)
        self._y_train = tf.convert_to_tensor(y_train)
        self._x_test = tf.convert_to_tensor(x_test)
        self._y_test = tf.convert_to_tensor(y_test)

    def x_train(self) -> tf.Tensor:
        """
            :return: The training input vectors.
        """
        return self._x_train

    def y_train(self) -> tf.Tensor:
        """
            :return: The training output label vectors.
        """
        return self._y_train

    def x_test(self) -> tf.Tensor:
        """
            :return: The test (evaluation) input vectors.
        """
        return self._x_test

    def y_test(self) -> tf.Tensor:
        """
            :return: The test (evaluation) output label vectors.
        """
        return self._y_test

    def transform_x(
            self, callback: typing.Callable[[tf.Tensor], tf.Tensor])\
            -> 'Dataset':
        """
            Creates a new dataset with the x (input) tensors transformed by the
            provided callback.

            :param callback: The callback to apply to the x_test and x_train
                tensors.
            :return: A new Dataset containing the transformed tensors.
        """
        return Dataset(
            callback(self._x_train), self._y_train, callback(self._x_test),
            self._y_test)

    def transform_y(
            self, callback: typing.Callable[[tf.Tensor], tf.Tensor])\
            -> 'Dataset':
        """
            Creates a new dataset with the y (output) values transformed by the
            provided callback.

            :param callback: The callback to apply to the y_test and y_train
                values.
            :return: A new Dataset containing the transformed values.
        """
        return Dataset(
            self._x_train, callback(self._y_train), self._x_test,
            callback(self._y_test))


def combine(data: typing.List[Dataset]) -> Dataset:
    """
        Combine entries from multiple datasets in randomly shuffled order. Test
        and training data is kept separated.
    """
    combined = []
    for sequence_name in ('x_train', 'y_train', 'x_test', 'y_test'):
        sequences = list(map(
            lambda dataset: getattr(dataset, sequence_name)(),
            data))
        combined.append(tf.concat(sequences, 0))
    for pair in ((0, 1), (2, 3)):
        indices = tf.random.shuffle(range(combined[pair[0]].shape[0]))
        for index in pair:
            combined[index] = tf.gather(combined[index], indices)
    return Dataset(*combined)


def mnist_data() -> Dataset:
    """
        :return: The MNIST dataset (http://yann.lecun.com/exdb/mnist/) of
            images of numerals 0 to 9 in 28 x 28 pixels greyscale (0.0 to 1.0).
    """
    number_of_classes = 10
    if not hasattr(mnist_data, 'dataset'):

        def x_transform(data):
            new_shape = list(data.shape)
            new_shape.append(1)
            return tf.reshape(data, new_shape) / np.float32(255)

        def y_transform(data):
            return keras.utils.to_categorical(data, number_of_classes)

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        mnist_data.dataset = (
            Dataset(x_train, y_train, x_test, y_test)
            .transform_x(x_transform)
            .transform_y(y_transform)
        )
    return mnist_data.dataset


def mnist_data_with_certainty() -> Dataset:
    """
        :return: The MNIST dataset (see mnist_data) with an additional channel
            added to the last axis with value 1.0
    """
    return mnist_data().transform_x(pixel_certainty.add_certainty_channel)


def damaged_data(
        binary_damage: bool = True, adversarial_damage: bool = True
        ) -> Dataset:
    """
        The MNIST dataset (see mnist_data) with an additional "certainty"
        channel added and then "damaged" (see
        pixel_certainty.damage_certainty).

        :return: The "damaged" Dataset.
    """
    dataset = mnist_data()
    x_train_shuffled = tf.random.shuffle(dataset.x_train())
    dataset = dataset.transform_x(pixel_certainty.add_certainty_channel)

    def apply_damage(data, binary_damage, adversarial_damage):
        if binary_damage:
            damage_tensor = tf.cast(
                tf.random.uniform(
                    shape=x_train_shuffled.shape, minval=0, maxval=2,
                    dtype=tf.int32),
                tf.float32)
        else:
            damage_tensor = None
        if adversarial_damage:
            damage_data = x_train_shuffled
        else:
            damage_data = None
        return pixel_certainty.damage_certainty(
            data, damage_tensor=damage_tensor, background_tensor=damage_data)

    return Dataset(
        apply_damage(dataset.x_train(), binary_damage, adversarial_damage),
        dataset.y_train(),
        apply_damage(dataset.x_test(), False, False),
        dataset.y_test()
    )


def baseline_data():
    """
        Returns a dataset where the pixel certainty is uniformly 0.0, the y
        values are equiprobable across all categories, and the pixel values
        are taken from the mnist dataset.

        This data assumes an equiprobable choice for unknown data is
        appropriate, rather than reflexting the distribution in the original
        dataset, if it is not balanced between all categories.

        :return: The "baseline" Dataset.
    """
    return (
        mnist_data()
        .transform_x(
            lambda data : pixel_certainty.add_certainty_channel(data, 0.))
        .transform_y(lambda data: tf.fill(data.shape, 1. / data.shape[-1]))
    )


def mixed_data():
    """
        A combination of full-certainty mnist data, adversarially damaged data,
        and baseline data.
    """
    return combine([
        mnist_data_with_certainty(), damaged_data(False), damaged_data(),
        baseline_data()])