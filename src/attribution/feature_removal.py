"""
    Feature attribution methods based on 'removing' / 'ablating' individual
    features.
"""

import typing

import numbers
import tensorflow as tf
import numpy as np
import keras

import pixel_certainty
import tensor_tools


def simple_feature_removal(
        data: tf.Tensor, model: keras.Model,
        baseline: typing.Union[float, tf.Tensor] = 0.) -> tf.Tensor:
    """
        Calculate an attribution by replacing each value with a value taken
        from a baseline, and returning the effect the replacement has on the
        probability of the predicted class.

        :param data: The data for which an attribution is to be made.
        :param model: The model for which an attribution is to be made. It
            should be a classifier.
        :param baseline: The replacement for each pixel in the data. It may be
            a number, in which case the same value is used everywhere, or a
            Tensor.
        :return: A tensor with the same shape as the data, with values
            indicating what effect replacing the data value at that location
            with the baseline value has on the probability of the predicted
            class.
    """
    if isinstance(baseline, numbers.Number):
        baseline = tf.fill(data.shape, baseline)
    result_shape = data.shape
    data = pixel_certainty.add_certainty_channel(tf.expand_dims(data, axis=0))
    reference_result = model(data)
    predicted_class_index = tf.argmax(reference_result[0])

    def choice_probability(result):
        return tf.gather(
            tf.nn.softmax(result, axis=-1), predicted_class_index, axis=-1)

    reference_probability = choice_probability(reference_result)[0]
    parallel_shape = list(data.shape)
    parallel_shape[0] = data.shape[1] * data.shape[2]
    removed = np.zeros(parallel_shape) + data.numpy()
    index = 0
    for coordinates in tensor_tools.Coordinates(result_shape):
        removed[tuple([index] + coordinates)] = baseline[tuple(coordinates)]
        index += 1
    return tf.reshape(
        reference_probability - choice_probability(model(removed)),
        result_shape)


def double_sided_feature_removal(
        data: tf.Tensor, model: keras.Model) -> tf.Tensor:
    """
        Calculate an attribution by replacing each value with a minimum (0.0)
        value and then a maximum (1.0) value, and returning the sum of the
        effects each have on the probability of the predicted class.

        :param data: The data for which an attribution is to be made.
        :param model: The model for which an attribution is to be made. It
            should be a classifier.
        :return: A tensor with the same shape as the data, with values
            indicating what effect replacing the data value at that location
            has on the probability of the predicted class.
    """
    high = simple_feature_removal(data, model)
    low = simple_feature_removal(data, model, 1.0)
    return (high + low) / 2.0


def feature_certainty_removal(
        data: tf.Tensor, model: keras.Model) -> tf.Tensor:
    """
        Calculate an attribution by replacing the certainty at each point with
        0.0 and returning the effects the replacements have on the probability
        of the predicted class.
    """
    result_shape = list(data.shape)
    result_shape[-1] = 1
    data = tf.expand_dims(data, axis=0)
    reference_result = model(data)
    predicted_class_index = tf.argmax(reference_result[0])

    def choice_probability(result):
        return tf.gather(
            tf.nn.softmax(result, axis=-1), predicted_class_index, axis=-1)

    reference_probability = choice_probability(reference_result)[0]
    parallel_shape = list(data.shape)
    parallel_shape[0] = data.shape[1] * data.shape[2]
    removed = np.zeros(parallel_shape) + data.numpy()
    index = 0
    for coordinates in tensor_tools.Coordinates(result_shape):
        coordinates[-1] = 1
        removed[tuple([index] + coordinates)] = 0.
        index += 1
    return tf.reshape(
        reference_probability - choice_probability(model(removed)),
        result_shape)
