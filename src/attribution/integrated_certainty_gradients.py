"""
    This module provides attribution methods based on Integrated Gradient using
    Artificial Uncertainty baselines.
"""

import typing

import keras
import tensorflow as tf
import numpy as np

import attribution.integrated_gradients as integrated_gradients


def certainty_aware_simple_integrated_gradients(
        images: tf.Tensor, model: keras.Model,
        baseline_value: typing.Union[float, tf.Tensor],
        output_class: int = None) -> tf.Tensor:
    """
        Perform an integrated gradients attribution where the baseline is
        filled with a single constant value, but the certainty channel is
        excluded from the calculation.
    """
    if (len(images.shape)) != 4:
        raise ValueError(
            'Image input should have shape (sample count, row count, '
            + 'column count, channel count)')
    value_channel_count = images.shape[-1] - 1
    baseline_value = tf.convert_to_tensor(baseline_value)
    if len(baseline_value.shape) == 0:
        baseline_value = tf.reshape(baseline_value, [1])
    elif len(baseline_value.shape) > 1:
        raise ValueError('Only a single channel axis is supported.')
    if baseline_value.shape[0] == 1:
        baseline_value = baseline_value * tf.ones(value_channel_count)
    elif baseline_value.shape[0] != value_channel_count:
        raise ValueError(
            'The baseline must be a single value or a 1d vector with size '
            + 'equal to the number of value channels (last axis size - 1)')
    masked_baseline = tf.stack([baseline_value, [0]], axis=0)
    masked_baseline = tf.reshape(
        masked_baseline, [1, 1, 1, value_channel_count + 1])
    image_mask = tf.concat([tf.zeros([value_channel_count]), [1]], axis=0)
    image_mask = tf.reshape(image_mask, [1, 1, 1, value_channel_count + 1])
    baseline = masked_baseline + (images * image_mask)
    return integrated_gradients.integrated_gradients(
        images, model, baseline, output_class=output_class)


def certainty_aware_double_sided_integrated_gradients(
        images: tf.Tensor, model: keras.Model, minimum: float = 0.,
        maximum: float = 1., output_class: int = None) -> tf.Tensor:
    """
        Performs Integrated Gradients taking an average from the minimum
        (default 0.0) and maximum (default 1.0) baselines.
    """
    lower = certainty_aware_simple_integrated_gradients(
        images, model, minimum, output_class=output_class)
    upper = certainty_aware_simple_integrated_gradients(
        images, model, maximum, output_class=output_class)
    return (lower + upper) / 2.


def image_integrated_certainty_gradients(
        image: tf.Tensor, model: keras.Model, output_class: int = None
        ) -> tf.Tensor:
    """
        Calculate the salience of each pixel to the prediction the model makes
        for each image using the Integrated Uncertainty Gradients method.

        The Integrated Uncertainty Gradients method builds upon the Integrated
        Gradients method (integrated_gradients). Instead of taking a user
        supplied baseline, it operates on data with uncertainty semantics,
        which implies a canonical baseline of maximum uncertainty.

        :param image: The image for which the salience of input pixels to the
            prediction is desired to be known.
        :param model: The model which is making the predictions.
        :param output_class: The class for which the prediciton probability
            attributions are calculated. If None, the class predicted by the
            model for each image is used.
        :return: Greyscale images where the value of each pixel represents the
            amount that the corresponding pixel in the input images contributes
            to the output prediction (its salience).
    """
    zero_confidence_vector = [np.float32(1.)] * image.shape[-1]
    zero_confidence_vector[-1] = np.float32(0.)
    baseline = image * zero_confidence_vector
    return integrated_gradients.integrated_gradients(
        image, model, baseline, output_class=output_class)


def image_expected_certainty_gradients(
        image: tf.Tensor, model: keras.Model, samples: int = 500,
        output_class: int = None) -> tf.Tensor:
    """
        Perform a hybrid Expected Gradients and Integrated Certainty Gradients
        attribution. Use randomly degraded certainties of the input image as
        a baseline.

        :param image: The input vector.
        :param model: The model making the prediction to attribute.
        :param samples: The number of samples to average over.
        :param output_class: Prediction class to calculate attribution for. If
            the argument is None, the predicted (highest value) class will be
            used.
    :return:
    """
    if image.shape[0] == 1:
        raise ValueError(
            'Image argument of image_expected_certainty_gradients should not '
            + "have a 'samples' axis.")
    if len(image.shape) != 3:
        raise ValueError(
            'Image argument of image_expected_certainty_gradients should have '
            + "3 axes: 'row', 'column', 'channel'.")
    baselines_values_shape = tf.concat(
        [[samples], image.shape[0:2], [image.shape[2] - 1]], 0)
    baselines_certainty_shape = tf.concat(
        [[samples], image.shape[0:2], [1]], 0)
    baselines = image * tf.concat(
        [
            tf.ones(baselines_values_shape),
            tf.random.uniform(baselines_certainty_shape),
        ],
        -1)
    return integrated_gradients.integrated_gradients(
        image, model, baselines, True, output_class=output_class)
