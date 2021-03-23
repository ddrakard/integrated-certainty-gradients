"""
    This module analyses the comparison between various attribution methods.
"""

import typing

import tensorflow as tf

import image_tensors
import tensor_tools


def show_attribution_distribution(
        attribution_function: typing.Callable, dataset: tf.Tensor,
        vectorised: bool = False, samples: int = 100) -> None:
    """
        Displays the average importance of input features according to
        an attribution function.

        Importance for negative attribution, positive attribution, absolute
        attribution and mean contribution are shown.

        :param attribution_function: The function that returns an attribution
            for an input vector.
        :param dataset: The dataset of input vectors.
        :param vectorised: Perform the attribution function in parallel. This
            may use a lot of memory.
        :param samples: The number of samples to average over.
    """
    if vectorised:
        attributions = attribution_function(
            tensor_tools.pick(dataset, samples))
    else:
        attributions = tf.map_fn(
            attribution_function, tensor_tools.pick(dataset, samples))
    (
        image_tensors.ImagePlot()
        .add_single_channel(
           tf.reduce_mean(tf.maximum(attributions, 0.), axis=0),
           True, title='Mean positive')
        .add_single_channel(
           tf.reduce_mean(tf.minimum(attributions, 0.), axis=0),
           True, title='Mean negative')
        .add_single_channel(
           tf.reduce_mean(attributions, axis=0),
           True, title='Mean combined')
        .add_single_channel(
            tf.reduce_mean(tf.abs(attributions), axis=0),
            True, title='Mean absolute')
        .show()
    )
