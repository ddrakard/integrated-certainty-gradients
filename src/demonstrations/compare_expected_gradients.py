"""
    Tools to compare various gradient based attribution methods.
"""

import tensorflow as tf
import keras

import image_tensors
import pixel_certainty
from attribution import integrated_gradients, integrated_certainty_gradients


def compare_expected_gradients(
        image: tf.Tensor, model: keras.Model, distribution: tf.Tensor,
        samples_count: int = 500) -> None:
    """
        Compare integrated certainty gradients, expected certainty gradients,
        and integrated gradients for a given image.

        :param image: An image with a certainty channel
        :param model: The classifier model to calculate attribution for.
        :param distribution: The baseline distribution from which the image was
            drawn (can be any images of the same size with certainty channels).
        :param samples_count: The number of samples to use in the
            expectation based methods.
    """
    expected_gradients = integrated_gradients.integrated_gradients(
        image, model, distribution, True, samples_count)
    expected_certainty_gradients = \
        integrated_certainty_gradients.image_expected_certainty_gradients(
            image, model, samples_count)
    certainty_gradients = \
        integrated_certainty_gradients.image_integrated_certainty_gradients(
            image, model)
    greyscale_image = image_tensors.rgb_to_greyscale(
        pixel_certainty.discard_certainty(image))
    (
        image_tensors.ImagePlot()
        .add_single_channel(tf.squeeze(expected_gradients), True)
        .add_single_channel(tf.squeeze(expected_certainty_gradients), True)
        .add_single_channel(tf.squeeze(certainty_gradients), True)
        .new_row()
        .add_overlay(
            greyscale_image,
            tf.expand_dims(tf.squeeze(expected_gradients), -1),
            True
        )
        .add_overlay(
            greyscale_image,
            tf.expand_dims(tf.squeeze(expected_certainty_gradients), -1),
            True
        )
        .add_overlay(
            greyscale_image,
            tf.expand_dims(tf.squeeze(certainty_gradients), -1),
            True
        )
        .add_rgb_image(pixel_certainty.discard_certainty(image))
        .show()
    )
