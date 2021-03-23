"""
    Demonstration that attribution is suppressed in areas of low certainty.
"""

import tensorflow as tf
import keras

import tensor_tools
import image_tensors
import pixel_certainty
import attribution.integrated_gradients as integrated_gradients


def check_uncertainty_baseline_attribution(
        image: tf.Tensor, model: keras.Model, band_top: int, band_width: int
        ) -> None:
    """
        Display images that show attribution is suppressed if certainty is
        reduced.

        :param image: The input image to calculate attribution for.
        :param model: The model to calculate attribution for.
        :param band_top: The distance from the top of the image at which to
            start suppressing certainty.
        :param band_width: The distance below band_top over which to suppress
            certainty.
    """
    band_bottom = band_top + band_width
    if band_bottom > image.shape[-3]:
        raise ValueError('Certainty ablation band is outside tensor volume.')
    image = (
        tensor_tools
        .Selection()[..., band_top:band_bottom, :, -1:]
        .multiplex(tf.zeros_like(image, tf.float32), image)
    )
    baseline_zero = tensor_tools.Selection()[..., 0:-1].multiplex(
        tf.zeros_like(image, tf.float32), image)
    attribution_zero = integrated_gradients.integrated_gradients(
        image, model, baseline_zero)
    baseline_one = tensor_tools.Selection()[..., 0:-1].multiplex(
        tf.ones_like(image, tf.float32), image)
    attribution_one = integrated_gradients.integrated_gradients(
        image, model, baseline_one)
    attribution_combined = attribution_one + attribution_zero
    (
        image_tensors.ImagePlot()
        .add_two_channel_positive_saturated(
            tf.squeeze(pixel_certainty.collapse_value_channels(image)),
            title='Image')
        .add_two_channel_positive_saturated(
            tf.squeeze(
                pixel_certainty.collapse_value_channels(baseline_zero)),
            title='Baseline Zero')
        .add_single_channel(
            tf.squeeze(attribution_zero), True, title='Attribution Zero')
        .add_two_channel_positive_saturated(
            tf.squeeze(
                pixel_certainty.collapse_value_channels(baseline_one)),
            title='Baseline One')
        .add_single_channel(
            tf.squeeze(attribution_one), True, title='Attribution One')
        .add_single_channel(
            tf.squeeze(attribution_combined), True,
            title='Attribution Combined')
        .show()
    )
