"""
    Utilities for working with images having an uncertainty channel (at the
    last channel in their last axis).
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability

import tensor_tools


def add_certainty_channel(
        images: tf.Tensor, certainty_value: float = 1.0,
        add_axis: bool = False) -> tf.Tensor:
    """
        Add an additional channel to the last axis of a tensor to represent the
        certainty of the values.

        :param images: A Tensor representing images with pixel values in the
            last axis.
        :param certainty_value: The value of elements in the certainty channel.
        :param add_axis: Add a new axis to the end of axes to insert the
            certainty channel into.
        :return: A Tensor with an additional channel in the last axis.
    """
    if add_axis:
        images = tf.expand_dims(images, -1)
    pads = ([(0, 0)] * (len(images.shape) - 1)) + [(0, 1)]
    return tf.pad(images, pads, constant_values=np.float32(certainty_value))


def disregard_certainty(image, new_certainty=np.float32(1.)):
    """
        Sets the certainty channel (the last channel of the last axis) to 1.0,
        or a constant given certainty.

        :param image: The image or images tensor to transform.
        :param new_certainty: The new certainty to set every pixel to.
        :return: The modified tensor.
    """
    certainty_channel = tensor_tools.Selection()[..., -1:None]
    return certainty_channel.multiplex(
        tf.fill(image.shape, new_certainty), image)


def discard_certainty(images: tf.Tensor) -> tf.Tensor:
    """
        Removes the certainty channel (the last channel of the last axis) from
        the images.

        :param images: The tensor to remove the certainty channel from.
        :return: An equivalent tensor without a certainty channel.
    """
    return tf.gather(images, range(0, images.shape[-1] - 1), axis=-1)


def discard_value(image: tf.Tensor) -> tf.Tensor:
    """
        Removes all channels except the last one (the certainty channel) of the
        last axis.
    """
    return image[..., -1:]


def collapse_value_channels(images: tf.Tensor) -> tf.Tensor:
    """
         Collapse the value (non-certainty) channels down to a single greyscale
         channel.
    """
    value = discard_certainty(images)
    certainty = discard_value(images)
    greyscale = tf.reduce_mean(value, axis=-1, keepdims=True)
    return tf.concat([greyscale, certainty], axis=-1)


def collapse_certainty_to_brightness(images: tf.Tensor) -> tf.Tensor:
    """
        Scale the brightness of each pixel by its certainty and discard the
        certainty channel.
    """
    value = discard_certainty(images)
    certainty = discard_value(images)
    return value * certainty


def certainty_mild_damage_distribution(
        damage_severity: float = 0.2
        ) -> tensorflow_probability.distributions.Distribution:
    """
        Provides a distribution from which to sample damage amounts for use
        with damage_certainty, with a mild to moderate amount of confidence
        damage.

        :param damage_severity: A larger value results in more damage.
        :return: A tensorflow_probability.distributions.Distribution to sample
            damage amounts between 0.0 and 1.0 from.
    """
    return tensorflow_probability.distributions.TruncatedNormal(
        np.float32(1.), damage_severity, np.float32(0.), np.float32(1.))


def certainty_high_damage_distribution(
        certainty_residue: float = 0.5
        ) -> tensorflow_probability.distributions.Distribution:
    """
        Provides a distribution from which to sample damage amounts for use
        with damage_certainty, this distribution results in mostly damage.

        :param certainty_residue: A larger value results in less damage.
        :return: A tensorflow_probability.distributions.Distribution to sample
            damage amounts between 0.0 and 1.0 from.
    """
    return tensorflow_probability.distributions.TruncatedNormal(
        np.float32(0.), certainty_residue, np.float32(0.), np.float32(1.))


def damage_certainty(images: tf.Tensor, damage_tensor: tf.Tensor = None,
                     background_tensor: tf.Tensor = None) -> tf.Tensor:
    """
        Applies "Damage" to the certainty of the images and their corresponding
        values according to the damage_tensor, where the damaged values are
        taken from background_tensor.

        "Damage" is applied by first randomly reducing the certainty channel of
        each pixel in proportion to the damage_tensor. Then the value channel
        of each element may be replaced with the corresponding value from the
        background_tensor with probability equal to one minus the damage_tensor
        value for that element. The "value" (first channel) of each pixel may
        be replaced with the equivalent value from background_tensor with
        probability one minus the certainty damage.

        :param images: A Tensor of images to damage.
        :param damage_tensor: A Tensor containing the amount to damage each
            pixel, between 0.0 and 1.0, with 0.0 being the most damage (least
            resulting certainty). If it is not supplied, each pixel has a
            random amount of damaged taken from
            certainty_mild_damage_distribution.
        :param background_tensor: A Tensor containing the values that damaged
            pixels may be replaced with. If it is not supplied, pixels are
            replaced with a random value between 0.0 and 1.0.
        :return: The damaged images tensor.
    """
    value_data = discard_certainty(images)
    if damage_tensor is None:
        damage_tensor = certainty_mild_damage_distribution(
            np.float32(1.0)).sample(tf.shape(value_data))
    if background_tensor is None:
        background_tensor = tf.random.uniform(tf.shape(value_data))
    channel_count = images.shape[-1] - 1  # excluding the certainty channel
    certainty_data = tf.gather(images, [channel_count], axis=-1)
    damaged_certainty = certainty_data * damage_tensor
    random_tensor = tf.random.uniform(tf.shape(certainty_data))
    damaged_values = tf.where(
        tf.greater(random_tensor, damage_tensor),
        background_tensor, value_data)
    return tf.concat([damaged_values, damaged_certainty], -1)
