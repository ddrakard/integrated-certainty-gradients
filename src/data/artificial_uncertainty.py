"""
    Tools for creating certainty-aware and certainty training datasets.
"""

import tensorflow as tf

import data.dataset as dataset
import pixel_certainty


def with_certainty(data: dataset.Dataset) -> dataset.Dataset:
    """
        :return: The input dataset with an additional channel added to the last
        axis with value 1.0
    """
    return data.map_x(pixel_certainty.add_certainty_channel)


def damaged(
        data: dataset.Dataset, binary_damage: bool = True,
        adversarial_damage: bool = True, variable_extents: bool = False
        ) -> dataset.Dataset:
    """
        The input with an additional "certainty" channel added and then
        "damaged" (see pixel_certainty.damage_certainty).

        :return: The "damaged" Dataset.
    """

    def damage_tensor(image):
        if not binary_damage:
            return None
        images_shape = tf.shape(image)
        if variable_extents:
            extents_shape = tf.concat(
                [images_shape[0:1], tf.convert_to_tensor([1, 1, 1])], 0)
            damage_extents = tf.random.uniform(
                extents_shape, 0., 1., tf.float16)
        else:
            damage_extents = 0.5
        pixels_shape = tf.concat(
            [images_shape[0:-1], tf.convert_to_tensor([1])], 0)
        result = tf.less(
            tf.random.uniform(
                shape=pixels_shape, minval=0., maxval=1.,
                dtype=tf.float16),
            damage_extents)
        return tf.cast(result, tf.float32)

    def adversarial_damage_tensors(images):
        background = tf.random.shuffle(images)
        return pixel_certainty.damage_certainty(
            pixel_certainty.add_certainty_channel(images),
            damage_tensor(images),
            background_tensor=background)

    if adversarial_damage:
        return (
            data
            .batch(500)
            .map_x(adversarial_damage_tensors)
            .unbatch()
        )
    else:
        return data.map_x(
            lambda image: pixel_certainty.damage_certainty(
                pixel_certainty.add_certainty_channel(image),
                damage_tensor(image)))


def baselines(data: dataset.Dataset) -> dataset.Dataset:
    """
        Returns a dataset where the pixel certainty is uniformly 0.0, the y
        values are equiprobable across all categories, and the pixel values
        are taken from the mnist dataset.

        This data assumes an equiprobable choice for unknown data is
        appropriate, rather than reflexting the distribution in the original
        dataset, if it is not balanced between all categories.

        :return: The "baseline" Dataset.
    """
    def transform_data(x, y):
        return (
            pixel_certainty.add_certainty_channel(x, 0.),
            tf.fill(y.shape, 1. / y.shape[-1])
        )
    return data.map(transform_data)


def mixed_data(
        data: dataset.Dataset, undamaged: bool = True,
        binary_damage: bool = True, intermediate_damage: bool = True,
        baseline: bool = True
        ) -> dataset.Dataset:
    """
        A combination of full-certainty input data, adversarially damaged data,
        and baseline data.
    """
    datasets = []
    if undamaged:
        datasets.append(with_certainty(data))
    if binary_damage:
        datasets.append(damaged(data))
    if intermediate_damage:
        datasets.append(damaged(data, False))
    if baseline:
        datasets.append(baselines(data))
    return dataset.combine(datasets)
