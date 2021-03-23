"""
    The cats_vs_dogs dataset
    (https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)
"""

import tensorflow as tf
import tensorflow_datasets
import keras
from keras.layers.experimental import preprocessing

from data import dataset


def cats_vs_dogs() -> dataset.Dataset:
    """
        :return: The cats_vs_dogs dataset
            (https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)
    """
    number_of_classes = None
    (train, test, _), metadata = tensorflow_datasets.load(
        'cats_vs_dogs',
        split=['train[:70%]', 'train[70%:80%]', 'train[80%:]'],
        with_info=True,
        as_supervised=True,
    )
    number_of_classes = metadata.features['label'].num_classes

    def format_data(x, y):
        normaliser = tf.keras.Sequential([
            preprocessing.Resizing(180, 180),
            preprocessing.Rescaling(1. / 255)
        ])
        return normaliser(x), tf.one_hot(y, number_of_classes)

    return dataset.Dataset(train, test).map(format_data)


def cats_vs_dogs_augmented() -> dataset.Dataset:
    """
        cats_vs_dogs dataset with data augmentation including horizontal flip,
        rotation, and zoom.
    """
    mapping = keras.Sequential([
        preprocessing.RandomFlip("horizontal", input_shape=(180, 180, 3)),
        preprocessing.RandomRotation(0.1),
        preprocessing.RandomZoom(0.1),
    ])
    return cats_vs_dogs().batch(10).map_x(mapping).unbatch()
