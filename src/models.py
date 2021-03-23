"""
    Some simple, easy to train, Keras model architectures.
"""

import tensorflow.keras.layers as layers
import tensorflow as tf


def greyscale_mnist_classifier() -> tf.keras.Model:
    """ Crate a model to classify 28 x 28 pixel greyscale images. """
    number_of_classes = 10
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(
        32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(number_of_classes, activation='softmax'))
    return model


def value_confidence_mnist_classifier() -> tf.keras.Model:
    """
        Crate a model to classify 28 x 28 pixel two channel (value and
        confidence) images.
    """
    number_of_classes = 10
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(
        32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(number_of_classes, activation='softmax'))
    return model


def value_confidence_flowers_classifier() -> tf.keras.Model:
    """
        Crate a model to classify 180 x 180 pixel 4 channel (3 x value and 1 x
        confidence) images.
    """
    height = 180
    width = 180
    number_of_classes = 5
    return tf.keras.models.Sequential([
        layers.Conv2D(
            16, 4, padding='same', activation='relu',
            input_shape=(height, width, 4)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(number_of_classes, activation='softmax')
    ])


def value_confidence_cats_vs_dogs() -> tf.keras.Model:
    """
        Crate a model to classify 180 x 180 pixel 4 channel (3 x value and 1 x
        confidence) images.
    """
    height = 180
    width = 180
    number_of_classes = 2
    return tf.keras.models.Sequential([
        layers.Conv2D(
            16, 4, padding='same', activation='relu',
            input_shape=(height, width, 4)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(number_of_classes, activation='softmax')
    ])
