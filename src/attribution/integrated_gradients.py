from typing import List

import keras
import numpy as np
import tensorflow as tf


def classifier_gradients(
        inputs: tf.Tensor, model: keras.Model, class_names: List[str] = None
        ) -> tf.Tensor:
    """
        Calculates the gradients of a classifier prediction with respect to its
        inputs.

        The gradient of the probability of the chosen class with respect to the
        elements of the input vectors is calculated. It is assumed the first
        axis is the sample number and the remaining axes are of the input
        vectors.
        :param inputs: The inputs ("x vectors") to the model.
        :param model: The model classifying the inputs.
        :param class_names: The names associated
        :return: A tensor containing the gradients for each element of each
            input.
    """
    # TODO: Implement class_names
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        logits = model(inputs)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predicted_class_indices = tf.argmax(probabilities, axis=-1)
        predicted_class_indices_as_tensor_coordinates = []
        for index in range(inputs.shape[0]):
            predicted_class_indices_as_tensor_coordinates.append(
                [index, predicted_class_indices[index]])
        predicted_class_probabilities = tf.gather_nd(
            probabilities, predicted_class_indices_as_tensor_coordinates)
        return tape.gradient(predicted_class_probabilities, inputs)


def integrated_gradients(
        images: tf.Tensor, model: keras.Model, baseline: tf.Tensor
        ) -> tf.Tensor:
    """
        Calculate the salience of each pixel to the prediction the model makes
        for each image using the Integrated Gradients method.

        The integrated gradients method, introduced
        https://arxiv.org/pdf/1703.01365.pdf, calculates an approximated
        integral of the gradient of the model output with respect to the input
        value, from a baseline image to the subject image at each pixel,
        approximating the contribution of that pixel as a Shapley value to the
        overall prediction.

        :param images: The images for which the salience of input pixels to the
            prediction is desired to be known.
        :param model: The model which is making the predictions.
        :param baseline: An image to use as the "neutral" image from which to
            start the integration.
        :return: Greyscale images where the value of each pixel represents the
            amount that the corresponding pixel in the input images contributes
            to the output prediction (its salience).
    """

    def interpolate_images(baseline, images, image_count):
        alphas = tf.linspace(start=0.0, stop=1.0, num=image_count)
        alphas_extended = alphas[
            :, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_extended = tf.expand_dims(baseline, axis=0)
        delta = images - baseline_extended
        images = baseline_extended + alphas_extended * delta
        return images

    def compute_gradients(images):
        with tf.GradientTape() as tape:
            tape.watch(images)
            original_shape = images.shape
            image_interpolations_count = original_shape[0]
            source_images_count = original_shape[1]
            # TODO: Reimplement with tensor_tools
            flat_shape = (
                source_images_count * image_interpolations_count,
                original_shape[2], original_shape[3], original_shape[4])
            flat_images = tf.reshape(images, flat_shape)
            logits_shape = (
                image_interpolations_count, source_images_count, 10)
            logits = tf.reshape(model(flat_images), logits_shape)
            probabilities = tf.nn.softmax(logits, axis=-1)

            def predicted_class_index(source_image_index):
                return tf.argmax(
                    (probabilities
                        [image_interpolations_count - 1]
                        [source_image_index]
                     )
                )

            predicted_class_indices = []
            for interpolation_index in range(image_interpolations_count):
                for source_image_index in range(source_images_count):
                    predicted_class_indices.append(
                        [
                            interpolation_index,
                            source_image_index,
                            predicted_class_index(source_image_index)
                        ])
            predicted_class_probabilities = tf.gather_nd(
                probabilities, predicted_class_indices)
            return tape.gradient(predicted_class_probabilities, images)

    if len(images.shape) == 3:
        images = tf.expand_dims(images, axis=0)
    riemann_partition_count = 50
    interpolated_images = interpolate_images(
        baseline, images, riemann_partition_count + 1)
    gradients = compute_gradients(interpolated_images)
    trapezia_heights = (gradients[:-1] + gradients[1:]) / np.float32(2.0)
    mean_gradient = tf.math.reduce_mean(trapezia_heights, axis=0)
    # Dot product in the last axis
    return tf.einsum('...i,...i->...', images - baseline, mean_gradient)


def image_certainty_integrated_gradients(
        image: tf.Tensor, model: keras.Model) -> tf.Tensor:
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
        :return: Greyscale images where the value of each pixel represents the
            amount that the corresponding pixel in the input images contributes
            to the output prediction (its salience).
    """
    zero_confidence_vector = [np.float32(1.)] * image.shape[-1]
    zero_confidence_vector[-1] = np.float32(0.)
    baseline = image * zero_confidence_vector
    return integrated_gradients(image, model, baseline)
