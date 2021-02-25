from typing import List

import keras
import numpy as np
import tensorflow as tf

import tensor_tools


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
    if class_names is not None:
        raise NotImplementedError('class_names parameter is not implemented.')
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
        images: tf.Tensor, model: keras.Model, baseline: tf.Tensor,
        monte_carlo: bool = False, samples: int = None,
        output_class: int = None) -> tf.Tensor:
    """
        Calculate the salience of each pixel to the prediction the model makes
        for each image using the Integrated Gradients method.

        The integrated gradients method, introduced
        https://arxiv.org/pdf/1703.01365.pdf, calculates an approximated
        integral of the gradient of the model output with respect to the input
        value, from a baseline image to the subject image at each pixel,
        approximating the contribution of that pixel as a Shapley value to the
        overall prediction.

        Using multiple baselines and monte_carlo is equivalent to the Expected
        Gradients method https://arxiv.org/pdf/1906.10670.pdf.

        :param images: The images for which the salience of input pixels to the
            prediction is desired to be known, as a Tensor with shape
            ([image index], pixel row, pixel column, channel).
        :param model: The model which is making the predictions.
        :param baseline: An image to use as the "neutral" image from which to
            start the integration, as a Tensor with shape
            ([baseline index], pixel row, pixel column, channel). If the
            baseline index is present, an average of attribution over the
            baselines will be taken.
        :param monte_carlo: By default this function uses Riemann sums
            to calculate the gradient integral. If this argument is True,
            Monte Carlo integration will be used instead.
        :param samples: How many samples to average over. If this is None,
            random baseline sampling will be disabled and each baseline will
            be used once. If monte_carlo is enabled, a random interpolation
            will be used for each sample, the total number of gradient
            calculations will be equal to this argument (or the number of
            baselines if this argument is None). If monte_carlo is disabled,
            each baseline will have 50 equally spaced interpolations, and the
            total number of gradient calculations will be equal to 50 times
            this argument (or fifty times the number of baselines if this
            argument is None).
        :param output_class: The class for which the prediciton probability
            attributions are calculated. If None, the class predicted by the
            model for each image is used.
        :return: Greyscale images where the value of each pixel represents the
            amount that the corresponding pixel in the input images contributes
            to the output prediction (its salience).
    """
    class_count = 10
    if len(images.shape) == 3:
        images = tf.expand_dims(images, axis=0)
    if len(baseline.shape) == 3:
        baseline = tf.expand_dims(baseline, axis=0)
    if samples is None:
        baseline_samples = baseline
    else:
        baseline_samples = tensor_tools.pick(baseline, samples)
    # Axes: (image, baseline_sample, interpolation, row, column, channel)
    image_count = images.shape[0]
    baseline_sample_count = baseline_samples.shape[0]
    if monte_carlo:
        interpolation_count = 1
    else:
        interpolation_count = 5
    row_count = images.shape[1]
    column_count = images.shape[2]
    channel_count = images.shape[3]
    evaluations_shape = [
        image_count, baseline_sample_count, interpolation_count]
    image_shape = [row_count, column_count, channel_count]
    interpolations_placeholder = tf.ones(
        [interpolation_count] + image_shape, tf.float16)
    # TODO: Allocating new axis but not tiling them to the full size
    # (broadcasting lazily) would save some unnecesary operations in later
    # steps.
    (broadcast_images, broadcast_baseline_samples, _) = \
        tensor_tools.axis_outer_operation(
            0, [images, baseline_samples, interpolations_placeholder],
            lambda tensors: tensors)
    if monte_carlo:
        interpolations = tf.random.uniform(broadcast_images.shape, 0., 1.)
    else:
        start = 0.5 / interpolation_count
        stop = 1. - (0.5 / interpolation_count)
        # Left un-broadcasted
        interpolations = tf.linspace(
            [[[[[start]]]]], [[[[[stop]]]]], interpolation_count, axis=2)
    deltas = broadcast_images - broadcast_baseline_samples
    interpolated_images = broadcast_baseline_samples + deltas * interpolations
    if output_class is None:
        predictions = tf.argmax(model(images), axis=-1)
    else:
        predictions = tf.fill([images.shape[0]], output_class)
    with tf.GradientTape() as tape:
        tape.watch(interpolated_images)
        unstructured_shape = [np.prod(evaluations_shape)] + image_shape
        logits = model(tf.reshape(interpolated_images, unstructured_shape))
        unstructured_probabilities = tf.nn.softmax(logits, axis=-1)
        structured_probabilities = tf.reshape(
            unstructured_probabilities, evaluations_shape + [class_count])
        prediction_indices = np.zeros(evaluations_shape + [4], np.int32)
        for coordinates in tensor_tools.Coordinates(evaluations_shape):
            prediction = predictions[coordinates[0]]
            prediction_indices[tuple(coordinates)] = coordinates + [prediction]
        prediction_probabilities = tf.gather_nd(
            structured_probabilities, prediction_indices)
        gradients = tape.gradient(
            prediction_probabilities, interpolated_images)
    partial_gradients = tf.einsum('...i,...i->...', deltas, gradients)
    return tf.math.reduce_mean(partial_gradients, axis=[1, 2])


def simple_integrated_gradients(
        images: tf.Tensor, model: keras.Model, baseline_value,
        output_class: int = None) -> tf.Tensor:
    """
        Performs Integrated Gradients where the same baseline value is used for
        each pixel.
    """
    if (len(images.shape)) != 4:
        raise ValueError(
            'Image input should have shape (sample count, row count, '
            + 'column count, channel count)')
    baseline_value = tf.convert_to_tensor(baseline_value)
    if len(baseline_value.shape) > 1:
        raise ValueError('Only a single channel axis is supported.')
    if len(baseline_value.shape) == 0:
        baseline_value = tf.reshape(baseline_value, [1])
    baseline_value = tf.reshape(
        baseline_value, [1, 1, 1] + list(baseline_value.shape))
    baseline_shape = list(images.shape)
    baseline_shape[0] = 1
    # Broadcast together
    baseline = tf.zeros(baseline_shape, baseline_value.dtype) + baseline_value
    return integrated_gradients(
        images, model, baseline, output_class=output_class)


def double_sided_integrated_gradients(
        images: tf.Tensor, model: keras.Model, minimum=0., maximum=1.,
        output_class: int = None) -> tf.Tensor:
    """
        Performs Integrated Gradients taking an average from the minimum
        (default 0.0) and maximum (default 1.0) baselines.
    """
    lower = simple_integrated_gradients(
        images, model, minimum, output_class=output_class)
    upper = simple_integrated_gradients(
        images, model, maximum, output_class=output_class)
    return (lower + upper) / 2.


def image_certainty_integrated_gradients(
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
    return integrated_gradients(
        image, model, baseline, output_class=output_class)
