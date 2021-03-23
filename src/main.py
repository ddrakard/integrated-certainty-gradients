"""
    Demonstration of the use of the Integrated Certainty Gradients feature
    attribution method and related methods and analysis.
"""

import os
import sys
from pathlib import Path

import keras
import pandas
import tensorflow as tf

import attribution.feature_removal as feature_removal
import attribution.integrated_certainty_gradients \
    as integrated_certainty_gradients
import attribution.integrated_gradients as integrated_gradients
import data.artificial_uncertainty
import data.mnist as mnist
import image_tensors
import model_tools
import pixel_certainty
import tensor_tools
from data import cats_vs_dogs
from pixel_certainty import discard_certainty

if __name__ == '__main__':

    TRAIN_MODEL = False
    DISPLAY_IMAGES = True

    SOURCE_DIRECTORY = Path(__file__).resolve().parent
    sys.path.insert(0, str(SOURCE_DIRECTORY))
    os.chdir(SOURCE_DIRECTORY.parent)

    MODEL = model_tools.load_latest_model('models/active/model')

    def train_model():
        """
            Train a model with artificial uncertainty semantics.
        """
        generations = 20
        epochs = 3
        batch_size = 128
        model_tools.ensure_directory_exists('statistics')
        MODEL.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy']
        )
        statistics_file = model_tools.unique_path(
            'statistics/statistics', 'csv')
        first_entry = True
        for generation in range(0, generations):
            print('generation ' + str(generation + 1)
                  + ' / ' + str(generations))
            dataset = data.artificial_uncertainty.damaged(
                cats_vs_dogs.cats_vs_dogs_augmented(),
                variable_extents=True)
            # dataset = data.artificial_uncertainty.damaged(
            #    mnist.mnist_data(), variable_extents=True)
            dataset = dataset.prepare(batch_size)
            history = MODEL.fit(
                dataset.train(),
                # batch_size=batch_size,
                epochs=epochs, verbose=1,
                validation_data=dataset.test())
            score = MODEL.evaluate(
                dataset.test(), verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            model_tools.safe_save_model(MODEL, 'models/active/model')
            if first_entry:
                pandas.DataFrame(history.history).to_csv(statistics_file)
                first_entry = False
            else:
                pandas.DataFrame(history.history).to_csv(
                    statistics_file, header=None, mode='a')

    def display_images():
        """
            Display images for the assessment of the Integrated Certainty
            Gradients feature attribution method.
        """
        include_simple_feature_removal = False
        sample_index = 3955

        dataset = data.artificial_uncertainty.with_certainty(
            mnist.mnist_data())
        # dataset = data.artificial_uncertainty.with_certainty(
        #    data.cats_vs_dogs.cats_vs_dogs())
        image = dataset.x_test_at(sample_index, add_sample_channel=True)
        dataset = dataset.take_test_x(2000)

        print('considering confidence:')
        results = MODEL.predict(image)
        print('Predicted value: ' + str(tf.argmax(results, axis=1).numpy()[0]))
        print(results)
        print('ignoring confidence:')
        image_value = pixel_certainty.disregard_certainty(image)
        results = MODEL.predict(image_value)
        print('Predicted value: ' + str(tf.argmax(results, axis=1).numpy()[0]))
        print(results)
        print('baseline values:')
        print(MODEL.predict(pixel_certainty.disregard_certainty(image, 0.)))

        # check_uncertainty_baseline_attribution(image, MODEL, 12, 5)

        expected_gradients = integrated_gradients.integrated_gradients(
            image, MODEL, dataset, True, 500)
        baseline_distribution = tensor_tools.pick(dataset, 500)
        distribution_baseline = tf.reduce_mean(baseline_distribution, 0)
        distribution_integrated_gradients = \
            integrated_gradients.integrated_gradients(
                image, MODEL, distribution_baseline)
        gradients = integrated_gradients.classifier_gradients(image, MODEL)
        certainty_gradients = integrated_certainty_gradients\
            .image_integrated_certainty_gradients(image, MODEL)

        plots = (
            image_tensors.ImagePlot()
            .add_hue_scale('Value')
            .add_unsigned_saturation_scale('Certainty')
            .add_signed_lightness_scale('Gradient')
            .add_unsigned_lightness_scale('Combined', minimum=-1.0)
        )
        if image.shape[-1] == 4:
            plots.add_two_channel_positive_saturated(
                pixel_certainty.collapse_value_channels(image[0]),
                title='Source image')
        else:
            plots.add_two_channel_positive_saturated(
                image[0], title='Source image')
        show_image_value = True
        if show_image_value:
            if image.shape[-1] == 4:
                plots.add_rgb_image(
                    pixel_certainty.discard_certainty(image_value[0]),
                    title='Image value')
            else:
                plots.add_single_channel(
                    pixel_certainty.discard_certainty(image_value[0]),
                    title='Image value')
        (
            plots
            # .add_two_channel_positive_saturated(
            #    distribution_baseline, title='Distribution baseline')
            .new_row()
            # .add_single_channel(
            #    discard_certainty(gradients[0]), True, title='Value gradient')
            .add_single_channel(
                pixel_certainty.discard_value(gradients[0]), True,
                title='Certainty gradient')
            # .add_two_channel_positive_white(
            #    gradients[0], True, title='Combined gradients')
            .add_single_channel(
                integrated_certainty_gradients
                .certainty_aware_simple_integrated_gradients(
                    image, MODEL, 0.0)[0],
                True, title='Zero integrated gradients')
            .add_single_channel(
                integrated_certainty_gradients
                .certainty_aware_simple_integrated_gradients(
                    image, MODEL, 0.5)[0],
                True, title='Middle integrated gradients')
            .add_single_channel(
                integrated_certainty_gradients
                .certainty_aware_double_sided_integrated_gradients(
                    image, MODEL)[0],
                True, title='Double sided integrated gradients')
            .add_single_channel(
                expected_gradients[0], True, title='Expected gradients')
            .add_single_channel(
                distribution_integrated_gradients[0], True,
                title='Distribution integrated gradients')
            .new_row()
        )
        if include_simple_feature_removal:
            (
                plots
                .add_single_channel(
                    feature_removal.simple_feature_removal(
                        discard_certainty(image[0]), MODEL),
                    True, title='Simple feature removal')
                .add_single_channel(
                    feature_removal.simple_feature_removal(
                        discard_certainty(image[0]), MODEL, 0.5),
                    True, title='Midpoint feature removal')
                .add_single_channel(
                    feature_removal.double_sided_feature_removal(
                        discard_certainty(image[0]), MODEL),
                    True, title='Double sided feature removal')
                .add_single_channel(
                    feature_removal.feature_certainty_removal(image[0], MODEL),
                    True, title='Feature certainty removal')
            )
        (
            plots
            .add_single_channel(
                certainty_gradients[0], True, title='Certainty gradients')
            .add_overlay(
                image_tensors.normalize_channel_centered(
                    image_tensors.brighten(
                       tf.expand_dims(certainty_gradients[0], -1), 10),
                    0, -1.0, 1.0, 0.),
                image_tensors.remap_channel(
                    image_tensors.rgb_to_greyscale(
                        pixel_certainty.discard_certainty(image[0])),
                    0, 0., 1., -0.5, 0.5),
                title='Overlaid certainty gradients A')
            .add_overlay(
                image_tensors.remap_channel(
                    image_tensors.rgb_to_greyscale(
                        pixel_certainty.discard_certainty(image[0])),
                    0, 0., 1., -0.5, 0.5),
                image_tensors.normalize_channel_centered(
                    image_tensors.brighten(
                        tf.expand_dims(certainty_gradients[0], -1), 10),
                    0, -1., 1., 0.),
                title='Overlaid certainty gradients B')
            .show()
         )

    if TRAIN_MODEL:
        train_model()
    if DISPLAY_IMAGES:
        display_images()
    print('Finished')
