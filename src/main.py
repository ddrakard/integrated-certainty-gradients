import os
import sys
from pathlib import Path

import keras
import pandas
import tensorflow as tf

import attribution.feature_removal as feature_removal
import attribution.integrated_gradients as integrated_gradients
import data
import image_tensors
import model_tools
import pixel_certainty
import tensor_tools
from pixel_certainty import discard_certainty

if __name__ == '__main__':

    train_model = False
    display_images = True

    source_directory = Path(__file__).resolve().parent
    sys.path.insert(0, source_directory)
    os.chdir(source_directory.parent)
    model = model_tools.load_latest_model('models/active/model')

    if train_model:
        generations = 1
        epochs = 1
        batch_size = 128
        model_tools.ensure_directory_exists('statistics')
        statistics_file = model_tools.unique_path(
            'statistics/statistics', 'csv')
        first_entry = True
        for generation in range(0, generations):
            print('generation ' + str(generation + 1)
                  + ' / ' + str(generations))
            dataset = data.damaged_data()
            model.compile(
                loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy']
            )
            history = model.fit(
                dataset.x_train(), dataset.y_train(), batch_size=batch_size,
                epochs=epochs, verbose=1,
                validation_data=(dataset.x_test(), dataset.y_test()))
            score = model.evaluate(
                dataset.x_test(), dataset.y_test(), verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            model_tools.safe_save_model(model, 'models/active/model')
            if first_entry:
                pandas.DataFrame(history.history).to_csv(statistics_file)
                first_entry = False
            else:
                pandas.DataFrame(history.history).to_csv(
                    statistics_file, header=None, mode='a')

    if display_images:
        dataset = data.mnist_data_with_certainty().x_test()
    #    dataset = data.damaged_data().x_test()

    #    sample_index = 832
    #    sample_index= 321
    #    sample_index = 3955
        sample_index = 3956
    #    sample_index = 7772
    #    sample_index = 333

        image = (
            tensor_tools
            .Selection()[sample_index:sample_index + 1, ...]
            .apply(dataset)
        )

        # x_test_damaged.gather([470], axis=0)
        print('considering confidence:')
        results = model.predict(image)
        print('Predicted value: ' + str(tf.argmax(results, axis=1).numpy()[0]))
        print(results)

        expected_gradients = integrated_gradients.integrated_gradients(
            image, model, dataset, True, 500)

        baseline_distribution = tensor_tools.pick(dataset, 500)
        distribution_baseline = tf.reduce_mean(baseline_distribution, 0)
        distribution_integrated_gradients = \
            integrated_gradients.integrated_gradients(
                image, model, distribution_baseline)

        gradients = integrated_gradients.classifier_gradients(image, model)

        certainty_gradients = integrated_gradients\
            .image_certainty_integrated_gradients(image, model)

        print('ignoring confidence:')
        image_value = pixel_certainty.disregard_certainty(image)
        results = model.predict(image_value)
        print('Predicted value: ' + str(tf.argmax(results, axis=1).numpy()[0]))
        print(results)

        (image_tensors.ImagePlot()
            .add_hue_scale('Value')
            .add_unsigned_saturation_scale('Certainty')
            .add_signed_lightness_scale('Gradient')
            .add_unsigned_lightness_scale('Combined', minimum=-1.0)
            .add_two_channel_positive_saturated(
                image[0], title='Source image')
            # .add_two_channel_positive_saturated(
            #   image_value[0], title='Image value')
            .add_two_channel_positive_saturated(
                distribution_baseline, title='Distribution baseline')
            .new_row()
            .add_single_channel(
                discard_certainty(gradients[0]), True, title='Value gradient')
            .add_single_channel(
                pixel_certainty.discard_value(gradients[0]), True,
                title='Certainty gradient')
            # .add_two_channel_positive_white(
            #    gradients[0], True, title='Combined gradients')
            .add_single_channel(
                integrated_gradients.simple_integrated_gradients(
                    image, model, 0.0)[0],
                True, title='Zero integrated gradients')
            .add_single_channel(
                integrated_gradients.simple_integrated_gradients(
                    image, model, 0.5)[0],
                True, title='Middle integrated gradients')
            .add_single_channel(
                integrated_gradients.double_sided_integrated_gradients(
                    image, model)[0],
                True, title='Double sided integrated gradients')
            .add_single_channel(
                expected_gradients[0], True, title='Expected gradients')
            .add_single_channel(
                distribution_integrated_gradients[0], True,
                title='Distribution integrated gradients')
            .new_row()
            .add_single_channel(
                feature_removal.simple_feature_removal(
                    discard_certainty(image[0]), model),
                True, title='Simple feature removal')
            .add_single_channel(
                feature_removal.simple_feature_removal(
                    discard_certainty(image[0]), model, 0.5),
                True, title='Midpoint feature removal')
            .add_single_channel(
                feature_removal.double_sided_feature_removal(
                    discard_certainty(image[0]), model),
                True, title='Double sided feature removal')
            .add_single_channel(
                feature_removal.feature_certainty_removal(image[0], model),
                True, title='Feature certainty removal')
            .add_single_channel(
                certainty_gradients[0], True, title='Certainty gradients')
            .add_overlay(
                image_tensors.remap_channel(
                    pixel_certainty.discard_certainty(image[0]),
                    0, 0., 1., -0.5, 0.5),
                image_tensors.normalize_channel_centered(
                    image_tensors.brighten(
                        tf.expand_dims(certainty_gradients[0], -1), 10),
                    0, -1.0, 1.0, 0.),
                title='Overlaid certainty gradients')
            .show()
         )

    print('Finished')
