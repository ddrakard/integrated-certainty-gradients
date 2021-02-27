import image_tensors
import tensor_tools
import tensorflow as tf


def show_attribution_distribution(
        attribution_function, dataset, vectorised=False, samples=100):
    if vectorised:
        attributions = attribution_function(
            tensor_tools.pick(dataset, samples))
    else:
        attributions = tf.map_fn(
            attribution_function, tensor_tools.pick(dataset, samples))
    (
        image_tensors.ImagePlot()
        .add_single_channel(
           tf.reduce_mean(tf.maximum(attributions, 0.), axis=0),
           True, title='Mean positive')
        .add_single_channel(
           tf.reduce_mean(tf.minimum(attributions, 0.), axis=0),
           True, title='Mean negative')
        .add_single_channel(
           tf.reduce_mean(attributions, axis=0),
           True, title='Mean combined')
        .add_single_channel(
            tf.reduce_mean(tf.abs(attributions), axis=0),
            True, title='Mean absolute')
        .show()
    )
