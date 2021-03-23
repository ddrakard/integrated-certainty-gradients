"""
    This module provides capabilities for working with image tensors. These
    are represented as tensors whose last three axes are row, column, and
    channel respectively. The channel axis commonly represents red, green, blue
    components when of size 3 and lightness/brightness when of size 1.
"""
import typing
import math

import matplotlib.pyplot as pyplot
import numpy as np
import tensorflow as tf

import tensor_tools


class ImagePlot:
    """
        Convenience class to display images on screen using matplotlib pyplot.
    """

    def __init__(self):
        self._images = []
        self._parameters = []
        self._positions = {}
        self._next_position = [0, 0]
        self._rows = 1
        self._columns = 1

    def show(self) -> None:
        """
            Display the images on the screen.

            All add image methods take the following optional parameters:
                title: str: Display this text above the image.
                xaxis: bool: Display an x axis scale if true.
                yaxis: bool: Display a y axis scale if true.
                xlabel: str: Display this text next to the x axis.
                ylabel: str: Display this text next to the y axis.
        """
        _, axes = pyplot.subplots(self._rows, self._columns)
        if self._rows == 1:
            axes = [axes]
        if self._columns == 1:
            axes = [axes]
        for rows in axes:
            for current_axes in rows:
                # This sets default behaviour for submitted axes and also hides
                # 'missing' axes
                current_axes.set_visible(False)
        for index, image in enumerate(self._images):
            try:
                position = self._positions[index]
                current_axes = axes[position[0]][position[1]]
                current_axes.set_visible(True)
                parameters = self._parameters[index]
                if 'ymin' in parameters:
                    y_minimum = parameters['ymin']
                else:
                    y_minimum = 0.0
                if 'ymax' in parameters:
                    y_maximum = parameters['ymax']
                else:
                    y_maximum = 1.0
                if 'xmin' in parameters:
                    x_minimum = parameters['xmin']
                else:
                    x_minimum = 0.0
                if 'xmax' in parameters:
                    x_maximum = parameters['xmax']
                else:
                    shape = image.shape
                    x_maximum = (y_maximum - y_minimum) * shape[1] / shape[0]
                current_axes.imshow(
                    image, extent=[x_minimum, x_maximum, y_minimum, y_maximum])
                current_axes.get_xaxis().set_visible(False)
                current_axes.get_yaxis().set_visible(False)
                if 'xaxis' in parameters and parameters['xaxis']:
                    current_axes.get_xaxis().set_visible(True)
                if 'yaxis' in parameters and parameters['yaxis']:
                    current_axes.get_yaxis().set_visible(True)
                if 'xlabel' in parameters:
                    current_axes.get_xaxis().set_label_text(
                        parameters['xlabel'])
                if 'ylabel' in parameters:
                    current_axes.get_yaxis().set_label_text(
                        parameters['ylabel'])
                if 'title' in parameters:
                    current_axes.set_title(parameters['title'])
            except Exception as exception:
                message = 'Exception showing graph number ' + str(index + 1)
                if 'title' in parameters:
                    message += ' titled "' + str(parameters['title']) + '"'
                message += ': ' + str(exception)
                raise Exception(message) from exception
        pyplot.show()

    def add_hue_scale(
            self, label: str = None, minimum: float = 0.0,
            maximum: float = 1.0) -> 'ImagePlot':
        """
            Display a scale to indicate the numerical meaning of hues.

            :param label: A text caption for the scale.
            :param minimum: The displayed value for the minimum (blue) end of
                the scale.
            :param maximum: The displayed value for the maximum (orange) end of
                the scale
            :return: self
        """
        image = np.linspace(np.outer([1.0] * 20, [1.0, 1.0]),
                            np.outer([1.0] * 20, [0.0, 1.0]), 256)
        self.add_two_channel_positive_saturated(
            image, yaxis=True, ylabel=label, ymin=minimum, ymax=maximum)
        return self

    def add_unsigned_saturation_scale(
            self, label: str = None, minimum: float = 0.0,
            maximum: float = 1.0) -> 'ImagePlot':
        """
            Display a scale to indicate the numerical meaning of saturation.

            :param label: A text caption for the scale.
            :param minimum: The displayed value for the minimum (black) end of
                the scale.
            :param maximum: The displayed value for the maximum (saturated) end
                of the scale
            :return: self
        """
        orange = np.linspace(np.outer([1.0] * 20, [1.0, 1.0]),
                             np.outer([1.0] * 20, [1.0, 0.0]), 256)
        blue = np.linspace(np.outer([1.0] * 20, [0.0, 1.0]),
                           np.outer([1.0] * 20, [0.0, 0.0]), 256)
        image = tf.concat([orange, blue], 1)
        self.add_two_channel_positive_saturated(
            image, yaxis=True, ylabel=label, ymin=minimum, ymax=maximum)
        return self

    def add_signed_saturation_scale(
            self, label: typing.Optional[str] = None, minimum: float = -1.0,
            maximum: float = 1.0) -> 'ImagePlot':
        """
            Display a scale to indicate the numerical meaning of saturation,
            from blue to black to orange.

            :param label: A text caption for the scale.
            :param minimum: The displayed value for the minimum (blue) end of
                the scale.
            :param maximum: The displayed value for the maximum (orange) end of
                the scale
            :return: self
        """
        orange = np.linspace(np.outer([1.0] * 20, [1.0, 1.0]),
                             np.outer([1.0] * 20, [1.0, 0.0]), 128)
        blue = np.linspace(np.outer([1.0] * 20, [0.0, 0.0]),
                           np.outer([1.0] * 20, [0.0, 1.0]), 128)
        image = tf.concat([orange, blue], 0)
        self.add_two_channel_positive_saturated(
            image, yaxis=True, ylabel=label, ymin=minimum, ymax=maximum)
        return self

    def add_unsigned_lightness_scale(
            self, label: str = None, minimum: float = 0.0,
            maximum: float = 1.0) -> 'ImagePlot':
        """
            Display a scale to indicate the numerical meaning of lightness,
            from near-black to saturated color to near-white.

            :param label: A text caption for the scale.
            :param minimum: The displayed value for the minimum (dark) end of
                the scale.
            :param maximum: The displayed value for the maximum (light) end of
                the scale
            :return: self
        """
        orange = np.linspace(np.outer([1.0] * 20, [1.0, 1.0]),
                             np.outer([1.0] * 20, [1.0, -1.0]), 256)
        grey = np.linspace(np.outer([1.0] * 20, [0.0, 1.0]),
                           np.outer([1.0] * 20, [0.0, -1.0]), 256)
        blue = np.linspace(np.outer([1.0] * 20, [-1.0, 1.0]),
                           np.outer([1.0] * 20, [-1.0, -1.0]), 256)
        image = tf.concat([orange, grey, blue], 1)
        self.add_two_channel_positive_white(image, yaxis=True, ylabel=label,
                                            ymin=minimum, ymax=maximum)
        return self

    def add_signed_lightness_scale(
            self, label: str = None, minimum: float = -1.0,
            maximum: float = 1.0) -> 'ImagePlot':
        """
            Display a scale to indicate the numerical meaning of lightness,
            from blue-white to blue to black to orange to orange-white.

            :param label: A text caption for the scale.
            :param minimum: The displayed value for the minimum (blue) end of
                the scale.
            :param maximum: The displayed value for the maximum (orange) end of
                the scale
            :return: self
        """
        image = np.linspace(
            np.outer([1.0] * 20, [1.0]), np.outer([1.0] * 20, [-1.0]), 256)
        self.add_single_channel(
            image, yaxis=True, ylabel=label, ymin=minimum, ymax=maximum)
        return self

    def add_single_channel(
            self, image: tf.Tensor, normalize: bool = False,
            **parameters: typing.Any
            ) -> 'ImagePlot':
        """
            Display a single channel image, in false color. The image can have
            positive or negative values.

            :param image: The image to display.
            :param normalize: If true, the values in the image will be scaled
                based on greatest absolute value whilst preserving 0.0.
            :param parameters: See ImagePlot class docstring.
            :return: self
        """
        if image.shape[-1] != 1:
            if len(image.shape) == 3:
                raise ValueError(
                    'The image has multiple channels. Shape: '
                    + str(image.shape))
            image = tf.expand_dims(image, axis=-1)
        if normalize:
            image = normalize_channel_centered(image, 0, -1.0, 1.0, 0.0)
        return self.add_rgb_image(
            single_channel_to_false_color_rgb(image), **parameters)

    def add_two_channel_positive_saturated(
            self, image: tf.Tensor, **parameters: typing.Any) -> 'ImagePlot':
        """
            Display a two unsigned channel image.

            Warmer colors from blue to orange correspond to increasing values
            in the first channel. Brighter colors from black to fully saturated
            correspond to increasing values in the second channel.
            :param image: The image to display.
            :param parameters: See ImagePlot class docstring.
            :return: self
        """
        return self.add_rgb_image(
            two_channel_to_positive_saturated_rgb(image), **parameters)

    def add_two_channel_positive_white(
            self, image: tf.Tensor, normalize: bool = False,
            **parameters: typing.Any
            ) -> 'ImagePlot':
        """
            Display a two signed channel image.

            Warmer colors from blue to orange correspond to increasing values
            in the first channel. Lighter colors from black to white correspond
            to increasing values in the second channel.
            :param image: The image to display.
            :param normalize: If true, the values in the image will be scaled
                based on greatest absolute value of their channel whilst
                preserving 0.0.
            :param parameters: See ImagePlot class docstring.
            :return: self
        """
        if normalize:
            image = normalize_channel_centered(image, 0, -1.0, 1.0, 0.0)
            image = normalize_channel_centered(image, 1, -1.0, 1.0, 0.0)
        return self.add_rgb_image(
            two_channel_to_positive_white_rgb(image), **parameters)

    def add_overlay(
            self, lightness_image: tf.Tensor, hue_image: tf.Tensor,
            normalize: bool = False, **parameters: typing.Any) -> 'ImagePlot':
        """
            Displays two single signed channel images merged.

            Color mapping is as with add_two_channel_positive_white
            :param lightness_image: The image from which pixel lightnesses,
                from black to white, are taken.
            :param hue_image: The images from which pixel hues, from blue to
                orange, are taken.
            :param normalize: If true, the values in the images will be scaled
                based on their greatest absolute values whilst preserving 0.0.
            :param parameters: See ImagePlot class docstring.
            :return: self
        """

        lightness_image = validate_image(lightness_image, 1, "lightness_image")
        hue_image = validate_image(hue_image, 1, "hue_image")
        return self.add_two_channel_positive_white(
            tf.concat([hue_image, lightness_image], -1),
            normalize, **parameters
        )

    def new_row(self) -> 'ImagePlot':
        """
            Place all additional images on a following row.

            :return: self
        """
        self._next_position = [self._next_position[0] + 1, 0]
        self._rows += 1
        return self

    def add_rgb_image(
            self, image: tf.Tensor, **parameters: typing.Any) -> 'ImagePlot':
        """
            Add a standard RGB (red green blue) images

            :param image: The image to display.
            :param parameters: See ImagePlot class docstring.
            :return: self
        """
        image = validate_image(image, 3)
        self._images.append(image)
        self._parameters.append(parameters)
        # Assign the current index to the current position
        self._positions[len(self._images) - 1] = tuple(self._next_position)
        self._next_position[1] = self._next_position[1] + 1
        if self._next_position[1] > self._columns:
            self._columns = self._next_position[1]
        return self


def validate_image(
        image: tf.Tensor, channel_count: int = None, image_name: str = None
        ) -> tf.Tensor:
    """
        Check that the provided tensor is a valid image, and return a
        normalised version.

    :param image: A tensor encoding an image, with shape ([sample], row,
        column, [channel])
    :param channel_count: The number of channels the image should have.
    :param image_name: Used in error message if the image does not pass
        validation.
    :return: A normalised version of the input image.
    """

    # TODO: add support for other checks and normalisations. For example
    #  leading (sample) and trailing (singleton channel) axes.

    def descriptor() -> str:
        result = 'Image tensor '
        if image_name is not None:
            result += '"' + image_name + '" '
        result += 'of shape ' + str(image.shape)
        return result

    if channel_count is not None and image.shape[-1] != channel_count:
        raise ValueError(
            descriptor() + ' has wrong number of channels, requires '
            + str(channel_count))
    return image


def channel_absolute(image: tf.Tensor, channel: int) -> tf.Tensor:
    """ Make the values in a channel positive. """
    return (
        tensor_tools.Selection().select_channel(-1, channel)
        .transform(tf.abs, image)
    )


def remap_channel(
        image: tf.Tensor, channel: int, input_minimum: float,
        input_maximum: float, output_minimum: float, output_maximum: float
        ) -> tf.Tensor:
    """
        Shift and scale a channel so it fits into a new range.

        :param image: The image or images tensor.
        :param channel: The index of the channel in the last axis to transform.
        :param input_minimum: The bottom of the range in the input tensor.
        :param input_maximum: The top of the range in the input tensor.
        :param output_minimum: The bottom of the range in the output tensor,
            mapped from input_minimum.
        :param output_maximum: The top of the range in the output tensor,
            mapped from input_maximum.
        :return: The transformed tensor.
    """

    def remap(data):
        output_range = output_maximum - output_minimum
        input_range = input_maximum - input_minimum
        scale_factor = output_range / input_range
        return (data - input_minimum) * scale_factor + output_minimum

    return tensor_tools.Selection().select_channel(-1, channel).transform(
        remap, image)


def normalize_channel_centered(
        image: tf.Tensor, channel: int, minimum: float, maximum: float,
        center_from: float) -> tf.Tensor:
    """
        Scale the channel values to fill the given range as much as possible,
            whilst preserving the center.

        :param image: The image or images tensor to transform.
        :param channel: The channel to transform in the last axis of the image
            tensor.
        :param minimum: The minimum of the output range.
        :param maximum: The maximum of the output range.
        :param center_from: The value of the center/origin of the data before
            the mapping. This will be mapped to the
            middle of the output range.
        :return: The transformed tensor.
    """

    def normalize_centered(data):
        centered_data = data - center_from
        normalized_data = centered_data / tf.reduce_max(tf.abs(centered_data))
        # Divide by 2 because normalized_data is from -1.0 to 1.0
        scaled_data = normalized_data * (maximum - minimum) / 2.0
        return scaled_data + ((maximum + minimum) / 2.0)

    return (
        tensor_tools.Selection().select_channel(-1, channel)
        .transform(normalize_centered, image)
    )


def normalize_channel_full_range(
        image: tf.Tensor, channel: int, minimum: float, maximum: float
        ) -> tf.Tensor:
    """
        Scale the channel values to fill the given range fully.

        :param image: The image or images tensor to transform.
        :param channel: The channel to transform in the last axis of the image
            tensor.
        :param minimum: The minimum of the output range.
        :param maximum: The maximum of the output range.
        :return: The transformed tensor.
    """

    def normalize_full_range(data):
        start_minimum = tf.reduce_min(data)
        start_maximum = tf.reduce_max(data)
        scale_factor = (maximum - minimum) / (start_maximum - start_minimum)
        return (data - start_minimum) * scale_factor + minimum

    return (
        tensor_tools.Selection().select_channel(-1, channel)
        .transform(normalize_full_range, image)
    )


def normalize_channels_full_range(
        image: tf.Tensor, channels: typing.Iterable[int], minimum: float,
        maximum: float) -> tf.Tensor:
    """
        Scale the channel values to fill the given range fully. This
        implementation may not be efficient.

        :param image: The image or images tensor to transform.
        :param channels: The channels to transform in the last axis of the
            image tensor.
        :param minimum: The minimum of the output range.
        :param maximum: The maximum of the output range.
        :return: The transformed tensor.
    """
    start_minimum = tf.reduce_min(tf.gather(image, channels, axis=-1))
    start_maximum = tf.reduce_max(tf.gather(image, channels, axis=-1))

    def normalize_full_range(data):
        scale_factor = (maximum - minimum) / (start_maximum - start_minimum)
        return (data - start_minimum) * scale_factor + minimum

    for channel in channels:
        selection = tensor_tools.Selection().select_channel(-1, channel)
        image = selection.transform(normalize_full_range, image)
    return image


def single_channel_to_false_color_rgb(image: tf.Tensor) -> tf.Tensor:
    """
        Maps values in the range -1.0 to 1.0 to colors White, Blue, Black,
        Orange, White
    """
    if tf.math.reduce_max(tf.math.abs(image)) > 1.0:
        raise Exception('The image data is out of range ' +
                        '(absolute value greater than 1.0).')

    def transform_pixel(value):
        # Compressing the range of the second channel makes it possible to
        # distinguish -1.0 and 1.0.
        value = value * 0.9
        base_red = value > 0.
        value = abs(value)
        peak_color = max(0., value * 2 - 1)
        base_color = min(1., value * 2)
        result = [peak_color, value, base_color]
        if base_red:
            result.reverse()
        return np.array(result, np.float32)

    if image.shape[-1] != 1:
        image = tf.expand_dims(image, axis=-1)
    return np.apply_along_axis(transform_pixel, -1, image)


def rgb_to_greyscale(image: tf.Tensor) -> tf.Tensor:
    """
        Convert a three channel red-green-blue image tensor to a single channel
        greyscale image tensor.
    """
    return tf.reduce_mean(image, axis=-1, keepdims=True)


def two_channel_to_positive_saturated_rgb(image: tf.Tensor) -> tf.Tensor:
    """
        Maps values from [0.0, 0.0] to [1.0, 1.0] to rgb colors.

        Warmer colors from blue to orange correspond to increasing values in
        the first channel. Brighter colors from black to fully saturated
        correspond to increasing values in the second channel.
    """

    def transform_pixel(pixel_vector):
        first = pixel_vector[0]
        # Compressing the range makes it easier to see the first channel in
        # dark pixels
        second = pixel_vector[1] * np.float32(0.8) + np.float32(0.2)
        # Any two channels carry full information of the image, so I hope this
        # helps color blind people.
        return [
            first * second,
            np.float32(0.5) * second,
            (np.int8(1) - first) * second
        ]

    return np.apply_along_axis(transform_pixel, -1, image)


def two_channel_to_positive_white_rgb(image: tf.Tensor) -> tf.Tensor:
    """
        Maps values from [-1.0, -1.0] to [1.0, 1.0] to rgb colors.

        Warmer colors from blue to orange correspond to increasing values in
        the first channel. Lighter colors from black to white correspond to
        increasing values in the second channel.
    """
    image = validate_image(image, 2)

    def transform_pixel(pixel_vector):
        # Change the range of the first channel to [0.0, 1.0]
        first = pixel_vector[0] * 0.5 + 0.5
        # Compressing the range of the second channel makes it easier to see
        # the first channel by preventing brightness saturating.
        second = pixel_vector[1] * np.float32(0.65) + 0.1
        if second > 0:
            return np.array([
                first + (np.int8(1) - first) * second,
                np.float32(0.5) * (np.int8(1) + second),
                (np.int8(1) - first) + first * second
            ])
        else:
            second = second + np.float32(1.)
            return np.array([
                first * second,
                np.float32(0.5) * second,
                (np.int8(1) - first) * second
            ])

    return np.apply_along_axis(transform_pixel, -1, image)


def brighten(image: tf.Tensor, extent: int = 2) -> tf.Tensor:
    """
        Shifts the values of a tensor away from 0.0 and towards -1.0 and 1.0.
        This may help make faint images clearer.

        :param image: The image to increase brightness of.
        :param extent: How much to increase the brightness.
        :return: The brightened image.
    """
    result = tf.math.sin(image * math.pi / 2)
    if extent == 1:
        return result
    else:
        return brighten(result, extent - 1)
