from typing import List, Callable

import numpy as np
import tensorflow as tf

"""
    ## Note on terminology ##

    The names "axes_count" is used for the number of axes of a tensor for lack
    of knowing a better option. The name "rank" is ambiguous, in mathematics it
    refers to the dimension of the span of a matrix. "order" is also
    potentially confusing and "degree" is not apparently common.
"""


class Selection:
    """
        Defines a cuboidal subspace of a tensor shape (or tensor shape family).
    """

    def __init__(self):
        self._axes_count = None
        self._slices = {}

    def slice_axis(self, axis: int, the_slice: slice) -> 'Selection':
        """
            Add a sliced axis to the selection.

            :param axis: The index of the axis to slice. Starts at 0. Negative
                indices count from the last axis backwards.
            :param the_slice: The dimensions to include in the slice. Should be
                a slice object.
            :return: self
        """
        if self._axes_count is not None and (
                axis >= self._axes_count or axis < - self._axes_count):
            # Unpleasant error message due to weak terminology and complexity
            # of negative indices.
            raise ValueError(
                'Tried to set axis (' + str(axis)
                + ') outside the axes count of the Selection ('
                + str(self._axes_count) + ')')
        self._slices[axis] = self._cast_slice(the_slice)
        return self

    def select_channel(self, axis: int, channel: int,
                       remove_axis: bool = False) -> 'Selection':
        """
            Slice a single index ("channel") of a single axis.

            :param axis: The axis to restrict.
            :param channel: The only index to be selected in the axis.
            :param remove_axis: If true the axis will be removed, otherwise an
                axis of size 1 will remain.
            :return: self
        """
        if remove_axis:
            raise Exception('Axis removal is not implemented.')  # TODO
        stop = channel + 1
        if stop == 0:
            stop = None
        self.slice_axis(axis, slice(channel, stop))
        return self

    def __getitem__(*arguments) -> 'Selection':
        """
            Specify axes to slice.

            For example:
            selection[:,:,2:,...,-10:8,-2:]
            means:
            Retain elements with index greater than or equal to 2 in axis 2,
            with index less than 8 and within 10 elements of the end in the
            second to last axis, and in the last two positions of the last
            axis.

            :param arguments: A number of slices (which can use colon syntax)
                and optionally an ellipsis at any point.
            :return: self
        """
        if not isinstance(arguments[0], Selection):
            # Called as a class method
            return Selection()[arguments[1]]
        self = arguments[0]
        arguments = arguments[1]
        if not isinstance(arguments, tuple):
            # unfortunately __getitem__ treats single and multiple arguments
            # differently
            arguments = tuple(arguments)
        if self._axes_count is not None:
            if (len(arguments) > self._axes_count or (
                    len(arguments) < self._axes_count
                    and Ellipsis not in arguments)):
                raise ValueError(
                    'Number of axes (' + str(len(arguments))
                    + ') does not match fixed degree ('
                    + str(self._axes_count) + ') of Selection.')
        fixed_size = True
        for index, argument in enumerate(arguments):
            if argument is Ellipsis:
                fixed_size = False
                break
            self._slices[index] = self._cast_slice(argument)
        if fixed_size:
            self._axes_count = len(arguments)
        else:
            for index, argument in enumerate(reversed(arguments)):
                if argument == Ellipsis:
                    break
                self._slices[-(1 + index)] = self._cast_slice(argument)
        return self

    def fixed_axes_count(self) -> bool:
        """
            Does the Selection only apply to tensors with a specific number of
            axes.
        """
        return self._axes_count is not None

    def axes_count(self) -> int:
        """
            If the Selection can only apply to tensors with a specific number
            of axes, return that number.
        """
        if self._axes_count is None:
            raise Exception('Selection does not have a fixed degree.')
        return self._axes_count

    def slices(self, input_axes_count=None) -> List[slice]:
        """
            Get a list of slices equivalent to the Selection.

            :param input_axes_count: The number of axes of the tensor to be
                sliced. Does not need to be provided if the Selection is for a
                predetermined number of axes (fixed axes count).
            :return: A list of slices of equivalent to this Selection for a
            tensor with the given axes count.
        """
        if input_axes_count is None:
            if self._axes_count is None:
                raise Exception(
                    'The Selection does not have a fixed axes count so am axes'
                    + 'count must be passed.')
            axes_count = self._axes_count
        else:
            if self._axes_count is not None:
                if self._axes_count != input_axes_count:
                    raise ValueError(
                        'Fixed axes count Selection has wrong axes count: '
                        + str(self._axes_count)
                        + ' axes but needed ' + str(input_axes_count))
            else:
                if input_axes_count < len(self._slices):
                    raise ValueError(
                        'Variable axes count Selection has wrong minimum '
                        + 'axes count: at least ' + str(len(self._slices))
                        + ' axes but needed ' + str(input_axes_count))
            axes_count = input_axes_count
        result = list(map(lambda _: slice(None), range(axes_count)))
        for index, defined_slice in self._slices.items():
            result[index] = defined_slice
        return result

    def shape(self, input_shape) -> List[int]:
        """
            Get the shape that a tensor would have after undergoing this
            Selection.

            :param input_shape: The shape of the tensor that would be passed
                in.
            :return: The shape that the resulting tensor would have.
        """
        if self._axes_count is not None and self._axes_count != len(
                input_shape):
            raise ValueError(
                'Tensor of wrong shape: needed ' + str(self._axes_count)
                + ' axes but there were ' + str(len(input_shape)))
        result = list(input_shape)
        for axis, the_slice in self._slices.items():
            if the_slice.start is None:
                start = 0
            elif the_slice.start < 0:
                start = input_shape[axis] + the_slice.start
            else:
                # TODO: Check in slice
                start = the_slice.start
            if the_slice.stop is None:
                stop = input_shape[axis]
            elif the_slice.stop < 0:
                # TODO: Check in slice
                stop = input_shape[axis] + 1 + the_slice.stop
            else:
                # TODO: Check in slice
                stop = the_slice.stop
            if the_slice.step is None:
                step = 1
            else:
                step = the_slice.step
            result[axis] = ((stop - start - 1) // abs(step)) + 1
        return result

    def apply(self, tensor: tf.Tensor) -> tf.Tensor:
        """
            Apply the selection to a tensor.

            :param tensor: The tensor to be selected from.
            :return: The result of applying all the slices in this Selection to
                the axes of the input tensor.
        """
        return tensor[self.slices(len(tensor.shape))]

    def transform(self, transformation: Callable[[tf.Tensor], tf.Tensor],
                  tensor: tf.Tensor) -> tf.Tensor:
        """
            Apply a change to the the selected part of the tensor.

            :param transformation: A function to apply the desired change to
                the selected part of the tensor. Must take a tensor and return
                a tensor of the same shape.
            :param tensor: The tensor which will have part changed.
            :return: A new tensor with the same shape as the input tensor, with
                the selected part changed.
        """
        result = tf.Variable(tensor)
        selection = self.apply(result)
        selection.assign(transformation(selection))
        return tf.convert_to_tensor(result.numpy())

    def mask(self, shape, positive=np.float32(1.),
             negative=np.float32(0.)) -> tf.Tensor:
        """
            Create a tensor where all the areas which would be selected by this
            Selection have the positive value, and all other areas have the
            negative value.

            :param shape: The shape the result should have.
            :param positive: The value applied to areas of the mask which would
                be selected by this Selection.
            :param negative: The value applied to areas of the mask which would
                not be selected by this Selection.
            :return: The mask tensor.
        """
        return self.multiplex(tf.fill(shape, positive),
                              tf.fill(shape, negative))

    def multiplex(self, update, baseline) -> tf.Tensor:
        """
            Replace areas of a tensor that are selected by this Selection with
            values from another tensor.

            NOTE: support for strides is not currently implemented.
            :param update: The tensor to take the replacement values from. It
                must have the same dimension as baseline.
            :param baseline: The tensor that the values not selected by this
                Selection come from.
            :return: A tensor formed by replacing the areas of baseline
                selected by this Selection with values from update.
        """
        baseline_shape = baseline.shape
        if baseline_shape != update.shape:
            # Disable broadcasting, this might cause a problem if one of the
            # range axes is resized
            raise Exception(
                'Baseline shape ' + str(baseline.shape) +
                ' does not match update shape ' + str(update.shape))
        mask = tf.Variable(tf.fill(baseline_shape, False))
        selection = self.apply(mask)
        selection.assign(tf.fill(self.shape(baseline_shape), True))
        return tf.where(mask, update, baseline)

    def _cast_slice(self, value):
        """
            Handle nicely the values passed to for selecting on axes.
        """
        if isinstance(value, slice):
            return value
        elif isinstance(value, int):
            raise Exception(
                'Selection by integer is not supported currently pending '
                + 'decision whether to remove axis or not.')
            if value == -1:
                return slice(value, None)
            else:
                return slice(value, value + 1)
        else:
            raise ValueError(
                'Unrecognised value passed for Selection slice: ' + str(value))


class Coordinates:
    """
        Provides an iterable sequence of all coordinates for a tensor shape.
    """

    class Iterator:
        def __init__(self, shape):
            self._shape = shape
            self._indices = [0] * len(shape)

        def __next__(self) -> List[int]:
            if len(self._indices) == 0:
                raise StopIteration
            index = -1
            while True:
                if self._indices[index] < self._shape[index]:
                    result = self._indices.copy()
                    self._indices[-1] += 1
                    return result
                else:
                    self._indices[index] = 0
                    index -= 1
                    if index < -len(self._shape):
                        raise StopIteration
                    self._indices[index] += 1

    def __init__(self, shape):
        self._shape = shape

    def __iter__(self):
        return self.Iterator(self._shape)


def index_tensor(shape) -> tf.Tensor:
    """
        Returns a tensor of the indices of a tensor with a shape the same as
        the shape argument.

        The result has the shape of the shape argument, except an extra axis at
        the end with size equal to the length of shape. When used as the
        indices argument to tensorflow.gather_nd, it results in the identity
        operation.
    """
    if len(shape) == 0:
        return tf.convert_to_tensor([], dtype=tf.int8)
    data = tf.constant(list(Coordinates(shape)))
    return tf.reshape(data, shape + [len(shape)])
