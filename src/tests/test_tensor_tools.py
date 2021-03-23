"""
    Tests for the tensor_tools module
"""

import tensorflow as tf

from tensor_tools import Selection, Coordinates, index_tensor, pick, \
    axis_outer_operation
from tests.utilities import sample_tensor, TensorflowTestCase


class TestSelection(TensorflowTestCase):
    """ Test the Selection class. """

    def test_constructor(self):
        """ Can a Selection be instantiated. """
        self.assertTrue(isinstance(Selection(), Selection))

    def test_simple_ranges(self):
        """
            Can a variable axis count selection be converted to a list of
            slices.
        """
        self.assertEqual(
            Selection()[..., 4:5].slices(4),
            [slice(None), slice(None), slice(None), slice(4, 5, None)])

    def test_slice_shape_simple(self):
        """ Test output shape with two positive indices """
        self.assertEqual(Selection()[2:4].shape([9]), [2])

    def test_slice_shape_unbound_below(self):
        """ Test output shape with a lower positive index only """
        self.assertEqual(Selection()[:3].shape([9]), [3])

    def test_slice_shape_unbound_above(self):
        """ Test output shape with an upper positive index only """
        self.assertEqual(Selection()[2:].shape([9]), [7])

    def test_slice_shape_full_range(self):
        """ Test output shape with an upper positive index only """
        self.assertEqual(Selection()[:].shape([9]), [9])

    def test_slice_shape_negative_below(self):
        """ Test output shape with an upper positive index only """
        self.assertEqual(Selection()[-4:7].shape([9]), [2])

    def test_slice_shape_negative_above(self):
        """ Test output shape with an upper positive index only """
        self.assertEqual(Selection()[3:-1].shape([9]), [5])

    def test_variable_axes_count_slice(self):
        """
            Can a range be applied to one axis of a tensor with any axis count.
        """
        self.assert_tensor_equal(
            Selection()[..., 2:4].apply(sample_tensor()),
            sample_tensor()[:, :, :, 2:4])

    def test_fixed_axes_count_slice(self):
        """
            Can a range be applied to one axis of a tensor with a specific axis
            count.
        """
        self.assert_tensor_equal(
            Selection()[:, :, :, 2:4].apply(sample_tensor()),
            sample_tensor()[:, :, :, 2:4])

    def test_variable_axes_count_transform(self):
        """ Can a selected part of a tensor be changed. """
        self.assert_tensor_equal(
            Selection()[..., 1:].transform(
                lambda t: tf.math.multiply(t, 2),
                sample_tensor(2, 3)),
            [[11, 24, 26], [21, 44, 46], [31, 64, 66]])

    def test_variable_axes_count_multiplex(self):
        """ Can the multiplex operation be applied to a tensor. """
        shape = sample_tensor().shape
        reference = tf.Variable(tf.fill(shape, False))
        reference[:, 2:3, 1:3, :].assign(tf.fill([4, 1, 2, 4], True))
        selection = (Selection()
                     .slice_axis(1, slice(2, 3))
                     .slice_axis(2, slice(1, 3)))
        self.assert_tensor_equal(
            reference,
            selection.multiplex(tf.fill(shape, True), tf.fill(shape, False)))


class TestCoordinates(TensorflowTestCase):
    """ Test the Coordinates class. """

    def test_empty(self):
        """ Can Coordinates be created for an empty tensor. """
        for _ in Coordinates([]):
            raise Exception('Should not be called for empty coordinates')

    def test_cardinality(self):
        """ Does Coordinates iterate the correct number of times. """
        counter = 0
        for _ in Coordinates([2, 4, 3, 1]):
            counter += 1
        self.assertEqual(counter, 2 * 4 * 3 * 1)


class TestIndexTensor(TensorflowTestCase):
    """ Test the index_tensor method """

    def test_empty(self):
        """ Can an index tensor be made for an empty tensor. """
        self.assert_tensor_equal([], index_tensor([]))

    def test_small(self):
        """ Can an index tensor be created for a 2 axis tensor. """
        self.assert_tensor_equal(
            [[[0, 0], [0, 1]], [[1, 0], [1, 1]]],
            index_tensor([2, 2]))

    def test_large(self):
        """ Can an index tensor be created for a 4 axis tensor. """
        tensor = index_tensor([4, 1, 5, 3])
        for axis0 in range(0, 4):
            axis1 = 0
            for axis2 in range(0, 5):
                for axis3 in range(0, 3):
                    self.assert_tensor_equal(
                        tensor[axis0][axis1][axis2][axis3],
                        [axis0, axis1, axis2, axis3])

    def test_gather_nd(self):
        """
            Does an index tensor act as the identity argument for
            tensorflow.gather_nd.
        """
        self.assert_tensor_equal(
            sample_tensor(),
            tf.gather_nd(sample_tensor(), index_tensor(sample_tensor().shape))
        )


class TestPick(TensorflowTestCase):
    """ Test the pick method """

    random_seed = 678

    def sample_tensor_subtensors(self, indices):
        """
            Create a tensor by picking a list of subtensors of the sample
            tensor along its first axis.
        """
        tensor = sample_tensor(3, 3)
        return tf.stack([tensor[index] for index in indices])

    def test_zero_count(self):
        """ Pick no tensors to get an empty tensor. """
        self.assert_tensor_equal(
            pick(sample_tensor(3, 3), 0, seed=self.random_seed),
            tf.zeros((0, 3, 3), dtype=tf.int32))

    def test_one_count(self):
        """ Randomly pick one tensor. """
        self.assert_tensor_equal(
            pick(sample_tensor(3, 3), 1, seed=self.random_seed),
            self.sample_tensor_subtensors([0]))

    def test_many_count(self):
        """ Randomly pick 10 tensors. """
        self.assert_tensor_equal(
            pick(sample_tensor(3, 3), 10, seed=self.random_seed),
            self.sample_tensor_subtensors(
                [1, 2, 1, 0, 0, 1, 0, 2, 2, 1]))


class TestAxisOuterOperation(TensorflowTestCase):
    """ Test the axis_outer_operation method """

    def test_last_axis(self):
        """ Check for correct handling of axis arithmetic on -1. """
        self.assert_tensor_equal(
            axis_outer_operation(
                -1, [sample_tensor(3, 2), sample_tensor(3, 2)],
                lambda tensors: 10000 * tensors[0] + tensors[1]),
            [
                [
                    [[1110111, 1110112], [1120111, 1120112]],
                    [[1210121, 1210122], [1220121, 1220122]],
                ],
                [
                    [[2110211, 2110212], [2120211, 2120212]],
                    [[2210221, 2210222], [2220221, 2220222]],
                ]
            ]
        )
