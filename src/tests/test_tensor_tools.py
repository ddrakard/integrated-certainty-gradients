import tensorflow as tf

from tensor_tools import Selection, Coordinates, index_tensor
from tests.utilities import sample_tensor, TensorflowTestCase


class Test_Selection(TensorflowTestCase):
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

    def test_variable_axes_count_slice(self):
        """
            Can a range be applied to one axis of a tensor with any axis count.
        """
        self.assertTensorEqual(
            Selection()[..., 2:4].apply(sample_tensor()),
            sample_tensor()[:, :, :, 2:4])

    def test_fixed_axes_count_slice(self):
        """
            Can a range be applied to one axis of a tensor with a specific axis
            count.
        """
        self.assertTensorEqual(
            Selection()[:, :, :, 2:4].apply(sample_tensor()),
            sample_tensor()[:, :, :, 2:4])

    def test_variable_axes_count_transform(self):
        """ Can a selected part of a tensor be changed. """
        self.assertTensorEqual(
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
        self.assertTensorEqual(
            reference,
            selection.multiplex(tf.fill(shape, True), tf.fill(shape, False)))


class Test_Coordinates(TensorflowTestCase):
    """ Test the Coordinates class. """

    def test_empty(self):
        """ Can Coordinates be created for an empty tensor. """
        for _ in Coordinates([]):
            raise Exception('Should not be called for empty coordinates')

    def test_cardinality(self):
        """ Does Coordinates iterate the correct number of times. """
        counter = 0
        for coordinates in Coordinates([2, 4, 3, 1]):
            counter += 1
        self.assertEqual(counter, 2 * 4 * 3 * 1)


class Test_index_tensor(TensorflowTestCase):
    """ Test the index_tensor method """

    def test_empty(self):
        """ Can an index tensor be made for an empty tensor. """
        self.assertTensorEqual([], index_tensor([]))

    def test_small(self):
        """ Can an index tensor be created for a 2 axis tensor. """
        self.assertTensorEqual(
            [[[0, 0], [0, 1]], [[1, 0], [1, 1]]],
            index_tensor([2, 2]))

    def test_large(self):
        """ Can an index tensor be created for a 4 axis tensor. """
        tensor = index_tensor([4, 1, 5, 3])
        for d3 in range(0, 4):
            d2 = 0
            for d1 in range(0, 5):
                for d0 in range(0, 3):
                    self.assertTensorEqual(
                        tensor[d3][d2][d1][d0],
                        [d3, d2, d1, d0])

    def test_gather_nd(self):
        """
            Does an index tensor act as the identity argument for
            tensorflow.gather_nd.
        """
        self.assertTensorEqual(
            sample_tensor(),
            tf.gather_nd(sample_tensor(), index_tensor(sample_tensor().shape))
        )