"""
    This module aids working with complete supervised learning datasets (input,
    output; test, train).
"""

import typing
import tensorflow as tf

from language import call_method


class Dataset:
    """
        A set of data for training and evaluating a neural network model.
    """

    def __init__(self, train: tf.data.Dataset, test: tf.data.Dataset):
        if train.element_spec != test.element_spec:
            raise ValueError('Mismatched test and training datasets.')
        self._train = train
        self._test = test

    def train(self) -> tf.data.Dataset:
        """
            :return: The training dataset.
        """
        return self._train

    def test(self) -> tf.data.Dataset:
        """
            :return: The test dataset.
        """
        return self._test

    def modify(
            self, callback: typing.Callable[[tf.data.Dataset], tf.data.Dataset]
            ) -> 'Dataset':
        """
            Modify the internal Tensorflow test and train Datasets.

            :param callback: The function to apply to the Datasets.
            :return: The modified dataset.
        """
        return Dataset(callback(self._train), callback(self._test))

    def train_prepared(self, batch_size):
        """
            Return the training dataset prepared for execution (shuffled,
            batched and prefetched).
        """
        return (
            self._train
                .shuffle(batch_size)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
        )

    def prepare(self, batch_size):
        """
            Return a copy prepared for execution (shuffled, batched and
            prefetched).
        """
        def operations(data):
            return (
                data
                .shuffle(batch_size)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

        return self.modify(operations)

    def batch(self, size: int) -> 'Dataset':
        """
            Call Dataset.batch on the internal Tensorflow Datasets.
        """
        return self.modify(call_method('batch', [size]))

    def unbatch(self) -> 'Dataset':
        """
            Call Dataset.unbatch on the internal Tensorflow Datasets.
        """
        return self.modify(call_method('unbatch'))

    def map(
            self,
            callback: typing.Callable[
                [tf.Tensor, tf.Tensor], typing.Tuple[tf.Tensor, tf.Tensor]]
            ) -> 'Dataset':
        """
            Return a copy with a function applied to each element.

            :param callback: A callback taking an x (input tensor) and y
                (output tensor) arguemnt and returning a tuple pair of
                transformed x and y tensors.
            :return: The transformed dataset.
        """
        return self.modify(call_method(
            'map', [callback], {'num_parallel_calls': tf.data.AUTOTUNE}))

    def map_x(
            self, callback: typing.Callable[[tf.Tensor], tf.Tensor]
            ) -> 'Dataset':
        """
            Return a copy with a function applied to each input element.
        """
        return self.map(lambda x, y: (callback(x), y))

    def map_y(
            self, callback: typing.Callable[[tf.Tensor], tf.Tensor]
            ) -> 'Dataset':
        """
            Return a copy with a function applied to each output element.
        """
        return self.map(lambda x, y: (x, callback(y)))

    def map_both(
            self, callback: typing.Callable[[tf.Tensor], tf.Tensor]
            ) -> 'Dataset':
        """
            Return a copy with a function applied to each input and each
            output element separately.
        """
        return self.map(lambda x, y: (callback(x), callback(y)))

    def x_train_at(
            self, index: int, add_sample_channel: bool = False) -> tf.Tensor:
        """
            Return a specific element from the training input samples.

            :param index: The index of the element to return.
            :param add_sample_channel: If true a leading singleton axis will
                 be prepended.
            :return:
        """
        result = next(iter(self._train.skip(index)))[0]
        if add_sample_channel:
            result = tf.expand_dims(result, 0)
        return result

    def y_train_at(
            self, index: int, add_sample_channel: bool = False) -> tf.Tensor:
        """
            Return a specific element from the training output samples.

            :param index: The index of the element to return.
            :param add_sample_channel: If true a leading singleton axis will
                 be prepended.
            :return:
        """
        result = next(iter(self._train.skip(index)))[1]
        if add_sample_channel:
            result = tf.expand_dims(result, 0)
        return result

    def x_test_at(
            self, index: int, add_sample_channel: bool = False) -> tf.Tensor:
        """
            Return a specific element from the test input samples.

            :param index: The index of the element to return.
            :param add_sample_channel: If true a leading singleton axis will
                 be prepended.
            :return:
        """
        result = next(iter(self._test.skip(index)))[0]
        if add_sample_channel:
            result = tf.expand_dims(result, 0)
        return result

    def y_test_at(
            self, index: int, add_sample_channel: bool = False) -> tf.Tensor:
        """
            Return a specific element from the test output samples.

            :param index: The index of the element to return.
            :param add_sample_channel: If true a leading singleton axis will
                 be prepended.
            :return:
        """
        result = next(iter(self._test.skip(index)))[1]
        if add_sample_channel:
            result = tf.expand_dims(result, 0)
        return result

    def take_train(self, count: int) -> tf.data.Dataset:
        """
            Return a Tensorflow Dataset with items from the start of the
            training data. Those items are removed from this dataset.

            :param count: The number of items to return.
            :return: The taken samples.
        """
        result = self._train.take(count)
        self._train = self._train.skip(count)
        return result

    def take_test(self, count: int) -> tf.data.Dataset:
        """
            Return a Tensorflow Dataset with items from the start of the
            test data. Those items are removed from this dataset.

            :param count: The number of items to return.
            :return: The taken samples.
        """
        result = self._test.take(count)
        self._train = self._test.skip(count)
        return result

    def take_train_x(self, count: int) -> tf.Tensor:
        """
            Return a Tensorflow Dataset with items from the start of the
            training input (x) data. Those items and the corresponding outputs
            are removed from this dataset.

            :param count: The number of items to return.
            :return: The taken samples.
        """
        return tf.convert_to_tensor([x for x, y in self.take_train(count)])

    def take_test_x(self, count: int) -> tf.Tensor:
        """
            Return a Tensorflow Dataset with items from the start of the
            test input (x) data. Those items and the corresponding outputs
             are removed from this dataset.

            :param count: The number of items to return.
            :return: The taken samples.
        """
        return tf.convert_to_tensor([x for x, y in self.take_test(count)])

    def x_element_spec(self):
        """
            Return the Tensorflow Dataset element_spec of the input tensors.
        """
        return self._train.element_spec[0]

    def y_element_spec(self):
        """
            Return the Tensorflow Dataset element_spec of the output tensors.
        """
        return self._train.element_spec[1]

    @staticmethod
    def from_tensors(
            x_train: tf.Tensor, y_train: tf.Tensor, x_test: tf.Tensor,
            y_test: tf.Tensor) -> 'Dataset':
        """
            Convert tensors into a dataset.
        """
        values = [x_train, y_train, x_test, y_test]
        for index in range(4):
            values[index] = tf.data.Dataset.from_tensor_slices(
                tf.unstack(tf.convert_to_tensor(values[index])))
        return Dataset(
            tf.data.Dataset.zip((values[0], values[1])),
            tf.data.Dataset.zip((values[2], values[3])))


def combine(data: typing.List[Dataset]) -> Dataset:
    """
        Combine entries from multiple datasets in randomly shuffled order. Test
        and training data is kept separated.
    """

    def pack(*pairs):
        sequences = zip(*pairs)
        lists = tuple(map(list, sequences))
        return tf.data.Dataset.from_tensor_slices(lists)

    def combine_datasets(datasets):
        return tf.data.Dataset.zip(datasets).flat_map(pack)

    train = tuple(dataset.train() for dataset in data)
    test = tuple(dataset.test() for dataset in data)
    return Dataset(combine_datasets(train), combine_datasets(test))
