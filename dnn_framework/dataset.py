import math

import numpy as np


class Dataset:
    """
    The "Dataset" class represents is the base class of every dataset.
    Every dataset class must inherit from this class and must overload the __len__ and __getitem__ methods.
    """

    def __len__(self):
        """
        :return: The example count
        """
        raise NotImplementedError()

    def __getitem__(self, index):
        """
        :param index: The example index
        :return: A tuple of two numpy arrays. The first one is the features and the second is the target.
        """
        raise NotImplementedError()


class DatasetLoader:
    """
    The "DatasetLoader" class is used to create batches.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __len__(self):
        """
        :return: The batch count
        """
        return math.ceil(len(self._dataset) / self._batch_size)

    def __iter__(self):
        """
        :return: An iterator of the batches
        """
        dataset_indexes = self._generate_dataset_indexes()

        while dataset_indexes.shape[0] > 0:
            batch_indexes = dataset_indexes[:self._batch_size]
            dataset_indexes = dataset_indexes[self._batch_size:]

            yield self._create_batch(batch_indexes)

    def _create_batch(self, batch_indexes):
        batch_features = []
        batch_targets = []

        for i in batch_indexes:
            features, target = self._dataset[i]
            batch_features.append(features[np.newaxis, ...])
            batch_targets.append(target[np.newaxis, ...])

        return np.concatenate(batch_features, axis=0), np.concatenate(batch_targets, axis=0)

    def _generate_dataset_indexes(self):
        dataset_indexes = np.arange(len(self._dataset))
        if self._shuffle:
            np.random.shuffle(dataset_indexes)
        return dataset_indexes
