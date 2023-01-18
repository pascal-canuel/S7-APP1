import gzip
import os
import pickle

from dnn_framework.dataset import Dataset


class MnistDataset(Dataset):
    def __init__(self, split):
        root = os.path.dirname(os.path.realpath(__file__))
        if split == 'training':
            path = os.path.join(root, 'mnist_training.pkl.gz')
        elif split == 'validation':
            path = os.path.join(root, 'mnist_validation.pkl.gz')
        elif split == 'testing':
            path = os.path.join(root, 'mnist_testing.pkl.gz')
        else:
            raise ValueError('Invalid split')

        with gzip.open(path, 'rb') as file:
            data = pickle.load(file)

        self._images = data['images'].astype(float)
        self._labels = data['labels']

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        return self._images[index], self._labels[index]
