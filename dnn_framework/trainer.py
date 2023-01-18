import os
import time

from tqdm import tqdm

from dnn_framework.dataset import DatasetLoader


class Trainer:
    """
    This is the base class of every class that trains a neural network.
    """

    def __init__(self, network,
                 training_dataset, validation_dataset, test_dataset,
                 loss, optimizer,
                 epoch_count, batch_size, output_path):
        """
        :param network: An instance of "Network" class
        :param training_dataset: An instance of a class that inherits "Dataset"
        :param validation_dataset: An instance of a class that inherits "Dataset"
        :param test_dataset: An instance of a class that inherits "Dataset"
        :param loss: An instance of a class that inherits "Loss"
        :param optimizer: An instance of a class that inherits "Optimizer"
        :param epoch_count: The number of epoch the network is train
        :param batch_size: The training batch size
        :param output_path: The output path
        """
        self._network = network
        self._epoch_count = epoch_count
        self._output_path = output_path

        self._loss = loss
        self._optimizer = optimizer
        self._training_dataset = training_dataset
        self._validation_dataset = validation_dataset
        self._test_dataset = test_dataset

        self._training_dataset_loader = DatasetLoader(self._training_dataset, batch_size=batch_size, shuffle=True)
        self._validation_dataset_loader = DatasetLoader(self._validation_dataset, batch_size=batch_size, shuffle=False)
        self._test_dataset_loader = DatasetLoader(self._test_dataset, batch_size=batch_size, shuffle=False)

    def train(self):
        """
        Train the network.
        """
        os.makedirs(self._output_path, exist_ok=True)

        for epoch in range(self._epoch_count):
            print('Training - Epoch [{}/{}]'.format(epoch + 1, self._epoch_count), flush=True)
            self._train_one_epoch()

            print('\nValidation - Epoch [{}/{}]'.format(epoch + 1, self._epoch_count), flush=True)
            self._validate()

            self._save_checkpoint(epoch + 1)
            self._save_figures(self._output_path)
            self._print_metrics()

        print('\nTest')
        self._network.eval()
        self._test(self._network, self._test_dataset_loader)

    def _train_one_epoch(self):
        self._clear_between_training_epoch()

        self._network.train()
        for x, target in tqdm(self._training_dataset_loader):
            y = self._network.forward(x)
            loss, y_grad = self._loss.calculate(y, target)
            parameter_grads = self._network.backward(y_grad)
            self._optimizer.step(parameter_grads)

            self._measure_training_metrics(loss, y, target)

    def _validate(self):
        self._clear_between_validation_epoch()

        self._network.eval()
        for x, target in tqdm(self._validation_dataset_loader):
            y = self._network.forward(x)
            loss, _ = self._loss.calculate(y, target)
            self._measure_validation_metrics(loss, y, target)

    def _save_checkpoint(self, epoch):
        self._network.save(os.path.join(self._output_path, 'checkpoint_epoch_{}.pkl'.format(epoch)))

    def _clear_between_training_epoch(self):
        """
        This method is call between epoch to clear the training metrics.
        """
        raise NotImplementedError()

    def _measure_training_metrics(self, loss, network_output, target):
        """
        This method is call for each batch during the training to calculate some metrics.
        :param loss: The batch loss
        :param network_output: The batch output
        :param target: The batch target
        """
        raise NotImplementedError()

    def _clear_between_validation_epoch(self):
        """
        This method is call between epoch to clear the validation metrics.
        """
        raise NotImplementedError()

    def _measure_validation_metrics(self, loss, network_output, target):
        """
        This method is call for each batch during the validation to calculate some metrics.
        :param loss: The batch loss
        :param network_output: The batch output
        :param target: The batch target
        """
        raise NotImplementedError()

    def _save_figures(self, output_path):
        """
        This method saves the learning curves.
        :param output_path: The output path
        """
        raise NotImplementedError()

    def _print_metrics(self):
        """
        This method prints the metrics.
        """
        raise NotImplementedError()

    def _test(self, network, test_dataset_loader):
        """
        This method test the network with the test dataset.
        :param network: The network
        :param test_dataset_loader: The test dataset loader
        """
        raise NotImplementedError()
