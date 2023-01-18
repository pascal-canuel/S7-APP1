import matplotlib.pyplot as plt
import numpy as np


class LossMetric:
    """
    This is a class to calculate the loss metric from batch loss values.
    """

    def __init__(self):
        self._loss = 0
        self._count = 0

    def clear(self):
        """
        This method must be called between epoch.
        """
        self._loss = 0
        self._count = 0

    def add(self, loss):
        """
        This method aggregates the batch loss value.
        :param loss: The batch loss value
        """
        self._loss += loss
        self._count += 1

    def get_loss(self):
        """
        :return: The loss metric
        """
        if self._count == 0:
            return 0
        return self._loss / self._count


class ClassificationAccuracyMetric:
    """
    This is a class to calculate the accuracy metric from batch outputs.
    """

    def __init__(self):
        self._good = 0
        self._total = 0

    def clear(self):
        """
        This method must be called between epoch.
        """
        self._good = 0
        self._total = 0

    def add(self, predicted_class_scores, target_classes):
        """
        This method aggregates the batch loss outputs.
        :param predicted_class_scores: The network output
        :param target_classes: The target
        :return:
        """
        predicted_classes = predicted_class_scores.argmax(axis=1)

        self._good += np.sum((predicted_classes == target_classes))
        self._total += target_classes.shape[0]

    def get_accuracy(self):
        """
        :return: The accuracy metric
        """
        if self._total == 0:
            return 0
        return self._good / self._total


class LossLearningCurves:
    """
    This class creates loss learning curves.
    """

    def __init__(self):
        self._training_loss_values = []
        self._validation_loss_values = []

    def add_training_loss_value(self, value):
        """
        Add the next training loss value.
        :param value: The next training loss value
        """
        self._training_loss_values.append(value)

    def add_validation_loss_value(self, value):
        """
        Add the next validation loss value.
        :param value: The next validation loss value
        """
        self._validation_loss_values.append(value)

    def save_figure(self, output_path):
        """
        Save the learning curves to an image file.
        :param output_path: The image output path
        """
        fig = plt.figure(figsize=(5, 5), dpi=300)
        ax1 = fig.add_subplot(111)

        epochs = range(1, len(self._training_loss_values) + 1)
        ax1.plot(epochs, self._training_loss_values, '-o', color='tab:blue', label='Training')
        ax1.plot(epochs, self._validation_loss_values, '-o', color='tab:orange', label='Validation')
        ax1.set_title(u'Loss')
        ax1.set_xlabel(u'Epoch')
        ax1.set_ylabel(u'Loss')
        ax1.legend()

        fig.savefig(output_path)
        plt.close(fig)


class LossAccuracyLearningCurves:
    """
    This class creates loss and accuracy learning curves.
    """

    def __init__(self):
        self._training_loss_values = []
        self._validation_loss_values = []
        self._training_accuracy_values = []
        self._validation_accuracy_values = []

    def add_training_loss_value(self, value):
        """
        Add the next training loss value.
        :param value: The next training loss value
        """
        self._training_loss_values.append(value)

    def add_validation_loss_value(self, value):
        """
        Add the next training accuracy value.
        :param value: The next training accuracy value
        """
        self._validation_loss_values.append(value)

    def add_training_accuracy_value(self, value):
        """
        Add the next validation loss value.
        :param value: The next validation loss value
        """
        self._training_accuracy_values.append(value)

    def add_validation_accuracy_value(self, value):
        """
        Add the next validation accuracy value.
        :param value: The next validation accuracy value
        """
        self._validation_accuracy_values.append(value)

    def save_figure(self, output_path):
        """
        Save the learning curves to an image file.
        :param output_path: The image output path
        """
        fig = plt.figure(figsize=(10, 5), dpi=300)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        epochs = range(1, len(self._training_accuracy_values) + 1)
        ax1.plot(epochs, self._training_loss_values, '-o', color='tab:blue', label='Training')
        ax1.plot(epochs, self._validation_loss_values, '-o', color='tab:orange', label='Validation')
        ax1.set_title(u'Loss')
        ax1.set_xlabel(u'Epoch')
        ax1.set_ylabel(u'Loss')
        ax1.legend()

        epochs = range(1, len(self._training_accuracy_values) + 1)
        ax2.plot(epochs, self._training_accuracy_values, '-o', color='tab:blue', label='Training')
        ax2.plot(epochs, self._validation_accuracy_values, '-o', color='tab:orange', label='Validation')
        ax2.set_title(u'Accuracy')
        ax2.set_xlabel(u'Epoch')
        ax2.set_ylabel(u'Accuracy')
        ax2.legend()

        fig.savefig(output_path)
        plt.close(fig)
