import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        y_target = np.zeros_like(x)
        y_target[np.arange(len(x)), target] = 1
        y_estimate = softmax(x)

        loss = -np.sum(y_target * np.log(y_estimate)) / x.shape[0]
        input_grad = softmax_simplified_backward(y_estimate, target)

        return loss, input_grad


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return y


def softmax_simplified_backward(y, target):
    mask = np.zeros_like(y)
    mask[np.arange(len(y)), target] = 1
    input_grad = y - mask
    input_grad /= y.shape[0]

    return input_grad


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        loss = np.mean((x - target) ** 2)
        input_grad = 2 * (x - target) / x.size

        return loss, input_grad
