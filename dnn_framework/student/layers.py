import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        self.w = np.random.randn(output_count, input_count) * np.sqrt(2 / (input_count + output_count))
        self.b = np.random.randn(output_count) * np.sqrt(2 / output_count)

    def get_parameters(self):
        return {'w': self.w, 'b': self.b}

    def get_buffers(self):
        return {}

    def forward(self, x):
        y = x @ self.w.T + self.b
        return y, {'x': x}

    def backward(self, output_grad, cache):
        input_grad = output_grad @ self.w
        w_grad = output_grad.T @ cache['x']
        b_grad = np.sum(output_grad, axis=0)
        return input_grad, {'w': w_grad, 'b': b_grad}


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        self.gamma = np.ones(input_count)
        self.beta = np.zeros(input_count)

        self.alpha = alpha

        self.global_mean = np.zeros(input_count)
        self.global_variance = np.zeros(input_count)

        self.eps = np.finfo(float).eps

        super().__init__()

    def get_parameters(self):
        return {'gamma': self.gamma, 'beta': self.beta}

    def get_buffers(self):
        return {'global_mean': self.global_mean, 'global_variance': self.global_variance}

    def forward(self, x):
        if Layer.is_training(self):
            y, cache = self._forward_training(x)
        else:
            y, cache = self._forward_evaluation(x)

        return y, cache

    def _forward_training(self, x):
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        self.global_mean = (1 - self.alpha) * self.global_mean + self.alpha * mean
        self.global_variance = (1 - self.alpha) * self.global_variance + self.alpha * var

        x_norm = (x - mean) / np.sqrt(var + self.eps)
        y = self.gamma * x_norm + self.beta

        return y, {'x': x, 'x_norm': x_norm, 'mean_b': mean, 'var_b': var}

    def _forward_evaluation(self, x):
        x_norm = (x - self.global_mean) / np.sqrt(self.global_variance + self.eps)
        y = self.gamma * x_norm + self.beta

        return y, {'x': x, 'x_norm': x_norm}

    def backward(self, output_grad, cache):
        M = output_grad.shape[0]

        x_norm_grad = output_grad * self.gamma

        var_grad = np.sum(x_norm_grad * (cache['x'] - cache['mean_b']) * (-1/2 * np.power((cache['var_b'] + self.eps), -3/2)), axis=0)
        mean_grad = (-np.sum(x_norm_grad, axis=0) / np.sqrt(cache['var_b'] + self.eps)) + (-2/M * var_grad * np.sum(cache['x'] - cache['mean_b'], axis=0))

        input_grad = (x_norm_grad / np.sqrt(cache['var_b'] + self.eps)) + (2/M * var_grad * (cache['x'] - cache['mean_b'])) + (1/M * mean_grad)

        gamma_grad = np.sum(output_grad * cache['x_norm'], axis=0)
        beta_grad = np.sum(output_grad, axis=0)

        return input_grad, {'gamma': gamma_grad, 'beta': beta_grad}


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y, {'y': y}

    def backward(self, output_grad, cache):
        return cache['y'] * (1 - cache['y']) * output_grad, {}


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        y = np.maximum(x, 0)
        return y, {'x': x}

    def backward(self, output_grad, cache):
        d_y = np.where(cache['x'] > 0, 1, 0) * output_grad
        return d_y, {}
