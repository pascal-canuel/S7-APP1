import inspect

import numpy as np

DELTA = 1e-5
MAX_GRAD_MEAN_ERROR = 1e-5


def test_layer_input_grad(layer, input_shape, delta=DELTA):
    """
    A function to test the backward input_grad values.
    :param layer: The layer instance
    :param input_shape: The input shape of the tensor
    :param delta: Finite difference delta
    :return: Absolute mean error between the analytical and numerical gradients
    """
    x = np.random.randn(*input_shape)
    y, cache = layer.forward(x)
    output_grad = np.random.randn(*y.shape)
    analytical_grad, _ = layer.backward(output_grad, cache)
    numerical_grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        old_value = x[i]
        x[i] += delta
        y_, _ = layer.forward(x)
        x[i] = old_value
        numerical_grad[i] = np.sum((y_ - y) * output_grad) / delta

        it.iternext()

    print('analytical:', analytical_grad)
    print('numerical:', numerical_grad)
    error = np.mean(np.abs(analytical_grad - numerical_grad))
    print(inspect.stack()[1][3], '- Absolute mean error:', error)
    return error < MAX_GRAD_MEAN_ERROR


def test_layer_parameter_grad(layer, input_shape, parameter_name, delta=DELTA):
    """
    A function to test the backward input_grad values.
    :param layer: The layer instance
    :param input_shape: The input shape of the tensor
    :param delta: Finite difference delta
    :return: Absolute mean error between the analytical and numerical gradients
    """
    x = np.random.randn(*input_shape)
    y, cache = layer.forward(x)
    output_grad = np.random.randn(*y.shape)
    _, parameter_grads = layer.backward(output_grad, cache)

    parameter = layer.get_parameters()[parameter_name]
    analytical_grad = parameter_grads[parameter_name]
    numerical_grad = np.zeros_like(parameter)

    it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        old_value = parameter[i]
        parameter[i] += delta
        y_, _ = layer.forward(x)
        parameter[i] = old_value
        numerical_grad[i] = np.sum((y_ - y) * output_grad) / delta

        it.iternext()

    error = np.mean(np.abs(analytical_grad - numerical_grad))
    print(inspect.stack()[1][3], '-', parameter_name, '- Absolute mean error:', error)
    return error < MAX_GRAD_MEAN_ERROR


def test_loss_input_grad(loss, input_shape, target, delta=DELTA):
    """
    A function to test the loss input_grad values.
    :param layer: The loss instance
    :param input_shape: The input shape of the tensor
    :param target: The target tensor
    :param delta: Finite difference delta
    :return: Absolute mean error between the analytical and numerical gradients
    """
    x = np.random.randn(*input_shape)
    y, analytical_grad = loss.calculate(x, target)
    numerical_grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        old_value = x[i]
        x[i] += delta
        y_, _ = loss.calculate(x, target)
        x[i] = old_value
        numerical_grad[i] = np.sum(y_ - y) / delta

        it.iternext()

    print('analytical:', analytical_grad)
    print('numerical:', numerical_grad)
    error = np.mean(np.abs(analytical_grad - numerical_grad))
    print(inspect.stack()[1][3], '- Absolute mean error:', error)
    return error < MAX_GRAD_MEAN_ERROR
