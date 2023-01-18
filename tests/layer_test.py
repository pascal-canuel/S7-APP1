import unittest

import numpy as np

from dnn_framework import FullyConnectedLayer, BatchNormalization, Sigmoid, ReLU
from tests import test_layer_input_grad, test_layer_parameter_grad, DELTA


class LayerTestCase(unittest.TestCase):
    def test_fully_connected_layer_forward(self):
        layer = FullyConnectedLayer(2, 1)
        layer.get_parameters()['w'][:] = np.array([[2, 3]])
        layer.get_parameters()['b'][:] = np.array([1])
        x = np.array([[-1.0, 0.5]])
        y, _ = layer.forward(x)

        self.assertAlmostEqual(y[0], 0.5, delta=DELTA)

    def test_fully_connected_layer_forward_backward(self):
        self.assertTrue(test_layer_input_grad(FullyConnectedLayer(4, 10), (2, 4)))
        self.assertTrue(test_layer_parameter_grad(FullyConnectedLayer(4, 10), (2, 4), 'w'))
        self.assertTrue(test_layer_parameter_grad(FullyConnectedLayer(4, 10), (2, 4), 'b'))

    def test_batch_normalization_forward_training(self):
        layer = BatchNormalization(2)
        layer.get_parameters()['gamma'][:] = np.array([1, 2])
        layer.get_parameters()['beta'][:] = np.array([-1, 1])

        x = np.array([[-1, -2], [1, -1], [0, -1.5]])
        y, _ = layer.forward(x)

        expected_y = np.array([[-2.22474487, -1.44948974], [0.22474487, 3.44948974], [-1.0, 1.0]])
        print('y:', y)
        print('expected:', expected_y)
        self.assertTrue(np.all(np.abs(y - expected_y) < DELTA))

    def test_batch_normalization_forward_evaluation(self):
        layer = BatchNormalization(2)
        layer.eval()
        layer.get_buffers()['global_mean'][:] = np.array([0.0, -1.5])
        layer.get_buffers()['global_variance'][:] = np.array([0.81649658, 0.40824829])
        layer.get_parameters()['gamma'][:] = np.array([1, 2])
        layer.get_parameters()['beta'][:] = np.array([-1, 1])

        x = np.array([[-1, -2], [1, -1], [0, -1.5]])
        y, _ = layer.forward(x)

        expected_y = np.array([[-2.10668124, -0.56508266], [0.10668124, 2.56508266], [-1.0, 1.0]])
        self.assertTrue(np.all(np.abs(y - expected_y) < DELTA))

    def test_batch_normalization_backward(self):
        self.assertTrue(test_layer_input_grad(BatchNormalization(4), (2, 4)))
        self.assertTrue(test_layer_parameter_grad(BatchNormalization(4), (2, 4), 'gamma'))
        self.assertTrue(test_layer_parameter_grad(BatchNormalization(4), (2, 4), 'beta'))

    def test_sigmoid_forward(self):
        layer = Sigmoid()
        x = np.array([-1.0, 0.5])
        y, _ = layer.forward(x)

        self.assertAlmostEqual(y[0], 0.2689414, delta=DELTA)
        self.assertAlmostEqual(y[1], 0.6224593, delta=DELTA)

    def test_sigmoid_backward(self):
        self.assertTrue(test_layer_input_grad(Sigmoid(), (2, 3)))

    def test_relu_forward(self):
        layer = ReLU()
        x = np.array([-1.0, 0.5])
        y, _ = layer.forward(x)

        self.assertAlmostEqual(x[0], -1.0, delta=DELTA)
        self.assertAlmostEqual(x[1], 0.5, delta=DELTA)
        self.assertAlmostEqual(y[0], 0.0, delta=DELTA)
        self.assertAlmostEqual(y[1], 0.5, delta=DELTA)

    def test_relu_backward(self):
        self.assertTrue(test_layer_input_grad(ReLU(), (2, 3)))


if __name__ == '__main__':
    unittest.main()
