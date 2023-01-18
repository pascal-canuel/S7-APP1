import unittest

import numpy as np

from dnn_framework import CrossEntropyLoss, MeanSquaredErrorLoss
from tests import test_loss_input_grad, DELTA


class LossTestCase(unittest.TestCase):
    def test_cross_entropy_loss(self):
        loss = CrossEntropyLoss()
        x = np.array([[1.0, 2.0, 3.0], [2.0, 5.0, 1.0]])
        target = np.array([0, 2])
        loss_value, input_grad = loss.calculate(x, target)

        self.assertAlmostEqual(loss_value, 3.236744934100905, delta=DELTA)
        self.assertTrue(test_loss_input_grad(loss, x.shape, target))

    def test_mean_squared_error_loss(self):
        loss = MeanSquaredErrorLoss()
        x = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 5.0]])
        target = x + 2
        loss_value, input_grad = loss.calculate(x, target)

        self.assertAlmostEqual(loss_value, 4, delta=DELTA)
        self.assertTrue(test_loss_input_grad(loss, x.shape, target))


if __name__ == '__main__':
    unittest.main()
