from dnn_framework.optimizer import Optimizer


class SgdOptimizer(Optimizer):
    """
    This class implements a stochastic gradient descent optimizer.
    """

    def __init__(self, parameters, learning_rate=0.01):
        raise NotImplementedError()

    def _step_parameter(self, parameter, parameter_grad, parameter_name):
        raise NotImplementedError()
