from dnn_framework.optimizer import Optimizer


class SgdOptimizer(Optimizer):
    """
    This class implements a stochastic gradient descent optimizer.
    """

    def __init__(self, parameters, learning_rate=0.01):
        self.learning_rate = learning_rate
        super().__init__(parameters)

    def _step_parameter(self, parameter, parameter_grad, parameter_name):
        return parameter - self.learning_rate*parameter_grad
