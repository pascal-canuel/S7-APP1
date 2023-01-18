class Optimizer:
    """
    This is the base class of every neural network layer.
    Every layer class must inherit from this class and must overload the _step_parameter method.
    """

    def __init__(self, parameters):
        self._parameters = parameters

    def step(self, parameter_grads):
        """
        This method updates all parameters.
        :param parameter_grads: The dictionary returns by the network backward method.
        """
        for name, parameter in self._parameters.items():
            parameter[:] = self._step_parameter(parameter, parameter_grads[name], name)

    def _step_parameter(self, parameter, parameter_grad, parameter_name):
        """
        This method returns the new value of the parameter.
        :param parameter: The parameter tensor
        :param parameter_grad: The gradient with respect to the parameter
        :param parameter_name: The parameter name
        :return: The new value of the parameter
        """
        raise NotImplementedError()
