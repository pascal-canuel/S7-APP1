class Layer:
    """
    This is the base class of every neural network layer.
    Every layer class must inherit from this class and must overload the get_parameters, forward and backwards methods.
    """

    def __init__(self):
        self._is_training = True

    def train(self):
        """
        This method switches the layer to the training mode.
        """
        self._is_training = True

    def eval(self):
        """
        This method switches the layer to the evaluation mode.
        """
        self._is_training = False

    def is_training(self):
        """
        :return: True if the layer is in the training mode, False if the layer is in the evaluation mode
        """
        return self._is_training

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        raise NotImplementedError()

    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        raise NotImplementedError()

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """
        raise NotImplementedError()
