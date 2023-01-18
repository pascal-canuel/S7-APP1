class Loss:
    """
    This is the base class of every loss function.
    Every loss class must inherit from this class and must overload the calculate method.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor
        :param target: The target tensor
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        raise NotImplementedError()
