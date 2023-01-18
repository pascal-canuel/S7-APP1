import pickle


class Network:
    def __init__(self, layers):
        self._layers = layers
        self._caches = []

    def train(self):
        """
        This method switches the network to the training mode.
        """
        for layer in self._layers:
            layer.train()

    def eval(self):
        """
        This method switches the network to the evaluation mode.
        """
        for layer in self._layers:
            layer.eval()

    def get_parameters(self):
        """
        :return: A dictionary containing all layer parameters
        """
        parameters = {}
        for i in range(len(self._layers)):
            for name, parameter in self._layers[i].get_parameters().items():
                parameters[str(i) + '.' + name] = parameter

        return parameters

    def get_buffers(self):
        """
        :return: A dictionary containing all layer buffers
        """
        buffers = {}
        for i in range(len(self._layers)):
            for name, parameter in self._layers[i].get_buffers().items():
                buffers[str(i) + '.' + name] = parameter

        return buffers

    def forward(self, x):
        """
        This method performs the forward pass of the network.
        :param x: The input tensor
        :return: The output value
        """
        self._caches = []

        for layer in self._layers:
            x, cache = layer.forward(x)
            self._caches.append(cache)

        return x

    def backward(self, output_grad):
        """
        This method performs the backward pass of the network.
        :param output_grad: The gradient with respect to the output of the network
        :return: A dictionary containing the gradient with respect to all layer parameters
        """
        parameter_grads = {}

        for i in reversed(range(len(self._layers))):
            output_grad, parameters_grad = self._layers[i].backward(output_grad, self._caches[i])
            for name, parameter in parameters_grad.items():
                parameter_grads[str(i) + '.' + name] = parameter

        return parameter_grads

    def save(self, path):
        """
        Save the network parameters and buffers.
        :param path: The file path
        """
        with open(path, 'wb') as file:
            data = {'parameters': self.get_parameters(), 'buffers': self.get_buffers()}
            pickle.dump(data, file)

    def load(self, path):
        """
        Load the network parameters and buffers.
        :param path: The file path
        """
        with open(path, 'rb') as file:
            loaded_parameters = pickle.load(file)
            parameters = self.get_parameters()
            buffers = self.get_buffers()

            for name, value in loaded_parameters['parameters'].items():
                parameters[name][:] = value

            for name, value in loaded_parameters['buffers'].items():
                buffers[name][:] = value
