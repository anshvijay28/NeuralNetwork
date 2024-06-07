import numpy as np
from util import *


class Layer:
    def __init__(self, prevNeurons, nextLayer):
        """
        Separate method defined for randomizing weights
        to allow for re-randomization of weights when
        trying to find optimal learning rate or other
        hyperparameters.
        """
        self.randomizeWeights(prevNeurons, nextLayer)

    def randomizeWeights(self, prevNeurons, nextLayer) -> None:
        """
        Initializes random weights and biases according
        to the number of incoming and outgoing weights
        for each neuron.

        Note: For for layer i, its weight/bias matrix will
        refer to the weights connecting layer i - 1 and i.
        Thus, input layer's weight or bias matrix will not
        be used.
        """
        self.prevNeurons = prevNeurons
        self.nextLayer = nextLayer
        bound = (1 / np.sqrt(prevNeurons)) * 10
        # Xavier Weight Initialization
        if type(nextLayer) == int:
            self.weights = np.random.uniform(
                -bound, bound, size=(nextLayer, prevNeurons)
            )
            self.bias = np.random.uniform(-bound, bound, size=(nextLayer, 1))
        else:
            self.weights = np.random.uniform(
                -bound,
                bound,
                size=(self.nextLayer.prevNeurons, prevNeurons),
            )
            self.bias = np.random.uniform(
                -bound, bound, size=(nextLayer.prevNeurons, 1)
            )

    def activation(self, inp):
        """
        Activation function for each Neuron in this Layer.
        Ex: Sigmoid, ReLU, Tanh, etc.
        """
        raise NotImplementedError

    def activationDeriv(self, inp):
        """
        Applies the derivative of the activation function
        on the input vector. Used during backpropagation.
        """
        raise NotImplementedError

    def forward(self, inp):
        """
        Takes in the previous layer's output vector and
        propagates it through the NN. Because of how this
        Neural Network was implemented, the next layer's
        activation function will be used.
        """
        u = self.nextLayer.weights @ inp
        if len(u.shape) == 1:
            u = u[..., np.newaxis]
        u += self.nextLayer.bias
        o = self.nextLayer.activation(u)
        return u, o

    def computeGradients(self, inputVector, error, prevOutput):
        """
        This method will take in the previous layer's output,
        the error propagated to this layer, weights of the
        next layer, and the previous layer's output vector in
        order to calculate the deltas for each weight of this
        layer. Used during backpropagation.
        """
        raise NotImplementedError

    def update(self, W_delta, B_delta, alpha):
        """
        Updates layer's weights and biases according to the
        learning rate and the deltas calculated during
        backpropagation. Used during backpropagation.
        """
        self.weights -= alpha * W_delta
        self.bias -= alpha * B_delta


class Input(Layer):
    """
    Input Layer to a 4 layered neural network using
    Sigmoid for the activation function.

    This layer technically has a weight and bias matrix,
    but it is not used since these weights would be
    connecting the input layer and the layer previous to
    the input layer, which does not exist. Values put in
    this constructor are inconsequential.
    """

    def __repr__(self) -> str:
        return (
            "Layer: Input\n"
            + f"Weights: None\n"
            + f"Bias: None\n"
            + "Next Layer: None\n"
        )

    def activation(self, inp: np.ndarray):
        """
        An activation function is trivial for the input layer,
        because there is no weight matrix to propagate the input
        through.
        """
        pass

    def activationDeriv(self, inp: np.ndarray):
        """
        Just like the activation function, the activationDeriv
        function trivial for the input layer, because there is
        no weight matrix to propagate the input through in the
        first place.
        """
        pass

    def computeGradients(self, inputVector, error, prevOutput):
        """
        There is no need for the input layer to have to compute
        gradients, because, in this implementation, the weight
        matrix belonging to the input layer connect layers one
        and zero. There is no layer zero.
        """
        pass


class Hidden(Layer):
    """
    Hidden Layer to a 4 layered neural network using
    Sigmoid for the activation function.
    """

    def __repr__(self) -> str:
        if type(self.nextLayer) == int:
            return (
                "Layer: Hidden\n"
                + f"Weight Shape: {self.weights.shape}\n"
                + f"Bias Shape: {self.bias.shape}\n"
                + f"Next Layer: Output\n"
            )
        return (
            "Layer: Hidden\n"
            + f"Weight Shape: {self.weights.shape}\n"
            + f"Bias Shape: {self.bias.shape}\n"
            + f"Next Layer: Hidden\n"
        )

    def activation(self, inp):
        return sigmoid(inp)

    def activationDeriv(self, inp):
        return sigmoidDeriv(inp)

    def computeGradients(self, inputVector, error, prevOutput) -> List[np.ndarray]:
        next_E = (self.nextLayer.weights.T @ error) * self.activationDeriv(inputVector)
        W_delta = next_E @ prevOutput.T
        B_delta = next_E

        return next_E, W_delta, B_delta


class Output(Layer):
    """
    Output Layer to a 4 layered neural network using
    Softmax for the activation function.
    """

    def __repr__(self) -> str:
        return (
            "Layer: Output\n"
            + f"Weight Shape: {self.weights.shape}\n"
            + f"Bias Shape: {self.bias.shape}\n"
            + "Next Layer: None\n"
        )

    def activation(self, inp):
        return softmax(nums=inp)

    def activationDeriv(self, inp, target):
        return softmaxDeriv(target, inp)

    def forward(self):
        """
        A forward function is trivial for the output layer
        because the softmax function is applied in the
        previous hidden layer's forward function.
        """
        pass

    def computeGradients(self, output, target, prevOutput):
        error = self.activationDeriv(output, target)
        W_delta = error @ prevOutput.T
        B_delta = error

        return error, W_delta, B_delta
