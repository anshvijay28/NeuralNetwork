import numpy as np
from util import *


class Layer:
    def __init__(self, numNeurons, nextLayer, numPrev=1):
        """
        Initializes random weights and biases according
        to the number of incoming and outgoing weights
        for each neuron.

        Note: because of implementation details, numNeurons
        refers to the number of neurons in the previous layer.
        Thus, input layer will not have a weight or bias matrix.
        """
        self.numNeurons = numNeurons
        self.bias = np.random.uniform(0.01, 1 / np.sqrt(numPrev), size=numNeurons)

        # This case is for the Output Layer, which has no next layer
        if type(nextLayer) == int:
            self.weights = np.random.uniform(
                0.01, 1 / np.sqrt(numPrev), size=(nextLayer, numNeurons)
            )
        else:
            self.nextLayer = nextLayer
            self.weights = np.random.uniform(
                0.01, 1 / np.sqrt(numPrev), size=(self.nextLayer.numNeurons, numNeurons)
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

    def forward(self):
        """
        This method will take in the previous layer's output
        vector (stochastic GD) and propagate it through the
        NN.
        """
        raise NotImplementedError

    def computeGradients(self, inputVector, error, prevOutput):
        """
        This method will take in the previous layer's output,
        the error propagated to this layer, weights of the
        next layer, and the previous layer's output vector in
        order to calculate the deltas for each weight of this
        layer.
        """
        raise NotImplementedError

    def update(self, W_delta, B_delta, alpha):
        """
        This method will update this layer's weights and bias vector
        according to the learning rate and the deltas calculated
        during backpropagation.
        """
        self.weights -= alpha * W_delta
        self.bias -= alpha * B_delta


class Input(Layer):
    """
    Input Layer to a 4 layered neural network using
    Sigmoid for the activation function.

    This layer actually doesn't actually "own" a
    bias vector and weight matrix. Values put in this
    constructor for numNeurons are inconsequential.
    """
    def __repr__(self) -> str:
        return "Layer: Input\n" + \
            f"Weights: None\n" + \
            f"Bias: None\n" + \
            "Next Layer: None"

    def activation(self, inp):
        pass

    def activationDeriv(self, inp):
        pass

    def forward(self, inp):
        u = (self.nextLayer.weights @ inp) + self.nextLayer.bias
        o = self.nextLayer.activation(u)
        return u, o

    def computeGradients(self, inputVector, error, prevOutput):
        """
        This method is trivial for the input layer, because
        layer one's weights logically belong to the 1st hidden
        layer.
        """
        pass


class Hidden(Layer):
    """
    Hidden Layer to a 4 layered neural network using
    Sigmoid for the activation function.
    """

    def __repr__(self) -> str:
        return (
            "Layer: Hidden\n"
            + f"Weight Shape: {self.weights.shape}\n"
            + f"Bias Shape: {self.bias.shape}\n"
            + f"Next Layer: Output"
            if type(self.nextLayer) == int
            else f"Next Layer: Hidden"
        )

    def activation(self, inp):
        return sigmoid(inp)

    def activationDeriv(self, inp):
        return sigmoidDeriv(inp)

    def forward(self):
        u = (self.nextLayer.weights @ self.inputVector) + self.nextLayer.bias
        o = self.nextLayer.activation(u)
        return u, o

    def computeGradients(self, inputVector, error, prevOutput):
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
            + "Next Layer: None"
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