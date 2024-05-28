import numpy as np
from util import *
from Layer import Input, Hidden, Output


class NN:
    def __init__(
        self,
        numNeurons: List[int],
        alpha: float,
        inputs: List[np.ndarray],
        outputs: List[np.ndarray],
    ) -> None:
        """
        Initializes a 4 layered Neural Network
        """
        self.alpha = alpha
        self.inputs = inputs
        self.outputs = outputs
        inp, h1, h2, out = numNeurons
        self.outputLayer = Output(prevNeurons=h2, nextLayer=out)
        self.hiddenLayer2 = Hidden(prevNeurons=h1, nextLayer=self.outputLayer)
        self.hiddenLayer1 = Hidden(prevNeurons=inp, nextLayer=self.hiddenLayer2)
        self.inputLayer = Input(prevNeurons=1, nextLayer=self.hiddenLayer1)

    def feedforward(self, inp: np.ndarray) -> List[np.ndarray]:
        """
        This method will take 1 input from self.inputs
        and predict the output using current weights of
        the network.
        """
        u1, o1 = self.inputLayer.forward(inp)
        u2, o2 = self.hiddenLayer1.forward(o1)
        u3, o3 = self.hiddenLayer2.forward(o2)

        return inp, u1, u2, u3, o1, o2, o3

    def backPropagation(self, outs: List[np.ndarray], target: np.ndarray) -> None:
        """
        This method will take an prediction and propagate
        its error throughout the network in order to update
        weights.
        """
        X, u1, u2, u3, o1, o2, o3 = outs
        # Propagate Error
        error_l3, W3_delta, B3_delta = self.outputLayer.computeGradients(
            output=o3, target=target, prevOutput=o2
        )
        error_l2, W2_delta, B2_delta = self.hiddenLayer2.computeGradients(
            inputVector=u2, error=error_l3, prevOutput=o1
        )
        error_l1, W1_delta, B1_delta = self.hiddenLayer1.computeGradients(
            inputVector=u1, error=error_l2, prevOutput=X
        )
        # Update Step
        self.outputLayer.update(W_delta=W3_delta, B_delta=B3_delta, alpha=self.alpha)
        self.hiddenLayer2.update(W_delta=W2_delta, B_delta=B2_delta, alpha=self.alpha)
        self.hiddenLayer1.update(W_delta=W1_delta, B_delta=B1_delta, alpha=self.alpha)

    def predict(self, inp: np.ndarray) -> int:
        """
        This method will take an input from the test dataset
        and output a prediction based on the learned weights
        and biases from training.
        """
        _, _, _, _, _, _, o3 = self.feedforward(inp)

        return np.argmax(o3) + 1

    def train(self) -> None:
        """
        This method will run train the neural network in order
        to achieve weights and biases that yield the least amount
        of loss. This method will return the accuracy of the model.
        """
        N = len(self.inputs)
        correct = 0
        for i in range(N):
            outs = self.feedforward(inp=self.inputs[i])
            self.backPropagation(outs=outs, target=self.outputs[i])
            if i % 100 == 0:
                loss = cross_entropy_loss(target=self.outputs[i], output=outs[-1])
                print(f"Loss on iteration {i} = {loss}")
