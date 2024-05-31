import numpy as np
from util import *
from Layer import Input, Hidden, Output


class NN:
    def __init__(
        self,
        numNeurons: List[int],
        alpha: float,
        train_X: np.ndarray,  # (N, 784)
        train_Y: np.ndarray,  # (N, 10)
        test_X: np.ndarray,  # (M, 784)
        test_Y: np.ndarray,  # (M, 10)
    ) -> None:
        """
        Initializes a 4 layered Neural Network
        """
        self.alpha = alpha
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X  # take this out
        self.test_Y = test_Y  # take this out
        self.inp, self.h1, self.h2, self.out = numNeurons
        self.outputLayer = Output(prevNeurons=self.h2, nextLayer=self.out)
        self.hiddenLayer2 = Hidden(prevNeurons=self.h1, nextLayer=self.outputLayer)
        self.hiddenLayer1 = Hidden(prevNeurons=self.inp, nextLayer=self.hiddenLayer2)
        self.inputLayer = Input(prevNeurons=1, nextLayer=self.hiddenLayer1)

    def __repr__(self) -> str:
        return (
            f"Input Layer: {self.inp} neurons\n"
            + f"H1 Layer: {self.h1} neurons\n"
            + f"H2 Layer: {self.h2} neurons\n"
            + f"Output Layer: {self.out} neurons\n"
        )

    def feedforward(self, inp: np.ndarray) -> List[np.ndarray]:
        """
        This method will take 1 input from self.inputs
        and predict the output using current weights of
        the network.
        """
        u1, o1 = self.inputLayer.forward(inp)
        u2, o2 = self.hiddenLayer1.forward(o1)
        u3, o3 = self.hiddenLayer2.forward(o2)

        return [inp, u1, u2, u3, o1, o2, o3]

    def backPropagation(self, outs: List[np.ndarray], target: np.ndarray) -> None:
        """
        This method will take an prediction and propagate
        its error throughout the network in order to update
        weights.
        """
        for i in range(len(outs)):
            if len(outs[i].shape) == 1:
                outs[i] = outs[i][..., np.newaxis]

        X, u1, u2, u3, o1, o2, o3 = outs
        # Propagate Error
        error_l3, W3_delta, B3_delta = self.outputLayer.computeGradients(
            output=o3, target=target, prevOutput=o2
        )
        error_l2, W2_delta, B2_delta = self.hiddenLayer2.computeGradients(
            inputVector=u2, error=error_l3, prevOutput=o1
        )
        _, W1_delta, B1_delta = self.hiddenLayer1.computeGradients(
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

        return np.argmax(o3)

    def train(self) -> None:
        """
        This method will run train the neural network in order
        to achieve weights and biases that yield the least amount
        of loss. This method will return the accuracy of the model.
        """
        N = len(self.train_X)
        for i in range(N):
            outs = self.feedforward(inp=self.train_X[i])
            self.backPropagation(outs=outs, target=self.train_Y[i])
            if (i + 1) % 100 == 0:
                loss = cross_entropy_loss(target=self.train_Y[i], output=outs[-1])
                print(f"Loss on iteration {i + 1} = {loss}")

    def test(self) -> None:
        """
        Runs trained model on test dataset and outputs
        model's accuracy.
        """
        N = len(self.test_X)
        correct = 0
        for i in range(N):
            yh = self.predict(self.test_X[i])
            y = np.argmax(self.test_Y[i])
            # print(f"yh = {yh}, y = {y}")
            if y == yh:
                correct += 1
        print(f"Accuracy on test data is {correct / N}")
