import numpy as np
from util import *
from typing import List, Tuple
from Layer import Input, Hidden, Output


class NN:
    def __init__(
        self,
        numNeurons: List[int],
        alpha: float,
        train_X: np.ndarray = None,  # (N, 784)
        train_Y: np.ndarray = None,  # (N, 10)
        test_X: np.ndarray = None,
        test_Y: np.ndarray = None,
    ) -> None:
        """
        Initializes a 4 layered Neural Network
        """
        self.alpha = alpha
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
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
        feedforward will take 1 input from self.inputs
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

    def predict(self, inp: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Makes a prediction based on the learned weights
        and biases from training.
        """
        _, _, _, _, _, _, o3 = self.feedforward(inp)

        return o3, np.argmax(o3)

    def train(self) -> List[float]:
        """
        Trains the neural network in order to achieve weights and
        biases that yield the least amount of loss. This method will
        return the accuracy of the model.
        """
        losses = []
        N = len(self.train_X)
        for i in range(N):
            outs = self.feedforward(inp=self.train_X[i])
            self.backPropagation(outs=outs, target=self.train_Y[i])
            loss = cross_entropy_loss(target=self.train_Y[i], output=outs[-1])
            losses.append(loss)
            if (i + 1) % 10000 == 0:
                print(f"Loss on iteration {i + 1} = {loss}")
        return losses

    def setTestData(self, test_X, test_Y) -> None:
        """
        Allow user to change test data outside the constructor.
        """
        self.test_X = test_X
        self.test_Y = test_Y

    def setTrainData(self, train_X, train_Y) -> None:
        """
        Allow user to change train data outside the constructor.
        """
        self.train_X = train_X
        self.train_Y = train_Y

    def setAlpha(self, new_alpha) -> None:
        """
        Allow user to change learning rate between training
        and testing loops.
        """
        self.alpha = new_alpha

    def randomizeWeights(self) -> None:
        """
        Randomizes weights and biases of all layers.
        """
        prevN_Inp, nextL_Inp = self.inputLayer.prevNeurons, self.inputLayer.nextLayer
        prevN_H1, nextL_H1 = self.hiddenLayer1.prevNeurons, self.hiddenLayer1.nextLayer
        prevN_H2, nextL_H2 = self.hiddenLayer2.prevNeurons, self.hiddenLayer2.nextLayer
        prevN_O, nextL_O = self.outputLayer.prevNeurons, self.outputLayer.nextLayer

        self.inputLayer.randomizeWeights(prevN_Inp, nextL_Inp)
        self.hiddenLayer1.randomizeWeights(prevN_H1, nextL_H1)
        self.hiddenLayer2.randomizeWeights(prevN_H2, nextL_H2)
        self.outputLayer.randomizeWeights(prevN_O, nextL_O)

    def test(self) -> Tuple[float, float]:
        """
        Runs trained model on test dataset and outputs
        model's accuracy.
        """
        N = len(self.test_X)
        correct = 0
        total_loss = 0
        for i in range(N):
            output, yh = self.predict(self.test_X[i])
            y = np.argmax(self.test_Y[i])
            total_loss += cross_entropy_loss(self.test_Y[i], output)
            if y == yh:
                correct += 1
        accuracy = correct / N
        loss = total_loss / N
        print(f"Accuracy = {accuracy}")
        print(f"Average loss = {loss}")
        return accuracy, loss
