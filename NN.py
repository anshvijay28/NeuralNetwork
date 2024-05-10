import numpy as np
from Neuron import *
from util import *


class NN:
    def __init__(self, numNeurons=[10, 8, 8, 10]) -> None:
        self.layers = []
        # init bias weights
        self.biasWeights = []
        for _ in range(len(numNeurons) - 1):
            weights = np.random.uniform(0.01, 1, size=numNeurons[i + 1])
            self.biasWeights.append(weights)
        # build input layer
        inputLayer = []
        for _ in range(numNeurons[0]):
            inputLayer.append(Neuron(outWeights=numNeurons[1]))
        # build hidden layers
        for i in range(1, len(numNeurons) - 1):
            layer = []
            for _ in range(numNeurons[i]):
                outWeights = numNeurons[i + 1]
                inWeights = numNeurons[i - 1] + 1  # add 1 for bias
                layer.append(Neuron(outWeights=outWeights, inWeights=inWeights))
            self.layers.append(layer)
        # build output layer
        outputLayer = []
        for _ in range(numNeurons[-1]):
            outputLayer.append(Neuron())
        self.layers.append(outputLayer)

    def forward(self, inp: np.ndarray) -> List[np.ndarray]:
        outs = []  # DOES NOT INCLUDE INPUT
        for i in range(len(self.layers) - 1):
            weights = getWeightMatrix(self.layers[i])
            if i == len(self.layers) - 2:
                # do softmax on last iteration
                inp = softmax((weights @ inp) + self.biasWeights[i])
            else:
                inp = sigmoid((weights @ inp) + self.biasWeights[i])
            outs.append(inp)
        return outs  # length = len(self.layers) - 1

    def backProp(target: np.ndarray, self, outs: List[np.ndarray]):
        loss = cross_entropy_loss(target=target, output=outs[-1])
        
        pass
