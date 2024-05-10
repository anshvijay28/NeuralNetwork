import numpy as np 

class Neuron():
    def __init__(self, outWeights = 0, inWeights = 1) -> None:
        self.weights = np.random.uniform(0.01, 1 / np.sqrt(inWeights), size=outWeights)