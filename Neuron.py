import numpy as np 

class Neuron():
    def __init__(self, numWeights):
        # init weights going from this neuron to the next layer
        self.weights = np.random.rand(1, numWeights)
    
    # anymore helper functions for the future 