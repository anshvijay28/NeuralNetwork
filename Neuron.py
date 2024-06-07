#### Deprecated ####
"""
It was inefficient to split weights by neuron because for
forward or backward passes you'd need to gather all weights 
into a matrix anyways. If there are n training examples, 
m neurons in each layer, and j layers, then this leads to 
an O(mnj) ==> O(n^3) time complexity. Separating the neural 
network into layers rather than neurons allows us to skip
the O(n^3) time complexity entirely because each layer has 
a .weights (np.ndarray) attribute. The only time 
overhead is from matrix multiplication, which I'm pretty 
sure has no alternative.
"""
import numpy as np 

class Neuron():
    def __init__(self, outWeights = 0, inWeights = 1) -> None:
        self.weights = np.random.uniform(0.01, 1 / np.sqrt(inWeights), size=outWeights)
        self.outWeights = outWeights
        self.inWeights = inWeights
    def __repr__(self):
        return f'Weights: {self.weights}\nOut weights: {self.outWeights}\nIn weights: {self.inWeights}'