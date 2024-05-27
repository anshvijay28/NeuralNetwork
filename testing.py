# THIS WILL LATER BE A UNIT TEST FILE 
from util import * 
import numpy as np 

neuron_1 = Neuron(outWeights=2)
neuron_2 = Neuron(outWeights=2)

input_layer = [neuron_1, neuron_2]

print(neuron_1)
print(neuron_2)
print()
print(getWeightMatrix(input_layer))