# THIS WILL LATER BE A UNIT TEST FILE 
from util import * 
import numpy as np 

a = np.random.rand(10)
aN = softmax(a)

print(a)
print(aN)
print(np.argmax(aN))