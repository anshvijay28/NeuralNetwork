import numpy as np
from NN import NN
from util import compute_clean_data

train_X, train_Y = compute_clean_data("data/mnist_train_100.csv")
test_X, test_Y = compute_clean_data("data/mnist_test_100.csv")

nn = NN(
    numNeurons=[784, 200, 100, 10],
    alpha=0.01,
    train_X=train_X,
    train_Y=train_Y,
    test_X=test_X,
    test_Y=test_Y,
)

nn.train()
nn.test()
