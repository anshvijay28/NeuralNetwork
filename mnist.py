import numpy as np
from NN import NN
from util import compute_clean_data

train_X, train_Y = compute_clean_data("data/mnist_train.csv")
test_X, test_Y = compute_clean_data("data/mnist_test.csv")

nn = NN(
    numNeurons=[784, 200, 100, 10],
    alpha=0.01,
    train_X=train_X,
    train_Y=train_Y,
    test_X=test_X,
    test_Y=test_Y,
)
yh = nn.predict(test_X[0])
y = np.argmax(test_Y[0])

print(yh)
print(y)

print(test_Y[0])


# nn.train()
# nn.test()

# TODO
    # 1) Research why its failing 
    # 2) Research proper ways to train data 
    # 3) Research ways to pick best hyperparameter 
