# Neural Network from scratch
Building a multi-layer Neural Network with no machine learning libraries.
<br />
<br />
This neural network was designed for multi-class classification on the mnist dataset for determining the value of hand drawn digits. The neural network has 4 layers with an input layer, 2 hidden layers, and an output layer. There are 784 neurons in the input layer, 200 neurons in the first hidden layer, 100 neurons in the second hidden layer, and 10 neurons in the output layer. 
<br />
<br />
The input to the network are images (28 x 28 pixels), which have been flattened to vectors of length 784. Each value within the input vectors are gray scale values (0 - 255) of each pixel, which have been normalized (0 - 1). The output of the network is a vector of length 10 where each value in the vector represents the probability of the image representing digit i, where i is the index of the vector. 

![example input](./images/example_image.png)

The activation function at each hidden layer is sigmoid, and the activation function at the output layer is a softmax operation. The softmax function will ensure the output vector is a probability vector. This allows the network to determine the digit by the one with the highest probability.
```
y = np.argmax(output)
```
This network uses stochastic gradient descent for training, which means the weights of the network are updated for each forward and backwards pass. Additionally, an optimal learning rate, a hyperparameter, was found through k fold cross validation. This process helps a network determine hyperparameters by going through a training and testing loops for each potential value of the hyperparameter. During each iteration the training and validation data change. The hyperparameter which yields the most accuracy is chosen. 
<br />
<br />
After iterating through 6 potential learning rates, the best one was found to be 0.0075. After training the network with this learning rate it yielded an accuracy of 96%. Once the correct learning rate was determined training was done with 15 epochs. A lower number of epochs resulted in a lower accuracy of 94%. Of course, the number of epochs is another hyperparameter that can be determined through another round of cross validation. Alternately, one could determine the best (learning rate, epoch) pair through nested cross validation. This would mean for each learning rate value you'd train the network with every potential epoch value. 
