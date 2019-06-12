# Convolutional Neural Network

Build a convolutional neural network for the hand-written digits recognition task with the MNIST data set. Use the cross-entropy error function, and run 5 epochs. Give the recognition accuracy rate and show the confusion matrix, both for the test set.

## Structure

1. Input layer
2. 2d-convolutional layer: filter size 3x3, depth=32, ReLU activation function
3. 2x2 max pooling layer
4. 2d-convolutional layer: filter size 3x3, depth=64, ReLU activation function
5. 2x2 max pooling layer
6. 2d-convolutional layer: filter size 3x3, depth=64, ReLU activation function
7. Fully-connected layer: 64 units, ReLU activation function
8. Fully-connected layer: 64 units, ReLU activation function
9. (Output layer) Fully-connected layer: 10 units, softmax activation function

## Get started

> First time

```bash
make all
```

> After first time

```bash
make start
```


## Dependency:

- Python 3+
- numpy
- sklearn
- keras
