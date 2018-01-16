"""
# 1. Define the neural structure
# 2. Initialize the model parameters
# 3. Loop:
#   A. Implement forward propagation
#   B. Compute loss
#   C. Implement backward propagation
#   D. Update parameters (gradient descent)
#
# Steps A-C can be merged together into one function (nn_model) for future predictions on new data
"""

import numpy as np


def layer_sizes(X, Y, n_h=4):
    """
    :param X: input data set of size m x n (features x examples)
    :param Y: labels of data set of size k x n (categories x examples)
    :param layer1: number of neurons in layer 1
    :return: n_x: size of the input layer
    :return: n_h: size of the hidden layer
    :return: n_y: size of the output layer
    """
    n_x = X.shape[0]
    n_h = n_h
    n_y = Y.shape[0]

    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    """
    :param n_x: size of the input layer
    :param n_h: size of the hidden layer
    :param n_y: size of the output layer
    :return: parameters: dictionary containing: - W1 (weight matrix shape n_h x n_x)
                                            - b1 (bias vector shape n_h x 1)
                                            - W2 (weight matrix shape n_y x n_h)
                                            - b2 (bias vector shape n_y x 1)
    """
    np.random.seed(80537)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    """
    :param X: input data of size (n_x, m)
    :param parameters: python dictionary containing the initialized parameters
    :return: A2: sigmoid output of second activation
    :return: cache: a dictionary containing Z1, A1, Z2, and A2
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def sigmoid(x):
    """
    :param x: A scalar or numpy array of any size
    :return: s: sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))

    return s


def compute_cost(A2, Y, parameters):
    """

    :param A2: sigmoid output of the second activation of shape (1 x examples)
    :param Y: true labels of shape (categories x examples)
    :param parameters: python dictionary containing your parameters W1, b1, W2 and b2
    :return: cost: cross-entropy cost
    """
    m = Y.shape[1]

    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), (1 - Y))
    cost = -1/m * np.sum(logprobs)
    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect. E.g., turns [[17]] into 17
    assert (isinstance(cost, float))

    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    :param parameters: dictionary containing parameters
    :param cache: dictionary containg Z1, A1, Z2, A2
    :param X: input data
    :param Y: true labels
    :return: grads: dictionary containing gradients with respect to different parameters
    """
    m = X.shape[1]

    # retrieve stored parameters
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    # calculate backward propagation
    dZ2 = A2 - Y
    dW2 = 1/m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    """
    :param parameters: dictionary containing parameters
    :param grads: dictionary containing gradients
    :param learning_rate:
    :return: parameters: dictionary containing upgraded parameters
    """
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # retrieve gradients
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # update parameters
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """

    :param X: dataset
    :param Y: labels
    :param n_h: size of the hidden layer
    :param num_iterations: number of epochs
    :param print_cost: if True, print the cost every 1000 iterations
    :return: parameters: parameters learnt by the model
    """
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)

    # loop gradient descent:
    for i in range(0, num_iterations):
        # forward propagation
        A2, cache = forward_propagation(X, parameters)

        # Cost function
        cost = compute_cost(A2, Y, parameters)

        # Backward propagation
        grads = backward_propagation(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    """

    :param parameters: dictionary containing parameters
    :param X: input datase
    :return: predictions: vector of predictions of the model
    """
    # compute probabilities using forward propagation
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions

