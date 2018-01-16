import tensorflow as tf
import numpy as np


# define, using operations, the feed-forward calculation of layer values
def feed_forward(x, weights, biases):
    # First hidden fully connected layer
    z1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    a1 = tf.nn.sigmoid(z1)
    # Second hidden fully connected layer
    z2 = tf.add(tf.matmul(a1, weights['h2']), biases['b2'])
    a2 = tf.nn.sigmoid(z2)
    # Output fully connected layer with a neuron for each class
    out = tf.add(tf.matmul(a2, weights['out']), biases['out'])
    return out


def next_batch(num, data, labels):
    idx = np.arange(0, data.shape[0])
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = data[idx]
    labels_shuffle = labels[idx]
    labels_shuffle = np.asarray(labels_shuffle.reshape(labels_shuffle.shape[0], labels_shuffle.shape[1]))

    return data_shuffle, labels_shuffle


def predict(x, weights, biases):
    # First hidden fully connected layer
    z1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    a1 = tf.nn.sigmoid(z1)
    # Second hidden fully connected layer
    z2 = tf.add(tf.matmul(a1, weights['h2']), biases['b2'])
    a2 = tf.nn.sigmoid(z2)
    # Output fully connected layer with a neuron for each class
    out = tf.add(tf.matmul(a2, weights['wout']), biases['bout'])
    return out
