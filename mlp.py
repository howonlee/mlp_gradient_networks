#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Multi-layer perceptron
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.

# shamelessly taken and modified by Howon Lee in 2016, cuz BSD allows that
# only Howon doesn't know which version of the BSD license Howon should replicate
# so whatever version NP Rougier wanted, consider it replicated
# dont think that NP Rougier has heard of Howon, because he prolly hasn't
# nor vice versa, now that Howon thinks about it
# -----------------------------------------------------------------------------

# This is an implementation of the multi-layer perceptron with retropropagation
# learning.

# modified by Howon Lee to make a point about the fractal nature of backprop
# play with the params as you like
# -----------------------------------------------------------------------------
import numpy as np
import numpy.random as npr
import cPickle
import gzip

def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2

class MLP:
    '''
    Multi-layer perceptron class.
    This is used via SGD only in the MNIST thing Howon rigged up
    '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0]+1))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0,]*len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        ''' Reset weights '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

        # Return output
        return self.layers[-1]


    def propagate_backward(self, target, lrate=1.0, grad_filename="grad_mat", weight_filename="weight_mat"):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0,delta)

        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            if i == 0:
                np.save("layer", self.layers[i])
                np.save("delta", deltas[i])
                np.save(grad_filename, dw)
                print "layer, delta, dw saved"
            self.weights[i] += lrate*dw
            if i == 0:
                np.save(weight_filename, self.weights[i])
                print "weights saved"
            self.dw[i] = dw

        # Return error
        return (error**2).sum()


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt

    def bitfield(n):
        """
        Don't care about the MLP results, so don't care about the output patterns
        So let's do this
        """
        return np.array([n >> i & 1 for i in range(7,-1,-1)])

    def onehots(n):
        arr = np.array([0.0] * 10)
        arr[n] = 1.0
        return arr

    def create_mnist_samples(filename="mnist.pkl.gz"):
        # only 500 datapoints, we don't even really need that many
        # we are not using this net, so can just use training only
        samples = np.zeros(50000, dtype=[('input',  float, 784), ('output', float, 10)])
        with gzip.open(filename, "rb") as f:
            train_set, valid_set, test_set = cPickle.load(f)
            for x in xrange(50000):
                samples[x] = train_set[0][x], onehots(train_set[1][x])
        return samples, 784

    def create_cifar_samples(filename="cifar-10-batches-py/data_batch_1"):
        samples = np.zeros(10000, dtype=[('input',  float, 3072), ('output', float, 10)])
        with open(filename, "rb") as f:
            cifar_dict = cPickle.load(f)
            for x in xrange(10000):
                # CIFAR is uint8s, but I would like floats
                samples[x] = cifar_dict["data"][x] / 256.0, onehots(cifar_dict["labels"][x])
        return samples, 3072

    def create_random_samples():
        # only 500 datapoints, we don't even really need that many
        # we are not using this net, so can just use training only
        # big numerical problems with this one!
        samples = np.zeros(50000, dtype=[('input',  float, 784), ('output', float, 10)])
        for x in xrange(50000):
            # try it with either one
            samples[x] = npr.random(784), onehots(5)
            # samples[x] = npr.random(784), onehots(npr.random_integers(1, 10))
        return samples, 784

    print "learning the patterns..."
    samples, dims = create_mnist_samples()
    network = MLP(dims, dims, 10)
    for i in range(500):
        print "pattern: ", i
        n = np.random.randint(samples.size)
        network.propagate_forward(samples['input'][n])
        network.propagate_backward(samples['output'][n])
    print "and then we don't use the net for anything"
