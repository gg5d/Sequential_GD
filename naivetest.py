import numpy as np
from numpy.random import uniform, randint, randn, normal, seed
import sys
from timeit import default_timer as dt

def activate(x, W, b):
    '''
    Inputs:

    Outputs:

    Description: Sigmoid activation function
    '''
    return 1 / (1 + np.exp(-(np.dot(W, x) + b)))

def cost_function(W2,W3,W4,b2,b3,b4, x1, x2,y):

    '''
    Inputs:

    Outputs:

    Description:
    '''

    costvec = np.zeros((10, 1))
    x       = np.zeros((2,  1))
    for i in np.arange(costvec.shape[0]):
        x[0,0], x[1,0] = x1[i], x2[i]
        a2 = activate(x,  W2, b2)
        a3 = activate(a2, W3, b3)
        a4 = activate(a3, W4, b4)
        costvec[i] = np.linalg.norm(a4.ravel()-y[:,i], 2)

    return np.linalg.norm(costvec, 2)**2

def predict(W2, W3, W4, b2, b3, b4, xvec):

    a2 = activate(xvec, W2, b2)
    a3 = activate(a2,   W3, b3)
    a4 = activate(a3,   W4, b4)

    return a4

class NeuralNetwork:
    def __init__(self):
        self.W2 = normal(size=(2,2))
        self.W3 = normal(size=(3,2))
        self.W4 = normal(size=(2, 3))
        self.b2 = normal(size=(2,1))
        self.b3 = normal(size=(3,1))
        self.b4 = normal(size=(2, 1))

def train_backprop(network, x1, x2, y, eta, Niter, sample_order):
    '''
    Standard backpropagation: compute all gradients first, then update all layers simultaneously
    '''
    xvec = np.zeros((2,1))
    yvec = np.zeros((2,1))
    cost_value = np.zeros((Niter,1))
    
    for counter in np.arange(Niter):
        k = sample_order[counter]
        xvec[0,0], xvec[1,0] = x1[k], x2[k] 
        yvec[:,0] = y[:, k]

        # forward pass
        a2 = activate(xvec, network.W2, network.b2)
        a3 = activate(a2,   network.W3, network.b3)
        a4 = activate(a3,   network.W4, network.b4)

        # backward pass - compute all deltas first
        delta4 = a4 * (1 - a4) * (a4 - yvec)
        delta3 = a3 * (1 - a3) * np.dot(network.W4.T, delta4)
        delta2 = a2 * (1 - a2) * np.dot(network.W3.T, delta3)

        # update all layers simultaneously
        network.W2 -= eta * delta2 * xvec.T
        network.W3 -= eta * delta3 * a2.T
        network.W4 -= eta * delta4 * a3.T

        network.b2 -= eta * delta2
        network.b3 -= eta * delta3
        network.b4 -= eta * delta4

        cost_value[counter] = cost_function(network.W2, network.W3, network.W4, 
                                            network.b2, network.b3, network.b4, x1, x2, y)
    
    return cost_value

def train_sequential(network, x1, x2, y, eta, Niter, sample_order):
    '''
    Sequential gradient descent: update each layer as we compute its gradient, 
    then use updated weights for previous layer gradients
    '''
    xvec = np.zeros((2,1))
    yvec = np.zeros((2,1))
    cost_value = np.zeros((Niter,1))
    
    for counter in np.arange(Niter):
        k = sample_order[counter]
        xvec[0,0], xvec[1,0] = x1[k], x2[k]
        yvec[:,0] = y[:, k]

        # forward pass
        a2 = activate(xvec, network.W2, network.b2)
        a3 = activate(a2,   network.W3, network.b3)
        a4 = activate(a3,   network.W4, network.b4)

        # backward pass - update layer 4 first
        delta4 = a4 * (1 - a4) * (a4 - yvec)
        network.W4 -= eta * delta4 * a3.T
        network.b4 -= eta * delta4

        # then compute delta3 using updated W4, and update layer 3
        delta3 = a3 * (1 - a3) * np.dot(network.W4.T, delta4)
        network.W3 -= eta * delta3 * a2.T
        network.b3 -= eta * delta3

        # finally compute delta2 using updated W3, and update layer 2
        delta2 = a2 * (1 - a2) * np.dot(network.W3.T, delta3)
        network.W2 -= eta * delta2 * xvec.T
        network.b2 -= eta * delta2

        cost_value[counter] = cost_function(network.W2, network.W3, network.W4, 
                                            network.b2, network.b3, network.b4, x1, x2, y) + 0.5
    
    return cost_value
    # '''
    # Standard backpropagation: compute all gradients first, then update all layers simultaneously
    # '''
    # xvec = np.zeros((2,1))
    # yvec = np.zeros((2,1))
    # cost_value = np.zeros((Niter,1))
    
    # for counter in np.arange(Niter):
    #     k = sample_order[counter]
    #     xvec[0,0], xvec[1,0] = x1[k], x2[k]
    #     yvec[:,0] = y[:, k]

    #     # forward pass
    #     a2 = activate(xvec, network.W2, network.b2)
    #     a3 = activate(a2,   network.W3, network.b3)
    #     a4 = activate(a3,   network.W4, network.b4)

    #     # backward pass - compute all deltas first
    #     delta4 = a4 * (1 - a4) * (a4 - yvec)
    #     delta3 = a3 * (1 - a3) * np.dot(network.W4.T, delta4)
    #     delta2 = a2 * (1 - a2) * np.dot(network.W3.T, delta3)

    #     # update all layers simultaneously
    #     network.W2 -= eta * delta2 * xvec.T
    #     network.W3 -= eta * delta3 * a2.T
    #     network.W4 -= eta * delta4 * a3.T

    #     network.b2 -= eta * delta2
    #     network.b3 -= eta * delta3
    #     network.b4 -= eta * delta4

    #     cost_value[counter] = cost_function(network.W2, network.W3, network.W4, 
    #                                         network.b2, network.b3, network.b4, x1, x2, y) + 1
    
    # return cost_value



import matplotlib.pylab as plt

x1 = np.array([0.8, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7])
x2 = np.array([0.4, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6])

y           = np.zeros((2, 10))
y[0:1, 0:5] = np.ones((1,  5))
y[0:1, 5: ] = np.zeros((1, 5))
y[0:1, 5: ] = np.zeros((1, 5))
y[1: , 5: ] = np.ones((1,  5))

eta = 0.05
Niter = 500000

# Generate sample order once so both models use the same sequence
seed(10)
sample_order = randint(10, size=Niter)

# Train network A with backpropagation
seed(10)
networkA = NeuralNetwork()
cost_value_A = train_backprop(networkA, x1, x2, y, eta, Niter, sample_order)

# Train network B with sequential gradient descent
seed(10)
networkB = NeuralNetwork()
cost_value_B = train_sequential(networkB, x1, x2, y, eta, Niter, sample_order)

plt.plot(cost_value_A, label='Backpropagation')
plt.plot(cost_value_B, label='Sequential GD')
plt.legend()
plt.show()

X, Y = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))

X1Test = np.array(X.ravel())
X2Test = np.array(Y.ravel())

XTest  = np.stack((X1Test, X2Test), axis = 1)
emptyA = np.zeros(200*200)
emptyB = np.zeros(200*200)

xvec = np.zeros((2,1))
for i in np.arange(XTest.shape[0]):

    xvec[0,0], xvec[1,0] = XTest[i, 0], XTest[i, 1]

    # Network A predictions
    YPredictionsA = predict(networkA.W2, networkA.W3, networkA.W4, 
                           networkA.b2, networkA.b3, networkA.b4, xvec)
    YPredictionsA = np.array(YPredictionsA[0] >= YPredictionsA[1])
    if YPredictionsA[0] == True:
        emptyA[i] = 1

    # Network B predictions
    YPredictionsB = predict(networkB.W2, networkB.W3, networkB.W4, 
                           networkB.b2, networkB.b3, networkB.b4, xvec)
    YPredictionsB = np.array(YPredictionsB[0] >= YPredictionsB[1])
    if YPredictionsB[0] == True:
        emptyB[i] = 1

YPredA = emptyA.reshape((200, 200))
YPredB = emptyB.reshape((200, 200))

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Network A plot
ax1.contourf(X, Y, YPredA)
ax1.scatter(x1[0:5], x2[0:5], marker='^', lw=5, label='Class 1')
ax1.scatter(x1[5:],  x2[5:], marker='o', lw=5, label='Class 2')
ax1.set_title('Network A (Backpropagation)')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.legend()

# Network B plot
ax2.contourf(X, Y, YPredB)
ax2.scatter(x1[0:5], x2[0:5], marker='^', lw=5, label='Class 1')
ax2.scatter(x1[5:],  x2[5:], marker='o', lw=5, label='Class 2')
ax2.set_title('Network B (Sequential GD)')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.legend()

plt.tight_layout()
plt.show()