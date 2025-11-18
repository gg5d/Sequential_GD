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
        # zero accumulators
        dW2 = np.zeros_like(network.W2)
        dW3 = np.zeros_like(network.W3)
        dW4 = np.zeros_like(network.W4)
        db2 = np.zeros_like(network.b2)
        db3 = np.zeros_like(network.b3)
        db4 = np.zeros_like(network.b4)

        # loop over ALL 10 samples
        for k in range(10):
            xvec[0,0], xvec[1,0] = x1[k], x2[k]
            yvec[:,0] = y[:, k]

            # forward
            a2 = activate(xvec, network.W2, network.b2)
            a3 = activate(a2,   network.W3, network.b3)
            a4 = activate(a3,   network.W4, network.b4)

            # backward
            delta4 = a4 * (1 - a4) * (a4 - yvec)
            delta3 = a3 * (1 - a3) * np.dot(network.W4.T, delta4)
            delta2 = a2 * (1 - a2) * np.dot(network.W3.T, delta3)

            # accumulate
            dW2 += delta2 * xvec.T
            dW3 += delta3 * a2.T
            dW4 += delta4 * a3.T
            db2 += delta2
            db3 += delta3
            db4 += delta4

        # ONE weight update per epoch
        network.W2 -= eta * dW2
        network.W3 -= eta * dW3
        network.W4 -= eta * dW4
        network.b2 -= eta * db2
        network.b3 -= eta * db3
        network.b4 -= eta * db4

        cost_value[counter] = cost_function(network.W2, network.W3, network.W4, 
                                            network.b2, network.b3, network.b4, x1, x2, y)
    
    return cost_value

def train_sequential(network, x1, x2, y, eta, Niter, sample_order):
    '''
    Full-batch SEQUENTIAL gradient descent:
    For each epoch, loop over ALL samples.
    But update each layer immediately (W4 -> W3 -> W2) for every sample.
    No accumulation.
    '''
    xvec = np.zeros((2,1))
    yvec = np.zeros((2,1))
    cost_value = np.zeros((Niter,1))

    for counter in np.arange(Niter):

        # FULL BATCH LOOP — go through all 10 samples
        for k in range(10):
            xvec[0,0], xvec[1,0] = x1[k], x2[k]
            yvec[:,0] = y[:, k]

            # forward pass with UPDATED weights
            a2 = activate(xvec, network.W2, network.b2)
            a3 = activate(a2,   network.W3, network.b3)
            a4 = activate(a3,   network.W4, network.b4)

            # sequential backward updates

            # (1) update output layer first
            delta4 = a4 * (1 - a4) * (a4 - yvec)
            network.W4 -= eta * delta4 * a3.T
            network.b4 -= eta * delta4

            # (2) now use UPDATED W4 to compute delta3
            delta3 = a3 * (1 - a3) * np.dot(network.W4.T, delta4)
            network.W3 -= eta * delta3 * a2.T
            network.b3 -= eta * delta3

            # (3) now use UPDATED W3 to compute delta2
            delta2 = a2 * (1 - a2) * np.dot(network.W3.T, delta3)
            network.W2 -= eta * delta2 * xvec.T
            network.b2 -= eta * delta2

        # cost AFTER finishing all samples (1 epoch)
        cost_value[counter] = cost_function(
            network.W2, network.W3, network.W4,
            network.b2, network.b3, network.b4,
            x1, x2, y
        )

    return cost_value

# stochastic sequential gradient descent
# def train_sequential(network, x1, x2, y, eta, Niter, sample_order):
#     '''
#     Sequential gradient descent: update each layer as we compute its gradient, 
#     then use updated weights for previous layer gradients
#     '''
#     xvec = np.zeros((2,1))
#     yvec = np.zeros((2,1))
#     cost_value = np.zeros((Niter,1))
    
#     for counter in np.arange(Niter):
#         k = sample_order[counter]
        # xvec[0,0], xvec[1,0] = x1[k], x2[k]
        # yvec[:,0] = y[:, k]

        # # forward pass
        # a2 = activate(xvec, network.W2, network.b2)
        # a3 = activate(a2,   network.W3, network.b3)
        # a4 = activate(a3,   network.W4, network.b4)

        # # backward pass - update layer 4 first
        # delta4 = a4 * (1 - a4) * (a4 - yvec)
        # network.W4 -= eta * delta4 * a3.T
        # network.b4 -= eta * delta4

        # # then compute delta3 using updated W4, and update layer 3
        # delta3 = a3 * (1 - a3) * np.dot(network.W4.T, delta4)
        # network.W3 -= eta * delta3 * a2.T
        # network.b3 -= eta * delta3

        # # finally compute delta2 using updated W3, and update layer 2
        # delta2 = a2 * (1 - a2) * np.dot(network.W3.T, delta3)
        # network.W2 -= eta * delta2 * xvec.T
        # network.b2 -= eta * delta2

#         cost_value[counter] = cost_function(network.W2, network.W3, network.W4, 
#                                             network.b2, network.b3, network.b4, x1, x2, y)
    
#     return cost_value




import matplotlib.pylab as plt

x1 = np.array([0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7])
x2 = np.array([0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6])

y           = np.zeros((2, 10))
y[0:1, 0:5] = np.ones((1,  5))
y[0:1, 5: ] = np.zeros((1, 5))
y[0:1, 5: ] = np.zeros((1, 5))
y[1: , 5: ] = np.ones((1,  5))

eta = 0.25
Niter = 10000

# Generate sample order once so both models use the same sequence
seed(22)
sample_order = randint(10, size=Niter)

# Train network A with backpropagation
seed(22)
networkA = NeuralNetwork()
cost_value_A = train_backprop(networkA, x1, x2, y, eta, Niter, sample_order)

# Train network B with sequential gradient descent
seed(22)
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
empty  = np.zeros(200*200)

xvec = np.zeros((2,1))
for i in np.arange(XTest.shape[0]):

    xvec[0,0], xvec[1,0] = XTest[i, 0], XTest[i, 1]


    YPredictions = predict(networkA.W2, networkA.W3, networkA.W4, 
                          networkA.b2, networkA.b3, networkA.b4, xvec)
    YPredictions = np.array(YPredictions[0] >= YPredictions[1])


    if YPredictions[0] == True:
        empty[i] = 1


YPred = empty.reshape((200, 200))
import matplotlib.pyplot as plt
plt.figure()
#plt.imshow(YPred)
plt.contourf(X, Y, YPred)

plt.scatter(x1[0:5], x2[0:5], marker='^', lw=5)
plt.scatter(x1[5:],  x2[5:], marker='o', lw=5)
# plt.show()