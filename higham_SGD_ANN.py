import os

from numpy import array, zeros, ones, arange, exp, dot, save, pi, linspace,\
                  matrix, ceil, mean, meshgrid, stack, mod
from numpy.random import randn, randint, uniform, normal, seed, shuffle
from numpy.linalg import norm
import matplotlib.pylab as plt
import sys

from matplotlib import rc

# rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern']})
# rc('text', usetex = False)

class GeneralNetwork:
    ''' A GeneralNetwork object contains information about network architecture.
        These include: weights, biases, number of layers, etc. Current
        implementation focuses solely on fully connected networks, though we
        plan to incorporate other variants.

    '''
    def __init__(self, number_of_layers : int, neurons_per_layer : list, verbose = 0, \
                       activation_function = "sigmoid", vis = False):
        '''
        Inputs: number_of_layers  (int)  : specifies number of layers in the network.
                neurons_per_layer (list) : specifies number of neurons in the network.
                                          Each entry gives the total number of neurons
                                          for one layer.
                verbose       (boolean) : output flag for user.
        Outputs: N/A
        Description: Initialization function of a GeneralNetwork object.
        '''

        self.plots_folder = 'plotsANN'
        os.makedirs(self.plots_folder, exist_ok=True)

        self.run = 0

        try:
            if number_of_layers != len(neurons_per_layer):
                print("The length of the neuron vector has to match the number \
                        of layers. Each entry refers to the number of neurons in\
                        that layer!")
                sys.exit()
        except:
            print("The variable number_of_layers must be an integer!")
            print("The variable neurons_per_layer must be a list!")
            sys.exit()

        list_of_weight_matrices = [normal(size=(neurons_per_layer[0], \
                                                neurons_per_layer[0]))]
        # create first entry connecting input layer to first hidden layer
        list_of_bias_vectors    = [normal(size=(neurons_per_layer[0], 1))]
        # Generate first entry in the bias vector. It corresponds to the bias
        # for the first layer


        # Initialization
        self.weights             = list_of_weight_matrices
        self.biases              = list_of_bias_vectors

        self.number_of_layers    = number_of_layers
        self.neurons_per_layer   = neurons_per_layer

        self.verbose             = verbose

        self.activation          = []
        self.delta               = []

        self.activation_function = activation_function
        self.vis                 = vis

        # Populate list of weight matrices per layer
        for i in arange(1, len(neurons_per_layer)):
            self.weights.append(normal(size=(neurons_per_layer[i],\
                                        neurons_per_layer[i - 1])))
            self.biases.append(normal(size=(neurons_per_layer[i], 1)))


    def activate(self, x : array, W : array, b : array):
        '''
        Inputs:    x (1D numpy array) : input from previous layer
                   W (2D numpy array) : weight matrix for current layer
                   b (1D numpy array) : bias vector   for current layer
        Outputs: Sigmoid function output, this is the activation function that
                 is currently applied (between 0 and 1).

        Description: Function calculates activation for a layer, where each
                     entry in the activation array represents the activation for
                     one neuron.
        '''

        if self.activation_function == "sigmoid":
            return 1 / (1 + exp(-(dot(W, x) + b)))
        elif self.activation_function == "relu":
            dummy = dot(W, x) + b
            for i in arange(dummy.shape[0]):
                if dummy[i, 0] > 0:
                    pass
                else:
                    dummy[i, 0] = 0.0
            return dummy
        elif self.activation_function == "leakyrelu":
            dummy = dot(W, x) + b
            for i in arange(dummy.shape[0]):
                if dummy[i, 0] > 0:
                    pass
                else:
                    dummy[i, 0] = 0.01 * dummy[i, 0]
            return dummy

    def predict(self, x : array):
        for s in arange(self.number_of_layers):
            a = self.activate(x, self.weights[s], self.biases[s])
            x = a

        return a

    def gradient(self, x : array):
        '''

        Inputs      : x     (2D numpy array)
        Outputs     : dummy (2D numpy array)
        Description : Computes gradient for each type of activation function

        Some errors still exist here.
        '''
        if self.activation_function == "sigmoid":
            return x * (1 - x)
        elif self.activation_function == "relu":
            dummy = x
            for i in arange(x.shape[0]):
                if dummy[i, 0] > 0:
                    dummy[i, 0] = 1.0

                else:
                    dummy[i, 0] = 0.0
            return dummy
        elif self.activation_function == "leakyrelu":
            dummy = x
            for i in arange(x.shape[0]):
                if dummy[i, 0] > 0:
                    dummy[i, 0] = 1.0

                else:
                    dummy[i, 0] = 0.01 * dummy[i, 0]
            return dummy

        return

    def train(self, Data, eta : float = 0.1, epochs : int = 1000, replacement : bool = 0):
        '''
        Inputs:   Data        (object) : this object contains information
                                         about the training data. The object
                                         has information about LABELLED data
                                         (x_train, y_train) pairs.
                  eta         (float)  : learning rate, i.e. how much we update.
                                         This is currently constant.
                  epochs      (int)    : number of cycles over a complete update
                                         of training data.
                  replacement (int)    : variable decides whether to use each
                                         piece of training data once in an epoch
                                         or sometimes some multiple and others
                                         not at all.
        Outputs:  cost (2D numpy array): contains the cost measurement after each
                                         update, per epoch.

        Description: This function performs gradient descent with respect to the
                     training data. It finds weights and biases which are tuned
                     such that the f_NN(x_train) = y_train, where f_NN is a function
                     outlining the architecture of the neural network.
        '''
        cost   = zeros((epochs, Data.xtrain.shape[1]))
        # create empty 2D numpy array for storage
        xtrain = zeros((Data.xtrain.shape[0], 1))
        # create empty 1D numpy array which will be overwritten with training data
        ytrain = zeros((Data.ytrain.shape[0], 1))
        # create empty 1D numpy array which will be overwritten with training data
        for update in arange(epochs):
            # begin looping over training data epochs many times
            shuffled_ints = array(arange(Data.xtrain.shape[1]))
            # generate integer array to access specific training data per update
            shuffle(shuffled_ints)
            # shuffle integers for no bias toward certain configuration
            for counter in arange(shuffled_ints.shape[0]):
                # begin cycle over training set
                if not replacement:
                        k = shuffled_ints[counter]
                        # without replacement
                else:
                        k = randint(Data.xtrain.shape[1])
                        # with replacement

                # Training Data
                xtrain[:, 0], ytrain[:, 0] = Data.xtrain[:, k], Data.ytrain[:, k]

                # Forward Prop
                for s in arange(self.number_of_layers):
                    self.activation.append(self.activate(xtrain,            \
                        self.weights[s], self.biases[s]))
                    xtrain = self.activation[s]

                #  Back Prop
                self.delta.append(self.gradient(self.activation[-1]) * (self.activation[-1] - ytrain))
                for s in arange(0, self.number_of_layers-1):
                    self.delta.append(self.gradient(self.activation[-2 - s]) * \
                    dot(self.weights[-1  - s].T, self.delta[s]))

                #  Update weights
                self.weights[0]  -= eta * self.delta[-1] * Data.xtrain[:, k].T
                for s in arange(1, self.number_of_layers):
                    if s >= 1:
                        self.weights[s]   -= eta * dot(self.delta[-(s + 1)], self.activation[s - 1].T)

                deltaflip = self.delta
                deltaflip.reverse()
                # reverse to put bias update in loop


                # Update biases
                for s in arange(self.number_of_layers):
                    self.biases[s] -= eta * deltaflip[s]

                # Reset activation and errors for next loop
                self.activation = []
                self.delta      = []

                # Save cost
                cost[update, counter] = self.cost_function(Data)
            if update == epochs - 1 and self.vis == True:
               self.visual(Data)
            if self.verbose:
                # verbosity flag prints to console
                print('Average cost for epoch ', update + 1, 'is :', mean(cost[update, :]))

        return cost

    def train_sequential(self, Data, eta: float = 0.1, epochs: int = 1000, replacement: bool = 0):
        """
        Sequential Gradient Descent Training Function
        """
        cost = zeros((epochs, Data.xtrain.shape[1]))
        xtrain = zeros((Data.xtrain.shape[0], 1))
        ytrain = zeros((Data.ytrain.shape[0], 1))

        for update in range(epochs):
            shuffled_ints = array(arange(Data.xtrain.shape[1]))
            shuffle(shuffled_ints)

            for counter in range(shuffled_ints.shape[0]):
                k = shuffled_ints[counter] if not replacement else randint(Data.xtrain.shape[1])
                xtrain[:, 0], ytrain[:, 0] = Data.xtrain[:, k], Data.ytrain[:, k]

                # Forward Propagation
                activations = [xtrain]
                for s in range(self.number_of_layers):
                    activations.append(self.activate(activations[-1], self.weights[s], self.biases[s]))

                # Backward Propagation (Sequential)
                deltas = [self.gradient(activations[-1]) * (activations[-1] - ytrain)]
                for s in range(self.number_of_layers - 1, 0, -1):
                    # Update Weights **Before** Computing Next Delta
                    self.weights[s] -= eta * dot(deltas[-1], activations[s].T)
                    self.biases[s] -= eta * deltas[-1]

                    # Compute delta using updated weight matrix
                    updated_W = self.weights[s]
                    delta_new = self.gradient(activations[s]) * dot(updated_W.T, deltas[-1])
                    deltas.append(delta_new)

                deltas.reverse()

                # Update first layer weights and biases
                self.weights[0] -= eta * dot(deltas[0], activations[0].T)
                self.biases[0] -= eta * deltas[0]

                cost[update, counter] = self.cost_function(Data)

            if update == epochs - 1 and self.vis == True:
                self.visual(Data)
            if self.verbose:
                # verbosity flag prints to console
                print('Average cost for epoch ', update + 1, 'is :', mean(cost[update, :]))

        return cost


    def visual(self, Data, no_of_points : int = 1000):
        '''
        '''
        x    = zeros((2,1))

        X, Y = meshgrid(linspace(0, 1, no_of_points), linspace(0, 1, no_of_points))
        # create grid of points
        X1, X2 = array(X.ravel()), array(Y.ravel())
        # vectorize
        Stack  = stack((X1, X2), axis = 1)
        # left to right stack, columnwise
        empty  = zeros(X.shape[0]*X.shape[0])
        # matrix of values

        for i in arange(Stack.shape[0]):
            # run over each coordinate
            x[0,0], x[1,0] = Stack[i, 0], Stack[i, 1]
            # input
            Predictions = self.predict(x)
            # predict
            Predictions = array(Predictions[0] >= Predictions[1])
            # create booleans
            if Predictions[0] == True:
                empty[i] = 1
                # assign 1 where true

        Pred = empty.reshape((X.shape[0], X.shape[0]))
        # reshape ready for plotting contour
        import matplotlib.pyplot as plt
        plt.style.use('Solarize_Light2')
        plt.figure()
        # plot figure
        plt.contourf(X, Y, Pred, cmap = 'cividis', alpha = 0.8)
        # plot contour

        # Plot all class 0 points (first 10)
        plt.scatter(Data.xtrain[0, 0:10], Data.xtrain[1, 0:10], marker='^', c='k', lw=3)

        # Plot all class 1 points (last 10)
        plt.scatter(Data.xtrain[0, 10:], Data.xtrain[1, 10:], marker='o', c='w', lw=3)


        # plot second half of training set
        if Data.highamdata:
            plt.savefig(os.path.join(self.plots_folder, f'plot_run_{self.run}.png'))
            self.run += 1
            # save figure
        else:
            plt.savefig(os.path.join(self.plots_folder, f'plot_run_{self.run}.png'))
            self.run += 1
            # save figure
        # plt.show()
        return

    def cost_function(self, Data):
        '''
        Inputs      : Data (object) - contains information on all training
                                      data.

        Outputs     : evaluated cost function.

        Description : This function evaluates the accuracy of the trained
                      function with the actual data.

        '''
        temp_cost = zeros((Data.xtrain.shape[1],1))
        # generate zeros 2D numpy array for storing cost
        x         = zeros((Data.xtrain.shape[0],1))
        # generate zeros 2D numpy array for storing training data before
        # evaluation
        for i in arange(temp_cost.shape[0]):
            # begin loop for computing each individual cost from each piece
            # of training data
            x[0,0], x[1,0] = Data.xtrain[0,i], Data.xtrain[1,i]

            # Feedforward
            for s in arange(self.number_of_layers):
                a = self.activate(x, self.weights[s], self.biases[s])
                x = a

            temp_cost[i] = norm(x.ravel() - Data.ytrain[:, i], 2)

        return norm(temp_cost, 2)**2
class Data:
    ''' Data object contains information about training data. Its initialization
        generates the training data for this particular problem
    '''
    def __init__(self, number_of_data_points : int = 20, highamdata = True):
        '''
        Inputs: number_of_data_points (int)     - total data points without higham
                                                  data.
                highamdata            (boolean) - whether to use data from higham
                                                  paper or not.
        '''

        self.highamdata = highamdata

        if not self.highamdata:
            # generate uniform random data
            self.x = linspace(0, 1, number_of_data_points)
            x1 = zeros((1, number_of_data_points))
            x2 = zeros((1, number_of_data_points))
            x1[0, :] = uniform(0, 1, self.x.shape[0])
            x2[0, :] = uniform(0, 1, self.x.shape[0])
            y1 = zeros((1, int(number_of_data_points / 2)))
            y2 = ones((1, int(number_of_data_points / 2)))
        else:
            # Hardcoded clustered data
            # Class 0 cluster near (0.25, 0.25)
            cluster0_x1 = [0.1, 0.25, 0.1, 0.6, 0.4, 0.25, 0.8, 0.1, 0.7, 0.5]
            cluster0_x2 = [0.1, 0.4, 0.5, 0.9, 0.2, 0.6, 0.1, 0.8, 0.2, 0.9]

            # Class 1
            cluster1_x1 = [0.6, 0.5, 0.8, 0.4, 0.7, 0.35, 0.3, 0.8, 0.4, 0.8]
            cluster1_x2 = [0.35, 0.6, 0.4, 0.4, 0.6, 0.9, 0.8, 0.8, 0.6, 0.9]

            x1 = array(cluster0_x1 + cluster1_x1)
            x2 = array(cluster0_x2 + cluster1_x2)

            y1 = zeros((1, 10))  # Class 0
            y2 = ones((1, 10))  # Class 1

        self.xtrain = zeros((2, number_of_data_points))
        self.ytrain = zeros((2, number_of_data_points))

        # Assign input data
        self.xtrain[0, :], self.xtrain[1, :] = x1, x2

        # Assign labels (one-hot encoding)
        self.ytrain[0, 0:10] = y1  # Class 0
        self.ytrain[0, 10:] = y2  # Class 1
        self.ytrain[1, 0:10] = y2  # Class 0
        self.ytrain[1, 10:] = y1  # Class 1


import os

# import ANN as N
N = sys.modules[__name__]
from numpy.random import seed
import matplotlib.pylab as plt
graph_folder = 'graphsANN'
os.makedirs(graph_folder, exist_ok=True)

def plot_cost_comparison(av_cost_normal, av_cost_sequential, run_number, save_dir):
    """Plot and save the cost comparison between Normal and Sequential Gradient Descent."""
    plt.figure(figsize=(8, 6))
    plt.plot(av_cost_normal, label="Normal GD", color='b')
    plt.plot(av_cost_sequential, label="Sequential GD", color='r')

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Cost', fontsize=14)
    plt.title(f'Training Cost Comparison (Run {run_number})', fontsize=16)
    plt.legend()
    plt.grid(True)
    filename = os.path.join(save_dir, f'graph_run_{run_number}.png')
    plt.savefig(filename)
    plt.close()



for run in range(1):
    # Set random seed for reproducibility
    seed(run+10)

    # Define two networks: one for normal backpropagation, one for sequential GD
    normalGD = N.GeneralNetwork(4, [2, 8, 8, 2], verbose=1, vis=True, activation_function="sigmoid")
    sequentialGD = N.GeneralNetwork(4, [2, 8, 8, 2], verbose=1, vis=True, activation_function="sigmoid")

    # Create training data
    Data = N.Data(20, highamdata=True)

    # Train using Normal Backpropagation
    cost_normal = normalGD.train(Data)
    cost_sequential = sequentialGD.train_sequential(Data)

    # Compute average cost per epoch
    av_cost_normal = cost_normal.mean(axis=1)
    av_cost_sequential = cost_sequential.mean(axis=1)

    # Call plotting function
    plot_cost_comparison(av_cost_normal, av_cost_sequential, run + 1, graph_folder)


# Update loss function for each layer
# Make the network more complex
# Make the training data more complex
# Try different values of eta
# Run multiple iterations