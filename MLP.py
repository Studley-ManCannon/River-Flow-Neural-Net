import numpy as np
import matplotlib.pyplot as plt
import time

class ANN(object):

    def __init__(self, sizes):
        """
        Initialises the neural net with a list input. The list contains the number of layers and the
        number of nodes in each layer.
        ``num_layers``: an int representing the number of layers in the neural net.
        ``sizes``: array layout of neural net.
        ``biases``: a list of numpy vectors where each vector contains random integers within standard
        normalisation values.
        ``weights``: a list of numpy matrices constructed similar to the biases list.
        ``prev_bias_deltas``: a list of numpy vectors where all values are set to 0. This is used to
        track bias deltas to calculate momentum.
        ``prev_weight_deltas``: a list of numpy matrices where all values are set to 0. This is used to
        track weight deltas to calculate momentum.
        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(i, 1) for i in sizes[1:]]
        self.weights = [np.random.randn(j, i)for i, j in zip(sizes[:-1], sizes[1:])]
        self.prev_bias_deltas = [np.zeros(bias.shape) for bias in self.biases]
        self.prev_weight_deltas = [np.zeros(weight.shape) for weight in self.weights]

    def feedForward(self, activation):
        """
        Takes an input vector and feeds it through the neural net. For each set of biases and weights, the
        new activation value is calculated from a sigmoid activation
        function on the dot product of weights and inputs, added
        with the bias vectors.
        ``activation``: a numpy vector containing all activation values for the input layer of the ANN.
        """

        for bias, weight in zip(self.biases, self.weights):
            activation = sigmoid(np.dot(weight, activation) + bias)
        return activation

    def backProp(self, activation, target_value):
        """
        Takes a set of inputs and the target output for those inputs and calculates the delta gradient between them.
        This is used further in the code to adjust the weights and biases.
        ``delta_bias_gradient``: a list of numpy vectors where all values are set to 0. This is used to
        track bias deltas gradients.
        ``delta_weight_gradient``: a list of numpy matrices where all values are set to 0. This is used to
        track weight deltas gradients
        .
        """

        # Two variables are declared in the same way as ``self.prev_bias_deltas`` and ``self.prev_weight_deltas``.
        delta_bias_gradient = [np.zeros(bias.shape) for bias in self.biases]
        delta_weight_gradient = [np.zeros(weight.shape) for weight in self.weights]


        # A feed-forward algorithm is created and for each set of weights and biases in the ANN, the updated input
        # values, and their activated values are saved to the ``new_vectors`` and ``activations`` arrays
        # respectively.
        activations = [activation]
        new_vectors = []
        for bias, weight in zip(self.biases, self.weights):
            new_vector = np.dot(weight, activation) + bias
            new_vectors.append(new_vector)
            activation = sigmoid(new_vector)
            activations.append(activation)
        
        # An activation delta is calculated. The last element of the ``activations`` array (the final output
        # of the ANN) and the target value are put through a function to calculate their cost
        # derivative, which is then multiplied with the prime activated value of the final vector in
        # ``new_vectors``.
        activation_delta = cost_derivative(activations[-1], target_value)*sigmoid_prime(new_vectors[-1])
        delta_bias_gradient[-1] = activation_delta
        delta_weight_gradient[-1] = np.dot(activation_delta, activations[-2].T)

        # Loop idea courtesy of Michael Nielsen (https://twitter.com/michael_nielsen)
        # Python can take negative indices when calling an element of a list.
        # Because of this, we can loop negatively through the neural net easily.
        for i in range(2, self.num_layers):
            # Working backwards through each layer, the current vector is saved by
            # accessing the negative element at location i. This is only possible
            # thanks to python's ability to use negatives to access array locations.
            # A new delta is calculated (which is then used recursively in this loop).
            # Still working backwards in each list, the activation delta is saved to the
            # bias gradient list and the dot product value is saved to the weight gradient
            # list.
            new_vector = new_vectors[-i]
            activation_delta = np.dot(self.weights[-i+1].T, activation_delta) * sigmoid_prime(new_vector)
            delta_bias_gradient[-i] = activation_delta
            delta_weight_gradient[-i] = np.dot(activation_delta, activations[-i-1].T)
        return (delta_bias_gradient, delta_weight_gradient)

    def grad_descent(self, training_data, epochs, mini_batch_size, learning_rate, momentum_rate=None, validation_data=None):
        """Stochastic Gradient Descent."""
        """Using mini-batches, the neural network is trained by manipulating the weights and biases.
        Each mini batch is processed to update the weights and biases.
        Additionally, an RMSE value is calculated at the end of each epoch."""

        training_data = list(training_data)
        training_size = len(training_data)

        # Validation allows us to track the progress of the neural net at regular intervals.
        # It would be unwise to test the neural net on the training data as we want fresh,
        # unprocessed data. Re-using training data could lead to overfitting.
        if validation_data:
            validation_data = list(validation_data)
            val_size = len(validation_data)

        for i in range(0, epochs):
            # The user inputs a mini-batch size. Mini-batches are created by splitting the
            # training data at regular intervals of ``mini_batch_size`` 
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, training_size, mini_batch_size)]
            for mini_batch in mini_batches:
                # If the momentum rate is given, run a momentum-based algorithm instead of
                # a regular stochacstic update.
                if momentum_rate:
                    self.momentum_update_batch(mini_batch, learning_rate, momentum_rate)
                else:
                    self.stochastic_update_batch(mini_batch, learning_rate)

            # If validation data is supplied, calculate the error margin.
            if validation_data:
                print("Epoch {}: Error Margin = {}%".format(i, 100*self.validation(validation_data)/val_size))
            else:
                print("Epoch {} complete".format(i))

            # Two empty arrays are initialised to store the target values for the validation
            # data and the output produced by the neural net. When all of these values are
            # collected, an RMSE function computer the RMSE and appends it to ``rmse_vals``
            # array.
            the_targets = []
            bot_output = []
            for (x,y) in validation_data:
                output = NeuralNet.feedForward(x)
                bot_output.append(output[0][0]*max_tar)
                the_targets.append(y*max_tar)
            rmse_vals.append(rmse(bot_output, the_targets))


    def stochastic_update_batch(self, mini_batch, learning_rate):
        """Regular Batch Updating"""
        """For a given batch, backpropagation is used to get delta gradients for each of
        the inputs and targets."""

        bias_gradient = [np.zeros(bias.shape) for bias in self.biases]
        weight_gradient = [np.zeros(weight.shape) for weight in self.weights]
        for x, y in mini_batch:
            delta_bias_gradient, delta_weight_gradient = self.backProp(x, y)
            updated_biases =  []
            updated_weights = []
            # Each bias and weight is added together with its respective delta value to
            # create an updated bias/weight. These values are appended to an array, and
            # once all values are collected, the ``bias_gradient`` and ``weight_gradient``
            # numpy arrays are updated.
            for bias, delta in zip(bias_gradient, delta_bias_gradient):
                updated_biases.append(bias + delta)
            bias_gradient = updated_biases

            for weight, delta in zip(weight_gradient, delta_weight_gradient):
                updated_weights.append(weight + delta)
            weight_gradient = updated_weights

        # Finally, biases and weights in the neural network are updated. The equation for this is:
        # Each new bias equals the new bias multiplied by the learning rate which is divided by the
        # length of the mini batches. The division by the length of the mini batch is implemented to
        # account for the sample size.
        self.biases = [bias - new_bias*(learning_rate/len(mini_batch)) for bias, new_bias in zip(self.biases, bias_gradient)]
        self.weights = [weight - new_weight*(learning_rate/len(mini_batch)) for weight, new_weight in zip(self.weights, weight_gradient)]

    def momentum_update_batch(self, mini_batch, learning_rate, momentum_rate):
        """Momentum-based Batch Updating"""
        """For a given batch, backpropagation is used to get delta gradients for each of
        the inputs and targets.
        Unlike with the stochastic batch updating function, this function takes the previous
        delta values into account when calculating new bias/weight values."""

        bias_gradient = [np.zeros(bias.shape) for bias in self.biases]
        weight_gradient = [np.zeros(weight.shape) for weight in self.weights]
        for x, y in mini_batch:
            delta_bias_gradient, delta_weight_gradient = self.backProp(x, y)
            updated_biases =  []
            updated_weights = []
            updated_bias_prev = []
            updated_weight_prev = []
            for new_bias, delta, prev_delta in zip(bias_gradient, delta_bias_gradient, self.prev_bias_deltas):
                # We add the new_bias and delta together as normal, but also add the momentum
                # rate multiplied by the previous delta.
                updated_biases.append(new_bias + delta + (momentum_rate*prev_delta))
                updated_bias_prev.append(delta)
            bias_gradient = updated_biases
            self.prev_bias_deltas = updated_bias_prev
            
            for new_weight, delta, prev_delta in zip(weight_gradient, delta_weight_gradient, self.prev_weight_deltas):
                updated_weights.append(new_weight + delta + (momentum_rate*prev_delta))
                updated_weight_prev.append(delta)
            weight_gradient = updated_weights
        self.prev_weight_deltas = updated_weight_prev

        self.biases = [bias - (learning_rate/len(mini_batch))*new_bias for bias, new_bias in zip(self.biases, bias_gradient)]
        self.weights = [weight - (learning_rate/len(mini_batch))*new_weight for weight, new_weight in zip(self.weights, weight_gradient)]

    def validation(self, val_data):
        """Return the culmulative cost derivative value for each input and target
        in the validation data set."""
        test_results = [(self.feedForward(x)[0][0], y) for (x, y) in val_data]
        total = 0
        for (x, y) in test_results:
            total += cost_derivative(x, y)
        return total

def cost_derivative(a, y):
    """Return a partial derivative vector."""
    return (a-y)
    
def rmse(bot_output, the_targets):
    """Returns the RMSE value for a set of NN-generated outputs and the real
    data set targets."""
    bot_output = np.asarray(bot_output, dtype=float)
    the_targets = np.asarray(the_targets, dtype=float)
    return np.sqrt(((bot_output - the_targets) ** 2).mean())

def sigmoid(x):
    """Sigmoid activation."""
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    """Sigmoid activation derivative."""
    return sigmoid(x)*(1 - sigmoid(x))

rmse_vals = []


def loadData():
    inputs = []
    targets = []

    with open('avg_data.csv', "r", encoding="utf-8-sig") as f: 
        raw_data = np.genfromtxt(f, dtype=float, delimiter=",")

    np.random.shuffle(raw_data)
    for line in raw_data:
        new_input = [line[0],line[1],line[2],line[3],line[4],line[5],line[6],line[7], line[8],line[9],line[10],line[11],line[12],line[13],line[14],line[15]]
        new_target = line[16]
        inputs.append(new_input)
        targets.append(new_target)

    inputs = np.asarray(inputs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    max_input = inputs.max(axis=0, keepdims=True)
    max_target = targets.max(axis=0, keepdims=True)

    inputs = 0.8*(inputs/max_input)+0.1
    targets = 0.8*(targets/max_target)+0.1
    training_raw = []
    validation_raw = []
    testing_raw = []
    training_raw = (inputs[0:876], targets[0:876])
    validation_raw = (inputs[876:1168], targets[876:1168])
    testing_raw = (inputs[1168:], targets[1168:])

    training_inputs = [np.reshape(x, (len(new_input), 1)) for x in training_raw[0]]
    training_data = zip(training_inputs, training_raw[1])
    validation_inputs = [np.reshape(x, (len(new_input), 1)) for x in validation_raw[0]]
    validation_data = zip(validation_inputs, validation_raw[1])
    test_inputs = [np.reshape(x, (len(new_input), 1)) for x in testing_raw[0]]
    test_data = zip(test_inputs, testing_raw[1])

    return (training_data, validation_data, test_data, max_input, max_target)

start = time.time()
training_data, validation_data, testing_data, max_in, max_tar = loadData()
NeuralNet = ANN([16,16,1])

NeuralNet.grad_descent(training_data, 100, 10, 0.9, 0.9, validation_data)

the_targets = []
bot_output = []
for (x,y) in testing_data:
    output = NeuralNet.feedForward(x)
    bot_output.append(output[0][0]*max_tar)
    the_targets.append(y*max_tar)

end = time.time()

plt.plot(the_targets, bot_output,'ro')
plt.xlabel("Actual Values")
plt.ylabel("NN-Predicted")
plt.title("Two Hidden Layers")
plt.show()
plt.plot(rmse_vals)
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.title("Two Hidden Layers")

plt.show()
print("Time:",end-start)