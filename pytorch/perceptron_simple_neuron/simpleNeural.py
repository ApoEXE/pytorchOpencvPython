import numpy as np

# sigmoid function to normalize inputs
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return x * (1 - x)

# input dataset
training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

# output dataset
training_outputs = np.array([[0,1,1,0]]).T

# seed random numbers to make calculation
np.random.seed(1)

# initialize weights randomly with mean 0 to create weight matrix, synaptic weights
synaptic_weights = 2 * np.random.random((3,1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)


# Iterate 10,000 times
for iteration in range(10000):

    # Define input layer
    input_layer = training_inputs
    # Normalize the product of the input layer with the synaptic weights
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # how much did we miss?
    error = training_outputs - outputs

    # multiply how much we missed by the
    # slope of the sigmoid at the values in outputs
    adjustments = error * sigmoid_derivative(outputs)

    # update weights
    synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic weights after training: ')
print(synaptic_weights)

print("Output After Training:")
print(outputs)

print("NEW Weights______________")

training_testing = np.array([[1,0,1],
                            [1,0,1],
                            [0,0,1],
                            [0,1,1]])
new_w=synaptic_weights
for iteration in range(1):

    # Define input layer
    input_layer = training_testing
    # Normalize the product of the input layer with the synaptic weights
    output_test = sigmoid(np.dot(input_layer, new_w))
print("output test:",output_test)
#ERROR WEIGHTED DERIVATE this is for training where we compute better weights
#Error = output - actual output
#Error_weighted_Derivate = error*input*d/dt sigma(output)dt