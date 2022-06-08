import numpy as np
def sigmoid(x):
    return 1/(1+ np.exp(-x))
def sigmoid_deriv(x):
    return x * (1 - x)

training_inputs = np.array([[0,0,1,1],
                            [1,1,0,1],
                            [1,0,1,1],
                            [0,1,1,0]])
training_outputs = np.array([[0,1,1,0]]).T
np.random.seed(1)

synaptic_weights = 2 * np.random.random((4, 1)) - 1

print(synaptic_weights)

for i in range(10000):
    #the greater the number of epochs of training, the more accurate the model will be
    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    error = training_outputs - outputs

    adjustments = error * sigmoid_deriv(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments)

print("synaptic weights are ")
print(synaptic_weights)

print("outputs are ")
print(outputs)

test_input = np.array([0,0,0])

