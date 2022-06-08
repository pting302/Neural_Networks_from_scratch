import numpy as np
class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((3,1)) - 1
        #generates 3 random weights between -1 and 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def sigmoid_deriv(self, x):
        return x * (1-x)
    def train(self, training_inputs, training_outputs, epoch):

        for i in range(epoch):
            output = self.test(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_deriv(output))
            self.synaptic_weights += adjustments
    def test(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output
nn = NeuralNetwork()

print("synaptic weights are ")
print(nn.synaptic_weights)
training_inputs = np.array([[0,0,1],
                        [1,1,1],
                        [1,0,1],
                        [0,1,1],
                        [0,0,0]])
training_outputs = np.array([[0,1,1,0,1]]).T
nn.train(training_inputs, training_outputs, 200000)

print("new synaptic weights: ")
print(nn.synaptic_weights)

A = str(input("input 1: "))
B = str(input("input 2: "))
C = str(input("input 3: "))

print("new data = ", A, B, C)
print("output is ")
print(nn.test(np.array([A, B, C])))