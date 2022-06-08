import numpy as np

class NeuralN():
    def __init__(self):
        np.random.seed(1)
        self.weight = 2 * np.random.random((5, 1)) - 1
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    def tanh_deriv(self, x):
        return 1 - (x ** 2)
    def train(self, training_inputs, training_outputs, epoch):

        for i in range(epoch):
            output = self.test(training_inputs)
            error = training_outputs - output
            adjustment = np.dot(training_inputs.T, error * self.tanh_deriv(output))
            self.weight += adjustment
    def test(self, inputs):
        inputs = inputs.astype(float)
        return self.tanh(np.dot(inputs, self.weight))
new = NeuralN()
print("weights are ")
print(new.weight)

training_input = np.array([[0,0,1,1,1],
                         [0,1,0,1,0],
                         [0,0,1,0,0],
                         [0,1,1,1,1],
                         [1,1,1,1,1],
                         [0,0,0,0,0]])
training_output = np.array([[1,0,0,1,1,0]]).T

new.train(training_input, training_output, 400000)

print("new weights are: ")
print(new.weight)

A = input("input 1 is: ")
B = input("input 2 is: ")
C = input("input 3 is: ")
D = input("input 4 is: ")
E = input("input 5 is: ")

print("result is: ")
print(new.test(np.array([A,B,C,D,E])))






