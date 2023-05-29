import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)
    
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        return self.outputs
    
    def backward(self, grad_outputs, learning_rate):
        grad_inputs = np.dot(grad_outputs, self.weights.T)
        grad_weights = np.dot(self.inputs.T, grad_outputs)
        grad_biases = np.sum(grad_outputs, axis=0)
        
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        
        return grad_inputs
