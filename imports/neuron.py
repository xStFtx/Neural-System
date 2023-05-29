import math
import random

class Neuron:
    def __init__(self):
        self.weights = []
        self.bias = 0.0
    
    def initialize(self, num_inputs):
        # Initialize weights with small random values
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]
        # Initialize bias with a random value
        self.bias = random.uniform(-0.5, 0.5)
    
    def activate(self, inputs):
        # Calculate the weighted sum of inputs
        weighted_sum = sum(x * w for x, w in zip(inputs, self.weights))
        # Add bias to the weighted sum
        weighted_sum += self.bias
        # Apply activation function
        activation = self.sigmoid(weighted_sum)
        return activation
    
    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + math.exp(-x))
    
    def relu(self, x):
        # Rectified Linear Unit (ReLU) activation function
        return max(0, x)
    
    def leaky_relu(self, x, alpha=0.01):
        # Leaky ReLU activation function
        return max(alpha * x, x)
    
    def tanh(self, x):
        # Hyperbolic tangent (tanh) activation function
        return math.tanh(x)
    
    def softmax(self, inputs):
        # Softmax activation function
        exp_values = [math.exp(x) for x in inputs]
        sum_exp = sum(exp_values)
        softmax_values = [x / sum_exp for x in exp_values]
        return softmax_values
    
    def initialize_random_uniform(self, num_inputs, lower_bound, upper_bound):
        # Initialize weights with random values from a uniform distribution
        self.weights = [random.uniform(lower_bound, upper_bound) for _ in range(num_inputs)]
        # Initialize bias with a random value from a uniform distribution
        self.bias = random.uniform(lower_bound, upper_bound)
    
    def initialize_random_normal(self, num_inputs, mean, std_dev):
        # Initialize weights with random values from a normal distribution
        self.weights = [random.normalvariate(mean, std_dev) for _ in range(num_inputs)]
        # Initialize bias with a random value from a normal distribution
        self.bias = random.normalvariate(mean, std_dev)
