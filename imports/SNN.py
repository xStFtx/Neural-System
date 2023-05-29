import numpy as np

class SpikingNeuron:
    def __init__(self, threshold=1.0, reset_potential=0.0, membrane_leakage=0.1):
        self.threshold = threshold
        self.reset_potential = reset_potential
        self.membrane_leakage = membrane_leakage
        self.potential = reset_potential
    
    def integrate(self, current):
        self.potential += current - self.membrane_leakage
        if self.potential >= self.threshold:
            self.fire()
    
    def fire(self):
        print("Neuron fired!")
        self.potential = self.reset_potential


class SpikingNeuralNetwork:
    def __init__(self, num_neurons):
        self.neurons = [SpikingNeuron() for _ in range(num_neurons)]
    
    def simulate(self, inputs):
        for neuron, current in zip(self.neurons, inputs):
            neuron.integrate(current)
