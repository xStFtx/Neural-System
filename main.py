import numpy as np
from imports.gauss import GaussNet
from imports.SNN import SpikingNeuralNetwork

def main():
    # Define the parameters for GaussNet
    W = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # Weight matrix
    b = np.array([0.7, 0.8])  # Bias vector
    sigma_b = 0.5  # Sigma b
    sigma_W = 0.3  # Sigma W
    din = 2  # Value of din

    # Create an instance of GaussNet
    gauss_net = GaussNet(W, b, sigma_b, sigma_W, din)

    # Example usage
    x = np.array([1, 2, 3])  # Input vector

    # Calculate the activation using the GaussNet instance
    activation = gauss_net.gauss_net_activation(x)
    print("Activation:", activation)

    # Calculate the kernel using the GaussNet instance
    x1 = np.array([1, 2, 3])  # Input vector 1
    x2 = np.array([4, 5, 6])  # Input vector 2
    kernel = gauss_net.KGauss_kernel(x1, x2)
    print("Kernel:", kernel)

    # Create a spiking neural network
    num_neurons = 5
    snn = SpikingNeuralNetwork(num_neurons)

    quantum_state = np.array([0.2, 0.7, 0.3, 0.9, 0.5])
    snn.simulate(quantum_state)


if __name__ == "__main__":
    main()
