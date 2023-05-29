import numpy as np
from scipy.fft import fft, ifft

class Neuron:
    def __init__(self, threshold=1.0, reset_potential=0.0):
        self.threshold = threshold
        self.reset_potential = reset_potential
        self.potential = reset_potential
        self.last_spike_time = -1.0
        self.trace = 0.0

    def integrate(self, current, time):
        self.potential += current
        if self.last_spike_time >= 0.0:
            delta_t = time - self.last_spike_time
            self.trace *= np.exp(-delta_t)

    def fire(self, time):
        if self.potential >= self.threshold:
            self.potential = self.reset_potential
            self.last_spike_time = time
            self.trace += 1.0
            return True
        return False

class Synapse:
    def __init__(self, weight, delay):
        self.weight = weight
        self.delay = delay

class NeuralNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.neurons = [Neuron() for _ in range(num_neurons)]
        self.synaptic_weights = np.random.rand(num_neurons, num_neurons)
        self.synaptic_delays = np.random.uniform(0.1, 5.0, size=(num_neurons, num_neurons))
        self.synapses = [[Synapse(self.synaptic_weights[i][j], self.synaptic_delays[i][j]) for j in range(num_neurons)] for i in range(num_neurons)]

    def propagate(self, inputs, time):
        spikes = [[] for _ in range(self.num_neurons)]  # List to store spike times of each neuron
        neuron_potentials = np.array([neuron.potential for neuron in self.neurons])
        neuron_traces = np.array([neuron.trace for neuron in self.neurons])
        neuron_last_spike_times = np.array([neuron.last_spike_time for neuron in self.neurons])

        # Integrate and fire neurons
        neuron_potentials += inputs
        integrate_mask = neuron_last_spike_times >= 0.0
        delta_t = time - neuron_last_spike_times[integrate_mask]
        neuron_traces[integrate_mask] *= np.exp(-delta_t)
        neuron_potentials[integrate_mask] -= inputs[integrate_mask]

        fire_mask = neuron_potentials >= self.threshold
        spike_times = np.where(fire_mask)[0]
        self.reset_neurons(spike_times, neuron_potentials, neuron_last_spike_times, neuron_traces)

        # Propagate spikes through synapses
        pre_spike_times = []
        post_spike_times = []
        for pre_idx, spike_time in enumerate(spike_times):
            pre_spike_times.extend([(pre_idx, spike_time)])
            post_spike_times.extend([(post_idx, spike_time + self.synapses[pre_idx][post_idx].delay) for post_idx in range(self.num_neurons)])

        pre_spike_times = np.array(pre_spike_times)
        post_spike_times = np.array(post_spike_times)

        self.update_neuron_integration(post_spike_times, neuron_potentials, time, self.synaptic_weights)
        self.reset_neurons(post_spike_times[:, 0], neuron_potentials, neuron_last_spike_times, neuron_traces)

    def update_neuron_integration(self, spike_times, neuron_potentials, time, synaptic_weights):
        pre_indices = spike_times[:, 0]
        post_indices = spike_times[:, 1]
        delays = np.array([self.synapses[pre][post].delay for pre, post in zip(pre_indices, post_indices)])
        weights = synaptic_weights[pre_indices, post_indices]
        delta_t = time - delays
        neuron_potentials += weights * self.delayed_trace(delta_t)
    
    def delayed_trace(self, t):
        return np.exp(-0.5 * ((t - self.delay) / self.sigma) ** 2)

    def reset_neurons(self, spike_times, neuron_potentials, neuron_last_spike_times, neuron_traces):
        neuron_potentials[spike_times] = self.reset_potential
        neuron_last_spike_times[spike_times] = time
        neuron_traces[spike_times] += 1.0

    def update_weights(self, pre_idx, post_idx, delta):
        self.synaptic_weights[pre_idx][post_idx] += delta

class STDPNeuralNetwork(NeuralNetwork):
    def __init__(self, num_neurons, learning_rate=0.001, trace_decay=0.99, spike_window=10.0):
        super().__init__(num_neurons)
        self.learning_rate = learning_rate
        self.trace_decay = trace_decay
        self.spike_window = spike_window
        self.traces = np.zeros((num_neurons, num_neurons))

    def propagate(self, inputs, time):
        for i, current in enumerate(inputs):
            self.neurons[i].integrate(current, time)
            if self.neurons[i].fire(time):
                spike_times = np.array([(i, time)])
                self.update_neuron_integration(spike_times, self.neurons, time, self.synaptic_weights)
                self.update_neurons(spike_times, self.neurons)
        # print(network.synaptic_weights)

    def update_neurons(self, spike_times, neurons):
        pre_indices = spike_times[:, 0]
        post_indices = spike_times[:, 1]
        delta = self.learning_rate * (1.0 - (post_time - pre_time) / self.spike_window)
        traces = self.traces[pre_indices, post_indices]
        synaptic_weights = self.synaptic_weights[pre_indices, post_indices]
        traces *= self.trace_decay
        traces += 1.0
        synaptic_weights += delta * traces
    
    def prune_weak_connections(self, threshold):
        self.synaptic_weights[self.synaptic_weights < threshold] = 0.0

    def apply_stdp_learning_rule(self, pre_spike_times, post_spike_times):
        pre_indices = pre_spike_times[:, 0]
        post_indices = post_spike_times[:, 0]
        pre_times = pre_spike_times[:, 1]
        post_times = post_spike_times[:, 1]

        mask = pre_times <= post_times
        delta = self.learning_rate * (1.0 - (post_times - pre_times) / self.spike_window)
        delta *= mask

        self.update_neurons(pre_indices, post_indices, delta)

class GaussianSynapse(Synapse):
    def __init__(self, weight, delay, sigma):
        super().__init__(weight, delay)
        self.sigma = sigma

    def delayed_trace(self, t):
        return np.exp(-0.5 * ((t - self.delay) / self.sigma) ** 2)

class QFTNeuralNetwork(NeuralNetwork):
    def __init__(self, num_neurons, learning_rate=0.001, trace_decay=0.99, spike_window=10.0):
        super().__init__(num_neurons)
        self.learning_rate = learning_rate
        self.trace_decay = trace_decay
        self.spike_window = spike_window
        self.traces = np.zeros((num_neurons, num_neurons))

    def propagate(self, inputs, time):
        for i, current in enumerate(inputs):
            self.neurons[i].integrate(current, time)
            if self.neurons[i].fire(time):
                spike_times = np.array([(i, time)])
                self.update_neuron_integration(spike_times, self.neurons, time, self.synaptic_weights)
                self.update_neurons(spike_times, self.neurons)

    def update_neurons(self, spike_times, neurons, time):
        pre_indices = spike_times[:, 0]
        post_indices = spike_times[:, 1]
        pre_times = np.full_like(pre_indices, time)
        post_times = np.full_like(post_indices, time)

        delta = self.learning_rate * (1.0 - (post_times - pre_times) / self.spike_window)
        traces = self.traces[pre_indices, post_indices]
        synaptic_weights = self.synaptic_weights[pre_indices, post_indices]
        traces *= self.trace_decay
        traces += 1.0
        synaptic_weights += delta * traces

    def fourier_transform(self):
        synaptic_weights_fft = fft(self.synaptic_weights)
        self.synaptic_weights = ifft(synaptic_weights_fft * self.learning_rate).real
        self.synaptic_weights = np.abs(self.synaptic_weights)

    def apply_qft_learning_rule(self, pre_spike_times, post_spike_times):
        pre_indices = pre_spike_times[:, 0]
        post_indices = post_spike_times[:, 0]
        pre_times = pre_spike_times[:, 1]
        post_times = post_spike_times[:, 1]

        mask = pre_times <= post_times
        delta = self.learning_rate * (1.0 - (post_times - pre_times) / self.spike_window)
        delta *= mask

        self.update_neurons(pre_indices, post_indices, delta)

# Example usage
network = STDPNeuralNetwork(num_neurons=10000, learning_rate=0.011, trace_decay=0.314, spike_window=9.67)
inputs = np.random.uniform(0.0, 1.0, size=network.num_neurons)
time = 0.0
network.propagate(inputs, time)
network.prune_weak_connections(0.01)
network.apply_stdp_learning_rule(pre_spike_times, post_spike_times)
network.fourier_transform()
network.apply_qft_learning_rule(pre_spike_times, post_spike_times)
