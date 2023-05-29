import numpy as np

class GaussNet:
    def __init__(self, W, b, sigma_b, sigma_W, din):
        self.W = W
        self.b = b
        self.sigma_b = sigma_b
        self.sigma_W = sigma_W
        self.din = din

    def gauss_net_activation(self, x):
        z = np.dot(self.W, x) + self.b
        sigma_squared = self.sigma_b**2 + self.sigma_W**2 * np.exp(2 * (self.sigma_b**2 + self.sigma_W**2 * np.dot(x, x)))
        activation = np.exp(z) / np.sqrt(sigma_squared)
        return activation

    def KGauss_kernel(self, x1, x2):
        diff = x1 - x2
        distance_squared = np.dot(diff, diff)
        kernel = self.sigma_b**2 + self.sigma_W**2 * np.exp(-self.sigma_W**2 * distance_squared / (2 * self.din))
        return kernel
