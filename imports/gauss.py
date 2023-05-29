import numpy as np

def gauss_net_activation(x, W, b, sigma_b, sigma_W, din):
    z = np.dot(W, x) + b
    sigma_squared = sigma_b**2 + sigma_W**2 * np.exp(2 * (sigma_b**2 + sigma_W**2 * np.dot(x, x)))
    activation = np.exp(z) / np.sqrt(sigma_squared)
    return activation

def KGauss_kernel(x1, x2, W, b, sigma_b, sigma_W, din):
    diff = x1 - x2
    distance_squared = np.dot(diff, diff)
    kernel = sigma_b**2 + sigma_W**2 * np.exp(-sigma_W**2 * distance_squared / (2 * din))
    return kernel
