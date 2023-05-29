import numpy as np

# "radix-2 Cooley-Tukey" algorithm
def qft(x):
    N = len(x)
    omega = np.exp(2 * np.pi * 1j / N)

    if N <= 1:
        return x

    even = qft(x[::2])
    odd = qft(x[1::2])
    factor = omega ** np.arange(N // 2)

    return np.concatenate([even + factor * odd, even - factor * odd])