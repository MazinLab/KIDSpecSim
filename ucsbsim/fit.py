import numpy as np

"""
Contains helper functions for Gaussian parameter fits.
"""

def gauss(x, mu, sig, A):
    """
    :param x: wavelength
    :param mu: mean
    :param sig: sigma
    :param A: amplitude
    :return: value of the Gaussian with those parameters at the wavelength
    """
    return (A * np.exp(- (x - mu) ** 2. / (2. * sig ** 2.))).T


def gauss_summed(x, *args):
    """
    :param x: wavelength range
    :param args: mu, sig, and A for each order Gaussian, as (mu_1, mu_2, ..., sig_1, sig_2, ..., A_1, A_2, ...)
    :return: sum of all Gaussian functions on the order axis
    """
    n = len(args)
    if n % 3 != 0:
        raise ValueError("You must include a mean, sigma, and A for each Gaussian.")
    mu = np.array(args[:int(n/3)])
    sig = np.array(args[int(n/3):int(2*n/3)])
    A = np.array(args[int(2*n/3):])
    return np.sum(gauss(x[:, None], mu, sig, A), axis=0)

def quad_formula(a, b, c):
    """
    :return: positive quadratic formula result for ax^2 + bx + c = 0
    """
    return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

def gauss_intersect(mu, sig, A):
    """
    :param mu: means of Gaussians in order
    :param sig: sigmas of Gaussians in order
    :param A: amplitudes of Gaussians in order
    :return: analytic calculation of the intersection point between adjacent 1D Gaussian functions
    """
    n = len(mu)
    if n != len(sig) or n != len(A):
        raise ValueError("mu, sig, and A must all be the same size.")
    a = [1 / sig[i] ** 2 - 1 / sig[i+1] ** 2 for i in range(n-1)]
    b = [2 * mu[i+1] / sig[i+1] ** 2 - 2 * mu[i] / sig[i] ** 2 for i in range(n-1)]
    c = [(mu[i] / sig[i]) ** 2 - (mu[i+1] / sig[i+1]) ** 2 - 2 * np.log(A[i] / A[i+1]) for i in range(n-1)]
    return quad_formula(np.array(a), np.array(b), np.array(c))


if __name__ == '__main__':
    # examples showing results of each function in this module
    import matplotlib.pyplot as plt
    wave = np.linspace(300,1000,10000)
    mu = np.arange(400,900,100)
    sig = np.arange(10,60,10)
    A = np.arange(100,600,100)
    params = np.concatenate((mu,sig,A))
    gausses = gauss(wave[:, None], mu, sig, A)
    summed = gauss_summed(wave,*params)
    intersection = gauss_intersect(mu, sig, A)
    plt.grid()
    plt.plot(wave, summed, color='C0', label='Sum of all')
    for i in range(len(mu)):
        plt.plot(wave,gausses[i,:],color=f'C{i+1}',label=f"mu:{mu[i]:.0f}, sig:{sig[i]:.0f}, A:{A[i]:.0f}")
    for i in intersection:
        plt.axvline(i,color='k',linestyle='--')
    plt.title("Example of Gaussian Summing and Intersections")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
