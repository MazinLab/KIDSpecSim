import numpy as np
import astropy.units as u
from scipy.constants import h, c


def gauss(x, mu, sig, A):
    """
    :param x: wavelength
    :param mu: mean
    :param sig: sigma
    :param A: amplitude
    :return: value of the Gaussian
    """
    return (A * np.exp(- (x - mu) ** 2. / (2. * sig ** 2.))).T


def gauss_intersect(mu, sig, A):
    """
    :param mu: 2 means of Gaussians
    :param sig: 2 sigmas of Gaussians
    :param A: 2 amplitudes of Gaussians
    :return: analytic calculation of the intersection point between 2 1D Gaussian functions
    """

    if sig.any() == 0 or A.any() == 0:
        raise ValueError("The sigmas and amplitudes must all be non-zero to find the Gaussian intersection.")

    assert mu.size == sig.size and mu.size == A.size, "mu, sig, and A must all be the same size."

    a = 1 / sig[0] ** 2 - 1 / sig[1] ** 2
    b = 2 * mu[1] / sig[1] ** 2 - 2 * mu[0] / sig[0] ** 2
    c = (mu[0] / sig[0]) ** 2 - (mu[1] / sig[1]) ** 2 - 2 * np.log(A[0] / A[1])
    if np.abs(a) <= 1e-10:
        return quad_formula(a=0, b=b, c=c)[0]
    else:
        solp, soln = tuple(quad_formula(a=a, b=b, c=c))
        if mu[0] < solp < mu[1]:
            return solp
        elif mu[0] < soln < mu[1]:
            return soln
        else:
            raise ValueError("The Gaussian intersection could not be found.")


def quad_formula(a, b, c):
    """
    :return: quadratic formula results for ax^2 + bx + c = 0 or linear bx + c = 0
    """
    if a == 0:
        return [-c / b]
    else:
        return [((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)), ((-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))]


def nearest_idx(array, value):
    """
    :param array: the array to probe
    :param value: the desired value
    :return: the index of the closest array value to the desired value
    """
    return (np.abs(array - value)).argmin()


def sig_to_R(sig, lam):
    """
    :param sig: sigma in the same units as lam
    :param lam: wavelength, energy, or phase
    :return: the spectral resolution
    """
    dlam = sig * 2 * np.sqrt(2 * np.log(2))
    R = lam / dlam
    return np.abs(R)


def wave_to_eV(wave):
    """
    :param wave: wavelength in astropy units
    :return: energy in eV
    """
    return ((h*u.J*u.s) * (c * u.m / u.s) / wave).to(u.eV)


def energy_to_nm(eV):
    """
    :param eV: energy in astropy units
    :return: wavelength in nm
    """
    return ((h*u.J*u.s) * (c * u.m / u.s) / eV).to(u.nm)


def n_bins(n_data: int, method: str = 'rice'):
    # TODO include more algorithms based on number of data points
    """
    :param int n_data: number of data points
    :param str method: # of bin method to use, only Rice available currently
    :return: best number of bins based on various algorithms
    """
    if method == 'rice':
        return int(4 * n_data ** (1 / 3))  # multiplied by 2 because sparsest pixel likely only has 2 gauss (need 4)
    else:
        raise ValueError(f'Method {method} not supported.')
