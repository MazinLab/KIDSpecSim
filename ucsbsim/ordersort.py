import numpy as np
import scipy.interpolate as interp
from scipy import optimize
import scipy
import scipy.signal as sig
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import time
import astropy.units as u
from mkidpipeline import photontable as pt
import statistics
import scipy.stats as stats
import math
import random

"""
Order-sorts photon table and turns it into a final spectrum.

*CHOOSE TO SHOW PLOTS (DEFAULT) OR NOT.
"""
plot = True


# counts-per-pixel border for low and high regime
threshold = 200

# change R0 if needed
R0 = 15


# loglike_mu returns the approximate means of the Gaussians
def loglike_mu(x, data, sig, norm):
    data = np.array(data)
    mu = x
    likes = np.empty([5, len(data)])
    var = sig ** 2
    for i in range(5):
        likes[i, :] = norm[i] * np.exp(-0.5 * (data - mu[i]) ** 2 / var[i]) / np.sqrt(2 * np.pi * var[i])
    likes_tot = np.sum(likes, axis=0)
    return -np.sum(np.log(likes_tot))


# loglike_sig returns the approximate standard deviations of the Gaussians
def loglike_sig(x, data, opt_mu, norm):
    data = np.array(data)
    sig = x
    likes = np.empty([5, len(data)])
    var = sig ** 2
    for i in range(5):
        likes[i, :] = norm[i] * np.exp(-0.5 * (data - opt_mu[i]) ** 2 / var[i]) / np.sqrt(2 * np.pi * var[i])
    likes_tot = np.sum(likes, axis=0)
    return -np.sum(np.log(likes_tot))


# standard form of a Gaussian with some amplitude
def gauss(x, mu, sig, A):
    return A * np.exp(- (x - mu) ** 2 / (2 * sig ** 2))


# gauss5 for use in curve_fit where all five 5 amplitudes need to be distinct variables
def gauss5(x, A0, A1, A2, A3, A4):
    a = A0 * np.exp(- (x - opt_mu[0]) ** 2 / (2 * opt_sig[0] ** 2))
    b = A1 * np.exp(- (x - opt_mu[1]) ** 2 / (2 * opt_sig[1] ** 2))
    c = A2 * np.exp(- (x - opt_mu[2]) ** 2 / (2 * opt_sig[2] ** 2))
    d = A3 * np.exp(- (x - opt_mu[3]) ** 2 / (2 * opt_sig[3] ** 2))
    e = A4 * np.exp(- (x - opt_mu[4]) ** 2 / (2 * opt_sig[4] ** 2))
    return a + b + c + d + e


# opening the 5 x 2048 central wavelengths and FWHMs, these become initial guesses for Gaussian parameter fits
with open(f'lambda_pixel.csv') as f:
    lambda_pixel = np.loadtxt(f, delimiter=",")
with open(f'dl_mkid.csv') as f:
    dl_mkid_pixel = np.loadtxt(f, delimiter=",") / 2.355

# opening the fractional order sorting file, these determine which wavelengths belong in which order
with open(f'fraccal_wave.csv') as f:
    wave_frac = np.loadtxt(f, delimiter=",")
frac = np.zeros([10000,5,2048])
for i in range(5):
    with open(f'fraccal{i}_R0of{R0}.csv') as f:
        frac[:,i,:] = np.loadtxt(f, delimiter=",")

# opening the h5 file containing the photon table
file = pt.Photontable(f'spec_phoenix_200ppp.h5')
waves = file.query(column='wavelength')
resID = file.query(column='resID')

# separating photons in each pixel by resID
by_pixel = []
resid_map = np.arange(2048, dtype=int) * 10 + 100
for i in range(2048):
    idx = np.where(resID == resid_map[i])
    by_pixel.append(waves[idx].tolist())

# large initial guesses for amplitude since normalized Gaussian peak is very small
A = np.array([0.1, 3 / 16, 5 / 8, 1.1, 1.9]) * 1e2

# beginning the loop through all pixels with separate behavior for low count & high count regime
tot_phot = np.zeros([5, 2048])
for j in range(2048):
    #if len(by_pixel[j]) < threshold:  # if less than threshold photons, use fake cal data
    count, edges = np.histogram(by_pixel[j], bins=wave_frac, density=False)
    tot_phot[:, j] = np.round(np.sum(count[:,None]*frac[1:,:,j], axis=0))
    # TODO following code WIP needs debugging
    """else:
        ordered = np.sort(by_pixel[i])  # if more than threshold photons, split using GMM
        n = len(ordered)  # obtaining bins
        q1 = statistics.median(ordered[:int(n / 2)])
        q3 = statistics.median(ordered[int(n / 2):])
        iqr = q3 - q1
        h = 2 * iqr * n ** (-1 / 3)
        bin_i = int((max(ordered) - min(ordered)) * 3 / h)
        opt_mu = optimize.minimize(loglike_mu, lambda_pixel[:, j], (by_pixel[j], dl_mkid_pixel[:, j], A),
                                   method="Powell").x  # optimized means
        opt_sig = optimize.minimize(loglike_sig, dl_mkid_pixel[:, j], (by_pixel[j], opt_mu, A), method="Powell").x 
        # optimized sigmas
        bin_heights, bin_borders, _ = plt.hist(by_pixel[j], bins=bin_i)
        bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
        opt_A, _ = optimize.curve_fit(gauss5, bin_centers, bin_heights, p0=A)  # optimized amplitudes
        newwave_binheights, _, _ = plt.hist(by_pixel[j], bins=wave_frac)

        # interpolating gaussians to produce new fractions
        hcount_interp = np.zeros([10000, 5])
        for i in range(5):
            hcount_interp[:, i] = interp.interp1d(bin_centers, gauss(bin_centers, opt_mu[i], opt_sig[i], opt_A[i]),
                                                     fill_value=0, bounds_error=False, copy=False)(wave_frac)
        sumd = np.sum(hcount_interp, axis=1)
        hcount_frac = hcount_interp / sumd[:, None]
        np.nan_to_num(hcount_frac, copy=False, nan=0)
        for i in range(5):
            tot_phot[i, j] = np.sum(newwave_binheights * hcount_frac[1:,i])"""

# import convolved blaze angles to divide out
with open(f'convolved_blaze.csv') as f:
    blaze = np.loadtxt(f, delimiter=",")

final_interp = np.empty([5, 10000])  # interpolating photon counts to sum to complete spectrum
wave = np.linspace(350, 800, 10000)
for i in range(5):
    final_interp[i, :] = interp.interp1d(lambda_pixel[i, :], tot_phot[i, :],
                                             fill_value=0, bounds_error=False, copy=False)(wave)
final_sumd = np.sum(final_interp, axis=0)
result_sumd = np.sum(result_interp, axis=0)

blaze_interp = np.empty([5, 10000])  # interpolating blaze efficiencies to divide out of spectrum
for i in range(5):
    blaze_interp[i] = interp.interp1d(lambda_pixel[i,:], blaze[i,:],
                                      fill_value=0, bounds_error=False, copy=False)(wave)
blaze_sumd = np.sum(blaze_interp, axis=0)
final_sumd /= blaze_sumd  # dividing out blaze
result_sumd /= blaze_sumd

if plot:
    plt.grid()
    for i in range(5):
        plt.plot(lambda_pixel[i, :], tot_phot[i, :], '.', label=f"Order {i+5}")
    plt.title("Photon Table to Spectrum (Low Count Regime)")
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Photon Count')
    plt.show()
    plt.grid()
    plt.plot(wave, final_sumd, label="Output")
    plt.title("Photon Table to Spectrum (Divided Out Blaze)")
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Photon Count')
    plt.show()
