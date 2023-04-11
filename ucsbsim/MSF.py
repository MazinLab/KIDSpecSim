import numpy as np
import scipy.interpolate as interp
from scipy import optimize
import scipy
import matplotlib.pyplot as plt
from mkidpipeline import photontable as pt

"""
Divides out blazed blackbody spectrum from calibration data without running through detector.py.
Calls file from pre-generated fake calibration spectrum source and photon table.

***BEFORE RUNNING THIS MSF.PY SCRIPT***

*CHOOSE IF YOU WANT A FINAL PLOT (DEFAULT) OR NOT.
*CHOOSE TO ENABLE CLICKTHROUGH FOR CHECKING FIT AT EACH PIXEL (PLOTS SAVED TO /cal_plots) OR NOT (DEFAULT).
*ENSURE PIXEL_LIM IS EQUAL TO GENERATED CALIBRATION SPECTRUM FROM MAIN.PY.
"""
plot = True
clickthrough = False
pixel_lim = 50000


# standard form of a Gaussian with some amplitude
def gauss(x, mu, sig, A):
    return A * np.exp(- (x - mu) ** 2 / (2 * sig ** 2))


# loglike_mu returns the approximate means of the Gaussians
def loglike_mu(x, data, sig, A, n):
    likes = np.zeros([n, len(data)])
    for i in range(n):
        likes[i, :] = gauss(data, x[i], sig[i], A[i]) / np.sqrt(2 * np.pi * sig[i] ** 2)
    likes_tot = np.sum(likes, axis=0)
    return -np.sum(np.log(likes_tot))


# gauss_4added for use in curve_fit where 4 amplitudes/sigmas need to be distinct variables, (no order 9 flux)
def gauss_4added(x, A, B, C, D, W, X, Y, Z):
    amp = [A, B, C, D]
    sig = [W, X, Y, Z]
    gausses = np.array([gauss(x, opt_mu[i], sig[i], amp[i]) for i in range(len(amp))])
    return np.sum(gausses, axis=0)


# gauss_5added for use in curve_fit where all five 5 amplitudes/sigmas need to be distinct variables
def gauss_5added(x, A, B, C, D, E, V, W, X, Y, Z):
    amp = [A, B, C, D, E]
    sig = [V, W, X, Y, Z]
    gausses = np.array([gauss(x, opt_mu[i], sig[i], amp[i]) for i in range(len(amp))])
    return np.sum(gausses, axis=0)


# finds intersection of resulting gaussians for binning
def gauss_intersect(x, mu, sig, A):
    a = 1 / sig[0] ** 2 - 1 / sig[1] ** 2
    b = 2 * mu[1] / sig[1] ** 2 - 2 * mu[0] / sig[0] ** 2
    c = (mu[0] / sig[0]) ** 2 - (mu[1] / sig[1]) ** 2 - 2 * np.log(A[0] / A[1])
    return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)  # solves for intersection using quadratic formula


# load blazed/blackbody calibration spectrum before detector.py simulation
# TODO update blaze with analytic convolution
with open(f'blaze_sumd.csv') as f:
    blaze_sumd = np.loadtxt(f, delimiter=",")
with open(f'lambda_pixel.csv') as f:
    lambda_pixel = np.loadtxt(f, delimiter=",")
new_x = np.linspace(100, 1000, 10000)
blaze_interp = []
for i in range(4):
    blaze_interp.append(interp.interp1d(lambda_pixel[::-1, :][i, :], blaze_sumd[::-1, :][i, :],
                                        fill_value='extrapolate', bounds_error=False, copy=False))
blaze_interp.append(interp.interp1d(lambda_pixel[::-1, :][4, :][blaze_sumd[::-1, :][4, :] > 0.1],
                                    blaze_sumd[::-1, :][4, :][blaze_sumd[::-1, :][4, :] > 0.1],
                                    fill_value='extrapolate', bounds_error=False, copy=False))

# opening the FWHMs, these become initial guesses for Gaussian parameter fits
with open(f'dl_mkid.csv') as f:
    dl_mkid_pixel = np.loadtxt(f, delimiter=",") / 2.355  # turning FWHM to sigma

lambda_pixel = lambda_pixel[::-1, :]  # flip orders (to be in ascending wavelength order)
dl_mkid_pixel = dl_mkid_pixel[::-1, :]

# opening the h5 file containing the photon table
file = pt.Photontable(f'spec_blackbody_{pixel_lim}ppp.h5')
waves = file.query(column='wavelength')
resID = file.query(column='resID')

colors = ['red', 'orange', 'green', 'blue', 'purple']
cov = np.zeros([5, 5, 2048])
photon_bins = np.zeros([6, 2048])
photon_bins[-1, :] = 1000
spec = np.zeros([5, 2048])
resid_map = np.arange(2048, dtype=int) * 10 + 100
p = 0
fig, axes = plt.subplots(5, 2, figsize=(8.5, 14))
ax = [sub for x in axes for sub in x]

for j in range(2048):  # sorting by resID and fitting GMM to each pixel
    n = 4 if lambda_pixel[0, j] < 400 else 5  # pixels where order 9 is below 400 nm will only have 4 functions
    idx = np.where(resID == resid_map[j])  # splitting photon table by resID
    photons_j = waves[idx].tolist()
    bins = int(6 * len(photons_j) ** (1 / 3) - 10)
    counts, edges_j = np.histogram(photons_j, bins=bins)  # binning photons
    counts_j = np.array([float(x) for x in counts])
    centers_j = edges_j[:-1] + np.diff(edges_j) / 2

    A = list(np.array([np.max(counts_j) for i in range(n)])) + \
        list(np.array([dl_mkid_pixel[5 - n + i, j] for i in range(n)]))  # guess amplitudes/guess sigmas
    opt_mu = optimize.minimize(loglike_mu, lambda_pixel[5 - n:, j],  # optimizing for means of each Gaussians
                               (photons_j, dl_mkid_pixel[5 - n:, j], A, n), method="Powell").x
    func = gauss_5added if n == 5 else gauss_4added
    opt_A, _ = optimize.curve_fit(func, centers_j, counts_j, p0=A)  # fitting amplitudes and sigmas
    opt_sig = opt_A[int(len(opt_A) / 2):]  # splitting array to sigmas
    opt_A = opt_A[:int(len(opt_A) / 2)]  # splitting array to amplitudes
    bb_blaze = np.array([blaze_interp[5 - n + i](opt_mu[i]) for i in range(n)])  # divide out blazed blackbody shape
    unblaze_A = opt_A / bb_blaze  # divide out bb/blaze to get flat field spectrum for fitted Gaussians

    photon_bins[6 - n:5, j] = np.array(
        [gauss_intersect(new_x, [opt_mu[i], opt_mu[i + 1]], [opt_sig[i], opt_sig[i + 1]],
                         [unblaze_A[i], unblaze_A[i + 1]]) for i in range(n - 1)])
    for m, o in enumerate(photon_bins[:, j]):
        photon_bins[m, j] = m if o == 0 else photon_bins[m, j]  # ensure bins are monotonically increasing if many 0s

    # cov: calculate % of photons that will fall into other orders from any given order (9 -> 5 for index 0 -> 4)
    for i in range(n):
        sum_i = np.sum(gauss(np.arange(new_x[0], new_x[-1], .01), opt_mu[i], opt_sig[i], unblaze_A[i]))
        for k in range(n):
            cov[5 - n + i, 5 - n + k, j] = np.sum(
                gauss(np.arange(photon_bins[5 - n + k, j], photon_bins[5 - n + k + 1, j], 0.01),
                      opt_mu[i], opt_sig[i], unblaze_A[i])) / sum_i

    hist, edges = np.histogram(photons_j, bins=photon_bins[5 - n:, j])  # binning photons again by photon_bins edges
    spec[5 - n:, j] = hist/bb_blaze  # each value is total photon count for order/pixel

    if clickthrough:  # this will generate 2048 plots in 10 subplot increments to check for parameter fits
        if j % 10 == 0 and j != 0:
            p = 0
            fig, axes = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(8.5, 14))
            ax = [sub for x in axes for sub in x]
        ax[p].grid()
        ax[p].set_title(f"Pixel {j + 1}")
        ax[p].set_xlim([350, 900])
        ax[p].hist(photons_j, bins=bins, color='k', alpha=0.5)
        for i in range(n):
            ax[p].plot(new_x, gauss(new_x, opt_mu[i], opt_sig[i], opt_A[i]), color=colors[5 - n + i])
        for i in range(n):
            ax[p].axvline(photon_bins[i + 1, j], color='r', linestyle='--', linewidth=1)
        ax[p].axvline(photon_bins[0, j], color='r', linestyle='--', linewidth=1, label='Bin Edges')
        ax[p].legend()
        if p == 9:
            ax[8].set_xlabel("Wavelength (nm)")
            ax[9].set_xlabel("Wavelength (nm)")
            ax[0].set_ylabel("Photon Count")
            ax[2].set_ylabel("Photon Count")
            ax[4].set_ylabel("Photon Count")
            ax[6].set_ylabel("Photon Count")
            ax[8].set_ylabel("Photon Count")
            plt.tight_layout()
            plt.savefig(f'cal_plots/cal_{j-9}to{j+1}.png')
            plt.close('all')
        else:
            plt.close('all')
        if j == 2047:
            ax[6].set_xlabel("Wavelength (nm)")
            ax[7].set_xlabel("Wavelength (nm)")
            ax[0].set_ylabel("Photon Count")
            ax[2].set_ylabel("Photon Count")
            ax[4].set_ylabel("Photon Count")
            ax[6].set_ylabel("Photon Count")
            ax[8].remove()
            ax[9].remove()
            plt.tight_layout()
            plt.savefig(f'cal_plots/cal_2040to2048.png')
            plt.close('all')
        else:
            plt.close('all')
        p += 1

np.savetxt('cal_bins.csv', photon_bins, delimiter=',')  # saving calibration bin edges to file for use in other spectra
for i in range(5):
    np.savetxt(f'cov_matrix{i}.csv', cov[:,i,:], delimiter=',')
    # when loading, pay attention to this syntax

if plot:
    # to plot covariance as errors, must sum the counts "added" from other orders as well as "stolen" by other orders
    # v giving order, > receiving order [g_idx, r_idx, pixel]
    #      9   8   7   6   5
    # 9 [  1   #   #   #   #  ]  < multiply counts*cov in Order 9 to add to other orders
    # 8 [  #   1   #   #   #  ]
    # 7 [  #   #   1   #   #  ]
    # 6 [  #   #   #   1   #  ]
    # 5 [  #   #   #   #   1  ]
    #      ^ multiply counts*cov in other orders to add to Order 9
    err_p, err_n = np.zeros([5, 2048]), np.zeros([5, 2048])
    for j in range(2048):
        err_p[:, j] = np.array([int(np.sum(cov[:, i, j] * spec[i, j]) - spec[i, j]) for i in range(5)])
        err_n[:, j] = np.array([int(np.sum(cov[i, :, j] * spec[i, j]) - spec[i, j]) for i in range(5)])
    plt.grid()
    for i in range(5):
        plt.plot(lambda_pixel[i, :], spec[i, :], 'k')
        plt.fill_between(lambda_pixel[i, :], spec[i, :] + err_p[i, :], spec[i, :] - err_n[i, :], alpha=0.5, edgecolor='gray',
                         facecolor='orange', linewidth=0.5)
    plt.title("Output Spectrum")
    plt.ylabel('Total Photons')
    plt.xlabel('Wavelength (nm)')
    plt.tight_layout()
    plt.show()
