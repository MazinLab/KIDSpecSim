import numpy as np
import scipy.interpolate as interp
from scipy import optimize
import scipy
from scipy.constants import h, c, k
import matplotlib.pyplot as plt
from mkidpipeline import photontable as pt
# from synphot import SourceSpectrum, SpectralElement
# from synphot.models import BlackBodyNorm1D
from specutils import Spectrum1D
import astropy.units as u
import fit

"""
This script:
-Takes calibration file (generated from main.py) from a smooth blackbody and fits Gaussians to each pixel. These values
are known as the MKID spread function (MSF).
-Separately generates the relative amplitudes of the blackbody function multiplied with the blaze function all
convolved to go from fluxden to flux space. Divides this out of the Gaussian amplitude fit to make normalized spectrum.
-Uses overlap points of the Gaussians to get bins for each order.
-Calculates relative overlap between fitted Gaussians and converts them into a fractional "covariance" matrix. This
matrix details what fraction of each order Gaussian was grouped into another order Gaussian.
-Saves blackbody-blaze amplitudes, order bins, and covariance to files.

***BEFORE RUNNING THIS MSF.PY SCRIPT***
*CHOOSE TO ENABLE 'CLICKTHROUGHS' FOR CHECKING PARAMETER FIT AT EACH PIXEL OR NOT (DEFAULT).
*ENSURE PIXEL_LIM/R0 ARE EQUAL TO PRE-GENERATED CALIBRATION SPECTRUM FROM MAIN.PY.
"""
clickthrough = True
pixel_lim = 50000
R0 = 8

# import pixel central wavelengths
with open(f'lambda_pixel.csv') as f:
    lambda_pixel = np.loadtxt(f, delimiter=",")
# randomized R0s that deviate slightly from theoretical
with open(f'generated_R0s.csv') as f:
    R0s = np.loadtxt(f, delimiter=",")

if __name__ == '__main__':
    # analytic normalized blackbody equation in PHOTLAM, verified to match BlackbodyNorm1D used in main.py
    u.photlam = u.photon / u.s / u.cm ** 2 / u.AA  # photon flux per wavelength
    u.flam = u.erg / u.s / u.cm ** 2 / u.AA  # energy flux per wavelength
    wave = np.linspace(100, 1000, 10000) * u.nm
    B_lam = (2 * (h * u.J * u.s) * (c * u.m / u.s) ** 2 / wave ** 5 /
             (np.exp((h * u.J / u.Hz) * (c * u.m / u.s) / wave / (k * u.J / u.K) / (4300 * u.K)) - 1)).to(u.flam)
    # ^ blackbody equation in energy fluxden per steradian for 4300 K source
    R_sun = 695700 * u.km
    d = 1 * u.kpc
    Omega = np.pi * R_sun ** 2 / d ** 2  # steradian calculation for source of size R_sun at distance of 1 kpc
    norm_B = (B_lam * Omega).decompose().to(u.flam)  # multiplying out steradians
    B_photlam = (wave / (h * u.J * u.s) / (c * u.m / u.s) * norm_B * u.ph).to(
        u.photlam)  # energy to photon fluxden calc

    # calculating the blaze equation with excess edges for calibration
    l0 = 800 * u.nm  # < v relevant spectrograph properties
    m0 = 5
    l0_center = l0 / (1 + 1 / (2 * m0))
    focal_length = 350 * u.mm
    npix = 2048
    pixel_size = 20 * u.micron
    angular_dispersion = 2 * np.arctan(pixel_size * npix / 2 / focal_length) / (l0_center / m0)
    incident_angle = np.arctan((l0_center / 2 * angular_dispersion).value) * u.rad
    d = m0 * l0_center / 2 / np.sin(incident_angle)  # groove size
    alpha = incident_angle
    delta = incident_angle
    beta_central_pixel = incident_angle
    pixel_scale = np.arctan(pixel_size / focal_length)  # angular extent of pixel in rad
    center_offset = pixel_size * (np.arange(npix, dtype=int) + 0.5 - npix / 2)
    new_beta = beta_central_pixel + np.arctan(center_offset / focal_length)

    blaze_bb = np.zeros([5, 10000])
    dl_pixel = np.zeros([5, 2048])
    for i in range(5):
        beta = np.arcsin((i + 5) * wave / d - np.sin(alpha))
        w = np.cos(beta) * np.cos(alpha - delta) / (np.cos(alpha) * np.cos(beta - delta))
        w[w > 1] = 1  # k must be the minimum value between k and 1
        q4 = np.cos(delta) - np.sin(delta) / np.tan((alpha + beta) / 2)
        if alpha < delta:
            rho = np.cos(delta)  # 2 different rho depending on whether alpha or delta is larger
        else:
            rho = np.cos(alpha) / np.cos(alpha - delta)
        blaze_i = w * np.sinc(((i + 5) * rho * q4).value) ** 2  # omit np.pi as np.sinc includes it
        delta_angle = np.tan(beta - beta_central_pixel)
        x = focal_length * delta_angle / pixel_size + npix / 2
        mask = (x >= -int(npix * (0.3 - i / 20))) & (x < int(npix * (1.35 - i / 20)))
        blaze_bb[i, mask] = blaze_i[mask]  # applying mask that is slightly beyond edges of array
        blaze_bb[i, :] *= B_photlam.value  # multiplying the blackbody through each order blaze
        dl_pixel[i, :] = pixel_scale / ((i + 5) / (d * np.cos(new_beta)) * u.rad)  # wavelength extent of each pixel

    # theoretical sigma of each pixel
    sigma_mkid = lambda_pixel ** 2 / (R0 * l0) / 2.355
    # 'true,' randomized sigma for each pixel that deviates slightly from theoretical
    dl_mkid_pixel = lambda_pixel ** 2 / (R0s[None, :] * l0) / 2.355

    blaze_bb = blaze_bb[::-1, :]  # flip orders (for orders to be in ascending wavelength 9 -> 5)
    lambda_pixel = lambda_pixel[::-1, :]
    dl_mkid_pixel = dl_mkid_pixel[::-1, :].value
    sigma_mkid = sigma_mkid[::-1, :].value
    dl_pixel = dl_pixel[::-1, :]

    # opening the h5 file containing the photon table
    file = pt.Photontable(f'blackbody_{pixel_lim}_R0{R0}.h5')
    waves = file.query(column='wavelength')
    resID = file.query(column='resID')

    colors = ['red', 'orange', 'green', 'blue', 'purple']
    cov = np.zeros([5, 5, 2048])
    photon_bins = np.zeros([6, 2048])
    unblazer = np.empty([5, 2048])
    photon_bins[-1, :] = 1000
    resid_map = np.arange(2048, dtype=int) * 10 + 100
    p = 0
    if clickthrough:
        fig, axes = plt.subplots(5, 2, figsize=(8.5, 14))
        ax = [sub for x in axes for sub in x]

    for j in range(2048):  # sorting by resID and fitting GMM to each pixel
        n = 4 if lambda_pixel[0, j] <= 390 else 5
        idx = np.where(resID == resid_map[j])  # splitting photon table by resID
        photons_j = waves[idx].tolist()
        bins = int(6 * len(photons_j) ** (1 / 3))  # 6x the cube root of the number of items
        counts, edges_j = np.histogram(photons_j, bins=bins)  # binning photons for shape fitting
        counts_j = np.array([float(x) for x in counts])
        centers_j = edges_j[:-1] + np.diff(edges_j) / 2
        # TODO fit fot all parameters simultaenously
        params = list(lambda_pixel[5-n:, j]) + list(dl_mkid_pixel[5 - n:, j]) + [np.max(counts_j) for i in range(n)]
        # ^ guess mu, sig, A
        opt_p, _ = optimize.curve_fit(fit.gauss_summed, centers_j, counts_j, p0=params)
        opt_mu = opt_p[:int(len(opt_p) / 3)]  # splitting array to mus
        opt_sig = opt_p[int(len(opt_p) / 3):int(2*len(opt_p) / 3)]  # splitting array to sigmas
        opt_A = opt_p[int(2*len(opt_p) / 3):]  # splitting array to amplitudes

        # unblazing routine, value of bb+blaze * pixel extent in lambda / sigma
        const = np.array([blaze_bb[5 - n + i, np.where(np.abs(wave.value - lambda_pixel[5 - n + i, j]) <
                                                       5e-2)[0][0]] for i in range(n)])
        unblazer[5 - n:, j] = const * dl_pixel[5 - n:, j] / opt_sig  # from analytic convolution
        #unblaze_A = opt_A
        unblaze_A = opt_A / unblazer[5-n:, j]  # divide out bb/blaze to get flat field spectrum for fitted Gaussians

        photon_bins[6 - n:5, j] = fit.gauss_intersect(opt_mu, opt_sig, unblaze_A)
        for m, o in enumerate(photon_bins[:, j]):
            photon_bins[m, j] = m if o == 0 else photon_bins[m, j]  # ensure bins are increasing if many 0s
        # cov: calculate % of photons that will fall into other orders from any given order (9 -> 5 for index 0 -> 4)
        for i in range(n):
            sum_i = np.sum(fit.gauss(np.arange(wave[0].value, wave[-1].value, .01), opt_mu[i], opt_sig[i], unblaze_A[i]))
            for k in range(n):
                cov[5 - n + i, 5 - n + k, j] = np.sum(
                    fit.gauss(np.arange(photon_bins[5 - n + k, j], photon_bins[5 - n + k + 1, j], 0.01),
                          opt_mu[i], opt_sig[i], unblaze_A[i])) / sum_i
        # TODO plot least squares sum also to check fit
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
                ax[p].plot(wave, fit.gauss(wave.value, opt_mu[i], opt_sig[i], opt_A[i]), color=colors[5 - n + i])
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
                plt.show()
            else:
                pass
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
                # plt.savefig(f'cal_plots/cal_2040to2048.png')
                plt.show()
            else:
                pass
            p += 1
    unblazer /= unblazer.max()
    np.nan_to_num(unblazer, 1)
    np.savetxt(f'cal_bins_R0{R0}.csv', photon_bins,
               delimiter=',')  # saving calibration bin edges to file for use in other spectra
    np.savetxt(f'unblazer_R0{R0}.csv', unblazer, delimiter=',')
    for i in range(5):
        np.savetxt(f'cov_R0{R0}_{i}.csv', cov[:, i, :], delimiter=',')
        # when loading, pay attention to this syntax
    print("Done.")
