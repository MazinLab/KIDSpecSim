import numpy as np
import scipy.interpolate as interp
from scipy import optimize
import scipy
import matplotlib.pyplot as plt
from mkidpipeline import photontable as pt
from synphot import SourceSpectrum, SpectralElement
from synphot.models import BlackBodyNorm1D
from specutils import Spectrum1D
import astropy.units as u
import fit
from spectrograph import GratingSetup, SpectrographSetup
from detector import MKIDDetector

"""
This script:

-Takes calibration file (see syntax below) and fits Gaussians to each pixel.
-If using simulation-generated calibration file, separately generates the relative amplitudes of the blackbody function
multiplied with the blaze function all convolved to go from fluxden to flux space.
-If using IRL calibration file, convolves the theoretical shape of the calibration spectrum with MKID response.
-Divides this convolved shape out of the Gaussian amplitude fit to make normalized spectrum. These normalized Gaussian
fits are known as the MKID spread function (MSF).

-Uses overlap points of the Gaussians to get bins for each order.

-Calculates relative overlap between Gaussians and converts them into a fractional "covariance" matrix. This
matrix details what fraction of each order Gaussian was grouped into another order Gaussian due to binning.
-Saves blackbody-blaze amplitudes, order bins, and covariance to files.


*********BEFORE RUNNING THIS msf.py SCRIPT*********

*USE SIM-GENERATED CALIBRATION FILE FROM main.py (generated_cal=True):
 ENSURE SPECTYPE/PIXEL_LIM/R0 ARE EQUAL TO SIM-GENERATED CALIBRATION SPECTRUM FROM MAIN.PY.

*OR IMPORT IRL CALIBRATION FILE (generated_cal=False).
 SUPPLY FILE NAME FOR CALIBRATION DATA FROM IRL OBSERVATION AND THEORETICAL SHAPE (BB, THORIUM, ETC.).

*CHOOSE TO ENABLE 'CHECKFITS' FOR CHECKING PARAMETER FIT AT EACH PIXEL OR NOT (DEFAULT).
"""
generated_cal = True
checkfits = False
if generated_cal:
    spec_type = 'blackbody'
    pixel_lim = 50000
    R0 = 8
    table = f'{spec_type}_{pixel_lim}_R0{R0}.h5'
else:
    cal_file = 'write_filename_here.h5'
    spec_type = 'supply_shape_here'  # as calibration sources are added, list will be expanded
    with open(cal_file) as f:
        table = np.loadtxt(f, delimiter=",")

# add another section for spectrograph properties if changing properties becomes necessary

# opening the h5 file containing the photon table
file = pt.Photontable(table)
waves = file.query(column='wavelength')
resID = file.query(column='resID')

if __name__ == '__main__':
    bb = SourceSpectrum(BlackBodyNorm1D, temperature=4300)  # flux for star of 1 R_sun at distance of 1 kpc
    wave = np.linspace(300, 1000, 10000) * u.nm

    grating = GratingSetup()
    detector = MKIDDetector(R0_fixed=True)
    spectrograph = SpectrographSetup(grating, detector)

    blaze = spectrograph.blaze(wave)
    blaze_bb = [interp.interp1d(wave.value, blaze[i] * bb(wave).value, fill_value=0) for i in range(5)]
    dl_pixel = spectrograph.pixel_scale / spectrograph.angular_dispersion()  # wavelength extent of each pixel
    lambda_pixel = spectrograph.pixel_center_wavelengths()

    # sigma of each pixel
    dl_mkid_pixel = detector.mkid_resolution_width(lambda_pixel, detector.pixel_indices) / 2.355

    blaze_bb = blaze_bb[::-1]  # flip orders (for orders to be in ascending wavelength 9 -> 5)
    lambda_pixel = lambda_pixel[::-1, :].value
    dl_mkid_pixel = dl_mkid_pixel[::-1, :].value
    dl_pixel = dl_pixel[::-1, :]

    cov = np.zeros([5, 5, 2048])
    photon_bins = np.zeros([6, 2048])
    unblazer = np.empty([5, 2048])
    spec = np.empty([5, 2048])
    photon_bins[-1, :] = 1000
    resid_map = np.arange(2048, dtype=int) * 10 + 100
    p = 0
    if checkfits:
        fig, axes = plt.subplots(5, 2, figsize=(8.5, 14))
        axes = axes.ravel()

    for j in range(2048):  # sorting by resID and fitting GMM to each pixel
        n = 4 if lambda_pixel[0, j] <= 390 else 5
        idx = np.where(resID == resid_map[j])  # splitting photon table by resID
        photons_j = waves[idx].tolist()
        bins = int(6 * len(photons_j) ** (1 / 3))  # 6x the cube root of the number of items
        counts, edges_j = np.histogram(photons_j, bins=bins)  # binning photons for shape fitting
        counts_j = np.array([float(x) for x in counts])
        centers_j = edges_j[:-1] + np.diff(edges_j) / 2
        params = list(lambda_pixel[5 - n:, j]) + list(dl_mkid_pixel[5 - n:, j]) + [np.max(counts_j) for i in range(n)]
        # ^ guess mu, sig, A
        opt_p, _ = optimize.curve_fit(fit.gauss_summed, centers_j, counts_j, p0=params)
        opt_mu = opt_p[:int(len(opt_p) / 3)]  # splitting array to mus
        opt_sig = opt_p[int(len(opt_p) / 3):int(2 * len(opt_p) / 3)]  # splitting array to sigmas
        opt_A = opt_p[int(2 * len(opt_p) / 3):]  # splitting array to amplitudes
        # TODO ensure again that this is analytically correct
        # unblazing routine, value of bb+blaze * pixel extent in lambda / sigma
        const = [blaze_bb[5 - n + i](lambda_pixel[5 - n + i, j]) for i in range(n)]
        unblazer[5 - n:, j] = const * dl_pixel[5 - n:, j]  # from analytic convolution
        # unblaze_A = opt_A
        unblaze_A = opt_A / unblazer[5 - n:, j]  # divide out bb/blaze to get flat field spectrum for fitted Gaussians

        photon_bins[6 - n:5, j] = fit.gauss_intersect(opt_mu, opt_sig, unblaze_A)
        for m, o in enumerate(photon_bins[:, j]):
            photon_bins[m, j] = m if o == 0 else photon_bins[m, j]  # ensure bins are increasing if many 0s

        spec[5 - n:, j], edges = np.histogram(photons_j, bins=photon_bins[5 - n:, j])
        # binning photons again by photon_bins edges

        # cov: calculate % of photons that will fall into other orders from any given order (9 -> 5 for index 0 -> 4)
        gauss_sum = [np.sum(fit.gauss(np.arange(wave[0].value, wave[-1].value, .01),
                                      opt_mu[i], opt_sig[i], unblaze_A[i])) for i in range(n)]
        cov[5 - n:, 5 - n:, j] = [[
            np.sum(fit.gauss(np.arange(photon_bins[5 - n + k, j], photon_bins[5 - n + k + 1, j], 0.01), opt_mu[i],
                             opt_sig[i], unblaze_A[i])) / gauss_sum[i] for k in range(n)] for i in range(n)]
        # TODO plot least squares sum also to check fit
        if checkfits:  # this will generate 2048 plots in 10 subplot increments to check for parameter fits
            if j % 10 == 0 and j != 0:
                p = 0
                fig, axes = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(8.5, 14))
                axes = axes.ravel()
            axes[p].grid()
            axes[p].set_title(f"Pixel {j + 1}")
            axes[p].set_xlim([350, 900])
            axes[p].hist(photons_j, bins=bins, color='k')
            for i in range(n):
                axes[p].plot(wave, fit.gauss(wave.value, opt_mu[i], opt_sig[i], opt_A[i]), color=f'C{5 - n + i}')
            for i in range(n):
                axes[p].axvline(photon_bins[i + 1, j], color='r', linestyle='--', linewidth=1)
            axes[p].axvline(photon_bins[0, j], color='r', linestyle='--', linewidth=1, label='Bin Edges')
            axes[p].legend()
            if p == 9:
                for i in [8, 9]:
                    axes[i].set_xlabel("Wavelength (nm)")
                for i in [0, 2, 4, 6, 8]:
                    axes[i].set_ylabel("Photon Count")
                plt.tight_layout()
                plt.show()
            else:
                pass
            if j == 2047:
                for i in [6, 7]:
                    axes[i].set_xlabel("Wavelength (nm)")
                for i in [0, 2, 4, 6]:
                    axes[i].set_ylabel("Photon Count")
                for i in [8, 9]:
                    axes[i].remove()
                plt.tight_layout()
                plt.show()
            else:
                pass
            p += 1
    unblazer /= unblazer[unblazer < 1].max()
    new_spec = spec / unblazer
    np.nan_to_num(new_spec)
    unblazer[unblazer > 1] = 0

    plt.grid()
    for i in range(5):
        # plt.plot(lambda_pixel[i, :], new_spec[i,:])
        plt.plot(lambda_pixel[i, :], spec[i, :] / spec.max(), 'k')
        plt.plot(lambda_pixel[i, :], unblazer[i, :], 'r')
    plt.title("Output Spectrum")
    plt.ylabel('Total Photons')
    plt.xlabel('Wavelength (nm)')
    plt.tight_layout()
    plt.show()

    np.savetxt(f'cal_bins_R0{R0}.csv', photon_bins, delimiter=',')
    # saving calibration bin edges to file for use in other spectra
    np.savetxt(f'unblazer_R0{R0}.csv', unblazer, delimiter=',')
    for i in range(5):
        np.savetxt(f'cov_R0{R0}_{i}.csv', cov[:, i, :], delimiter=',')
        # when loading, pay attention to this syntax

    print("Done.")
