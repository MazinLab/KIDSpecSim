import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy import optimize
import scipy
import astropy.units as u
from astropy.table import QTable
import time
from datetime import datetime as dt
import logging
from sorcery import dict_of

from lmfit import Parameters, minimize, fit_report
from mkidpipeline import photontable as pt
from synphot import SourceSpectrum, SpectralElement
from synphot.models import BlackBodyNorm1D
from specutils import Spectrum1D
from engine import Engine, quick_plot
import engine
from spectrograph import GratingSetup, SpectrographSetup
from detector import MKIDDetector

"""
Extraction of the MKID Spread Function from calibration spectrum. The steps are:
-Load the calibration spectrum and fit Gaussians to each pixel. Some have n_ord curves, some n_ord-1 (due to bandpass).
-If calibration spectrum type is known, the relative amplitudes of the spectrum shape multiplied with the blaze, all
 converted to pixel space, are generated.
-Divide this calibration-blaze shape out of the Gaussian amplitude fit to make a flux-normalized spectrum.
 These normalized Gaussian fits are known as the MKID spread function (MSF).
-Use overlap points of the Gaussians to get bin edges for each order.
-Calculate fractional overlap between Gaussians and converts them into an n_ord x n_ord "covariance" matrix for each
 pixel. This matrix details what fraction of each order Gaussian was grouped into another order Gaussian due to binning.
 This will later be used to determine the error band of some extracted spectrum.
-Saves newly obtained bin edges and covariance matrices to files.
"""

if __name__ == '__main__':
    tic = time.time()  # recording start time for script

    # ========================================================================
    # Constants which may or may not be needed in the future:
    # ========================================================================

    # ========================================================================
    # Assign spectrograph properties below if different from defaults. Check the module spectrograph for syntax.
    # ========================================================================
    # R0 = 8
    # etc.

    # ========================================================================
    # MSF extraction settings:
    # ========================================================================
    type_of_spectra = 'blackbody'
    plot_fits = False  # warning: will show 2048 plots if True

    # ========================================================================
    # MSF extraction begins:
    # ========================================================================
    now = dt.now()
    logging.basicConfig(filename=f'output_files/{type_of_spectra}/logging/msf_{now.strftime("%Y%m%d_%H%M%S")}.log',
                        format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.info(f"The process of recovering the MKID Spread Function from the {type_of_spectra} "
                 f"calibration spectrum is recorded."
                 f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    # setting up engine and spectrograph, obtaining various needed properties:
    eng = Engine(SpectrographSetup(GratingSetup(), MKIDDetector()))
    R0 = eng.spectrograph.detector.design_R0
    nord = len(eng.spectrograph.orders)
    npix = eng.spectrograph.detector.n_pixels
    resid_map = np.arange(npix, dtype=int) * 10 + 100  # TODO replace once known
    lambda_pixel = eng.spectrograph.pixel_center_wavelengths().to(u.nm)[::-1]
    sigma_pixel = eng.spectrograph.sigma_mkid_pixel().to(u.nm)[::-1]
    wave = np.linspace(eng.spectrograph.minimum_wave.value - 100, eng.spectrograph.detector.waveR0.value + 100,
                       10000) * u.nm

    # retrieving the blazed calibration spectrum shape and converting to pixel-space:
    if type_of_spectra == 'blackbody':
        bb = SourceSpectrum(BlackBodyNorm1D, temperature=4300)  # flux for star of 1 R_sun at distance of 1 kpc
        blaze_shape = eng.pixelspace_blaze(wave, bb)[::-1]
        blaze_shape /= np.max(blaze_shape)  # normalize max to 1
        blaze_shape[blaze_shape == 0] = 1  # prevent divide by 0 or very small # issue
    elif type_of_spectra == 'something else':
        pass  # do something

    # initializing empty arrays for for-loop: TODO would like to streamline this
    covariance = np.zeros([nord, nord, npix])
    photon_bins = np.zeros([nord + 1, npix])
    photon_bins[-1, :] = eng.spectrograph.detector.waveR0.value + 300
    spec = np.empty([nord, npix])
    reduced_chisq, flagger = np.empty(npix), np.zeros(npix)

    # open and sort calibration spectrum photon table:
    cal_table = f'output_files/calibration/table_R0{R0}.h5'
    photons_pixel = eng.open_table(resid_map, cal_table)
    logging.info(f'Obtained calibration photon table from {cal_table}.')

    # determine number of Gaussian functions per pixel:
    n_gauss = np.sum((np.asarray(lambda_pixel.value) >= 400).astype(int), axis=0)

    # do least-squares fit:
    for j in eng.spectrograph.detector.pixel_indices:
        s = 1 if n_gauss[j] == eng.spectrograph.nord-1 else 0  # determines starting index
        opt_params, old_params = eng.fit_gaussians(j, photons_pixel[j], n_gauss[j], plot=plot_fits)
        reduced_chisq[j] = opt_params.redchi
        mu, sig, A = engine.param_to_array(opt_params, n_gauss[j], post_opt=True)
        old_mu, old_sig, _ = engine.param_to_array(old_params, n_gauss[j], post_opt=False)

        # if optimized parameters are near boundaries, flag:
        if (np.abs(old_mu - mu) > 0.95 * old_sig).any():
            flagger[j] += 1  # TODO make this more robust
        if (np.abs(old_sig - sig) > 0.95 * old_sig / 2).any():
            flagger[j] += 2

        # divide out blaze-calibration shape before determining bin edges:
        unblaze_A = A / blaze_shape[s:, j]
        photon_bins[s + 1:nord, j] = engine.gauss_intersect(mu, sig, unblaze_A)
        spec[s:, j], _ = np.histogram(photons_pixel[j], bins=photon_bins[s:, j])

        # determine covariance between orders:
        gauss_sum = np.sum(engine.gauss(wave.value, mu[:, None], sig[:, None], unblaze_A[:, None]),
                           axis=0) * (wave[1] - wave[0]).value
        gauss_int = [interp.InterpolatedUnivariateSpline(wave.value, engine.gauss(
            wave.value, mu[i], sig[i], unblaze_A[i]), k=1, ext=1) for i in range(n_gauss[j])]
        covariance[s:, s:, j] = [[gauss_int[i].integral(photon_bins[s + k, j], photon_bins[s + k + 1, j]) / gauss_sum[i]
                                  for k in range(n_gauss[j])] for i in range(n_gauss[j])]

    photon_bins[photon_bins == 0] = np.where(photon_bins == 0)[0]  # ensure bins increase trivially if initial 0s

    logging.info(f"Finished fitting all {npix} pixels. There are "
                 f"{np.sum(flagger == 1)}/{np.sum(flagger == 2)}/{np.sum(flagger == 3)} "
                 f"pixels flagged for means/sigmas/both being within 5% of boundaries.")

    # save bins and covariance to FITS file format:
    # assortment of spectrograph properties TODO figure out what else to add
    meta = dict_of(R0, eng.spectrograph.detector.waveR0.value, npix, nord, type_of_spectra)
    bins = QTable(list(photon_bins * u.nm), names=[f'{i}' for i in range(nord+1)],
                  meta={'name': 'Bin Edges for Pixel', **meta})
    bins_file = f'output_files/calibration/msf_bins_R0{R0}.fits'
    bins.write(bins_file, format='fits', overwrite=True)

    # TODO not sure how to best save a 3D array to FITS
    cov_files = [f'output_files/calibration/msf_cov{i}_R0{R0}.fits' for i in range(nord)]
    for i in range(nord):
        cov = QTable(list(covariance[:, i, :]), names=[f'{j}' for j in eng.spectrograph.orders],
                     meta={'name': f'Covariance (for other orders) in Order {i}', **meta})
        cov.write(cov_files[i], format='fits', overwrite=True)

    logging.info(f'\nSaved order bin edges to {bins_file} and covariance matrices to {cov_files}.')
    logging.info(f'\nTotal script runtime: {((time.time() - tic) / 60):.2f} min.')
    # ========================================================================
    # MSF extraction ends
    # ========================================================================


    # ========================================================================
    # Plots for debugging and reasoning through:
    # ========================================================================

    # plot reduced chi-sq (should not contain significant outliers):
    fig1, ax1 = plt.subplots(1, 1)
    quick_plot(ax1, [range(npix)], [reduced_chisq], title='Reduced Chi-Sq', xlabel='Pixel', first=True)
    plt.show()

    # plot unblazed, unshaped calibration spectrum (should be mostly flat line):
    fig2, ax2 = plt.subplots(1, 1)
    quick_plot(ax2, lambda_pixel, spec/blaze_shape, labels=[f'O{i}' for i in eng.spectrograph.orders[::-1]],
               first=True, title='Continuum-normalized calibration', xlabel='Wavelength (nm)', ylabel='Photon Count')
    plt.show()
