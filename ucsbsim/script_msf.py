import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy import optimize
import scipy
import astropy.units as u
import time
import argparse
from datetime import datetime as dt
import logging
import os

from lmfit import Parameters, minimize
from mkidpipeline import photontable as pt
from synphot import SourceSpectrum, SpectralElement
from synphot.models import BlackBodyNorm1D
from specutils import Spectrum1D
import engine
from msf import MKIDSpreadFunction
from spectrograph import GratingSetup, SpectrographSetup
from detector import MKIDDetector
from mkidpipeline.photontable import Photontable
from plotting import quick_plot
import tables
from simsettings import SpecSimSettings

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

    # ==================================================================================================================
    # CONSTANTS
    # ==================================================================================================================


    # ==================================================================================================================
    # PARSE ARGUMENTS
    # ==================================================================================================================
    arg_desc = '''
               Extract the MSF Spread Function from the calibration spectrum.
               --------------------------------------------------------------
               This program loads the calibration photon table and conducts non-linear least squares fits to determine
               bin edges and the covariance matrix.
               '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)

    # required MSF args:
    parser.add_argument('output_dir',
                        metavar='OUTPUT_DIRECTORY',
                        help='Directory for the output files (str).')
    parser.add_argument('caltable',
                        metavar='CALIBRATION_PHOTON_TABLE',
                        help='Directory/name of the calibration photon table.')

    # optional MSF args:
    parser.add_argument('-pr', '--plotresults',
                        action='store_true',
                        default=False,
                        help='If passed, indicates that plots showing goodness-of-fit should be shown.')

    # set arguments as variables
    args = parser.parse_args()

    # ==================================================================================================================
    # START LOGGING TO FILE
    # ==================================================================================================================
    now = dt.now()
    logging.basicConfig(filename=f'{args.output_dir}/msf_{now.strftime("%Y%m%d_%H%M%S")}.log',
                        format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.info("The process of recovering the MKID Spread Function calibration spectrum is recorded."
                 f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    # ==================================================================================================================
    # OPEN PHOTON TABLE AND PULL NECESSARY DATA
    # ==================================================================================================================
    pt = Photontable(args.caltable)
    sim = pt.query_header('sim_settings')

    resid_map = np.arange(sim.npix, dtype=int) * 10 + 100  # TODO replace once known
    waves = pt.query(column='wavelength')
    resID = pt.query(column='resID')
    idx = [np.where(resID == resid_map[j]) for j in range(sim.npix)]
    photons_pixel = [waves[idx[j]].tolist() for j in range(sim.npix)]

    logging.info(f'Obtained calibration photon table from {args.caltable}.')

    # ==================================================================================================================
    # INSTANTIATE SPECTROGRAPH & DETECTOR
    # ==================================================================================================================
    if not os.path.exists(sim.R0s_file):
        IOError('File does not exist, check path and file name.')
    else:
        R0s = np.loadtxt(sim.R0s_file, delimiter=',')
        logging.info(f'\nThe individual R0s were imported from {sim.R0s_file}.')

    detector = MKIDDetector(sim.npix, sim.pixelsize, sim.designR0, sim.l0, R0s, resid_map)
    grating = GratingSetup(sim.l0, sim.m0, sim.m_max, sim.pixelsize, sim.npix, sim.focallength, sim.nolittrow)
    spectro = SpectrographSetup(grating, detector, sim.pixels_per_res_elem)
    eng = engine.Engine(spectro)
    nord = spectro.nord
    lambda_pixel = spectro.pixel_wavelengths().to(u.nm)[::-1]  # flip order, extracts in ascending wavelengths

    # ==================================================================================================================
    # MSF EXTRACTION STARTS
    # ==================================================================================================================
    wave = np.linspace(sim.minwave.to(u.nm).value - 100, sim.maxwave.to(u.nm).value + 100, 10000) * u.nm

    # retrieving the blazed calibration spectrum shape and converting to pixel-space:
    if sim.type_spectra == 'blackbody':
        spectra = SourceSpectrum(BlackBodyNorm1D, temperature=sim.temp)  # flux for star of 1 R_sun at distance of 1 kpc
    elif sim.type_spectra == 'something else':
        pass  # only blackbody supported now
    blazed_spectrum, _, _ = eng.blaze(wave, spectra)
    pix_leftedge = spectro.pixel_wavelengths(edge='left').to(u.nm).value
    blaze_shape = [eng.lambda_to_pixel_space(wave, blazed_spectrum[i], pix_leftedge[i]) for i in range(nord)][::-1]
    blaze_shape /= np.max(blaze_shape)  # normalize max to 1
    blaze_shape[blaze_shape == 0] = 1  # prevent divide by 0 or very small num. issue

    # initializing empty arrays for for-loop:
    covariance = np.zeros([nord, nord, sim.npix])
    photon_bins = np.zeros([nord + 1, sim.npix])
    photon_bins[-1, :] = sim.maxwave.to(u.nm).value + 300
    spec = np.empty([nord, sim.npix])
    reduced_chi2, flagger = np.empty(sim.npix), np.zeros(sim.npix)

    # determine number of Gaussian functions per pixel:
    n_gauss = np.sum((np.asarray(lambda_pixel.to(u.nm).value) >= sim.minwave.to(u.nm).value).astype(int), axis=0)

    # do least-squares fit:
    for j in detector.pixel_indices:
        s = 1 if n_gauss[j] == nord-1 else 0  # determines starting index
        opt_params, old_params = eng.fit_gaussians(j, photons_pixel[j], n_gauss[j])
        reduced_chi2[j] = opt_params.redchi
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

    logging.info(f"Finished fitting all {sim.npix} pixels. There are "
                 f"{np.sum(flagger == 1)}/{np.sum(flagger == 2)}/{np.sum(flagger == 3)} "
                 f"pixels flagged for means/sigmas/both being within 5% of boundaries.")

    # assign bin edges and covariance matrix to MSF class and save:
    msf = MKIDSpreadFunction(bin_edges=photon_bins, cov_matrix=covariance, orders=spectro.orders, sim_settings=sim)
    msf_file = f'{args.output_dir}/msf_R0{sim.designR0}_{sim.pixellim}.npz'
    msf.save(msf_file)
    logging.info(f'\nSaved MSF bin edges and covariance matrix to {msf_file}.')
    logging.info(f'\nTotal script runtime: {((time.time() - tic) / 60):.2f} min.')
    # ==================================================================================================================
    # MSF EXTRACTION ENDS
    # ==================================================================================================================


    # ==================================================================================================================
    # DEBUGGING PLOTS
    # ==================================================================================================================
    #msf.plot()  # TODO

    # plot reduced chi-sq (should not contain significant outliers):
    fig1, ax1 = plt.subplots(1, 1)
    quick_plot(ax1, [range(sim.npix)], [reduced_chi2], title='Reduced Chi-Sq', xlabel='Pixel', first=True)
    plt.show()

    # plot unblazed, unshaped calibration spectrum (should be mostly flat line):
    fig2, ax2 = plt.subplots(1, 1)
    quick_plot(ax2, lambda_pixel, spec/blaze_shape, labels=[f'O{i}' for i in spectro.orders[::-1]],
               first=True, title='Continuum-normalized calibration', xlabel='Wavelength (nm)', ylabel='Photon Count')
    plt.show()
