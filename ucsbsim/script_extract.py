import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime as dt
import logging
import astropy.units as u
import argparse
from astropy.io import fits
from astropy.table import Table
import os

from ucsbsim.spectrograph import GratingSetup, SpectrographSetup
from ucsbsim.detector import MKIDDetector, wave_to_phase
import ucsbsim.engine as engine
from ucsbsim.plotting import quick_plot
from synphot.models import BlackBodyNorm1D, ConstFlux1D
from synphot import SourceSpectrum
from ucsbsim.msf import MKIDSpreadFunction
from mkidpipeline.photontable import Photontable

"""
Extraction of an observation spectrum using the MSF products. The steps are:
-Open the MSF products: order bin edges and covariance matrices.
-Open the observation/emission photon table and bin for orders.
-Calculate the errors on each point by multiplying the covariance matrices through the spectrum.
-Divide out the blaze function from the resulting spectrum and +/- errors and save to FITS.
-Show final spectrum as plot.
"""

if __name__ == '__main__':
    tic = time.time()  # recording start time for script
    u.photlam = u.photon / u.s / u.cm ** 2 / u.AA  # photon flux per wavelength

    # ==================================================================================================================
    # CONSTANTS
    # ==================================================================================================================


    # ==================================================================================================================
    # PARSE ARGUMENTS
    # ==================================================================================================================
    arg_desc = '''
    Extract a spectrum using the MKID Spread Function.
    --------------------------------------------------------------
    This program loads the observation photon table and uses the MSF bins and covariance matrix to
    extract the spectrum.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)

    # required MSF args:
    parser.add_argument('output_dir',
                        metavar='OUTPUT_DIRECTORY',
                        help='Directory for the output files (str).')
    parser.add_argument('msf_file',
                        metavar='MKID_SPREAD_FUNCTION_FILE',
                        help='Directory/name of the MKID Spread Function file (str).')
    parser.add_argument('obstable',
                        metavar='OBSERVATION_PHOTON_TABLE',
                        help='Directory/name of the observation spectrum photon table (str).')

    # get arguments
    args = parser.parse_args()

    # ==================================================================================================================
    # CHECK AND/OR CREATE DIRECTORIES
    # ==================================================================================================================
    os.makedirs(f'{args.output_dir}/logging', exist_ok=True)

    # ==================================================================================================================
    # START LOGGING TO FILE
    # ==================================================================================================================
    now = dt.now()
    logging.basicConfig(filename=f'{args.output_dir}/logging/extract_{now.strftime("%Y%m%d_%H%M%S")}.log',
                        format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info(f"The extraction of an observed spectrum is recorded."
                 f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    # ==================================================================================================================
    # OPEN MSF FILE AND OBSERVATION PHOTON TABLE
    # ==================================================================================================================
    msf = MKIDSpreadFunction(filename=args.msf_file)
    sim_msf = msf.sim_settings.item()

    logging.info(f'Obtained MKID Spread Function from {args.msf_file}.')

    pt = Photontable(args.obstable)
    sim = pt.query_header('sim_settings')

    # do a check to ensure all simulation settings are equal
    if sim_msf.npix != sim.npix or sim_msf.l0 != sim.l0 or sim_msf.minwave != sim.minwave:
        raise ValueError('Simulation settings between the calibration and observation are not the same, '
                         'this extraction cannot be performed.')

    resid_map = np.arange(sim.npix, dtype=int) * 10 + 100  # TODO replace once known
    waves = pt.query(column='wavelength')
    resID = pt.query(column='resID')
    idx = [np.where(resID == resid_map[j]) for j in range(sim.npix)]
    photons_pixel = [waves[idx[j]].tolist() for j in range(sim.npix)]

    logging.info(f'Obtained observation photon table from {args.obstable}.')

    # ==================================================================================================================
    # INSTANTIATE SPECTROGRAPH & DETECTOR
    # ==================================================================================================================
    if not os.path.exists(sim.R0s_file):
        IOError('File does not exist, check path and file name.')
    else:
        R0s = np.loadtxt(sim.R0s_file, delimiter=',')
        logging.info(f'\nThe individual R0s were imported from {sim.R0s_file}.')

    detector = MKIDDetector(sim.npix, sim.pixelsize, sim.designR0, sim.l0, R0s, None, resid_map)
    grating = GratingSetup(sim.alpha, sim.delta, sim.groove_length)
    spectro = SpectrographSetup(sim.m0, sim.m_max, sim.l0, sim.pixels_per_res_elem, sim.focallength, grating, detector)
    eng = engine.Engine(spectro)
    nord = spectro.nord
    lambda_pixel = spectro.pixel_wavelengths().to(u.nm)[::-1]
    sim_phase = wave_to_phase(spectro.pixel_wavelengths().to(u.nm)[::-1], minwave=sim.minwave, maxwave=sim.maxwave)
    # flip order axis, ascending phase

    # ==================================================================================================================
    # OBSERVATION SPECTRUM EXTRACTION STARTS
    # ==================================================================================================================
    '''
    # obtain the pixel-space blaze function:
    wave = np.linspace(sim.minwave.value - 100, sim.maxwave.value + 100, 10000) * u.nm
    # retrieving the blazed calibration spectrum shape assuming it is known and converting to pixel-space wavelengths:
    if sim.type_spectra == 'blackbody':
        spectra = SourceSpectrum(BlackBodyNorm1D, temperature=sim.temp)  # flux for star of 1 R_sun at distance of 1 kpc
    else:
        spectra = SourceSpectrum(ConstFlux1D, amplitude=1)  # only blackbody supported now
    blazed_spectrum, _, _ = eng.blaze(wave, spectra)
    # TODO can we assume any knowledge about blaze shape? if not, how to divide out eventually?
    lambda_left = spectro.pixel_wavelengths(edge='left').to(u.nm).value
    blaze_shape = [eng.lambda_to_pixel_space(wave, blazed_spectrum[i], lambda_left[i]) for i in range(nord)][::-1]
    blaze_shape /= np.max(blaze_shape)  # normalize max to 1
    blaze_shape[blaze_shape == 0] = 1  # prevent divide by 0 or very small num. issue
    '''
    spec = np.zeros([nord, sim.npix])
    for j in detector.pixel_indices:
        spec[msf.val_idx[j], j], _ = np.histogram(
            photons_pixel[j], bins=msf.bin_edges[np.isfinite(msf.bin_edges[:, j]), j])
        # binning photons by MSF bins edges

    # to plot covariance as errors, must sum the counts "added" from other orders as well as "stolen" by other orders
    # v giving order, > receiving order [g_idx, r_idx, pixel]
    #      9   8   7   6   5
    # 9 [  1   #   #   #   #  ]  < multiply counts*cov in Order 9 to add to other orders
    # 8 [  #   1   #   #   #  ]
    # 7 [  #   #   1   #   #  ]
    # 6 [  #   #   #   1   #  ]
    # 5 [  #   #   #   #   1  ]
    #      ^ multiply counts*cov in other orders to add to Order 9

    err_p = np.array([[int(np.sum(msf.cov_matrix[:, i, j] * spec[:, j])) -
                       msf.cov_matrix[i, i, j] * spec[i, j] for j in detector.pixel_indices] for i in range(nord)])
    err_n = np.array([[int(np.sum(msf.cov_matrix[i, :, j] * spec[:, j]) -
                           msf.cov_matrix[i, i, j] * spec[i, j]) for j in detector.pixel_indices] for i in range(nord)])

    # saving extracted and unblazed spectrum to file
    fits_file = f'{args.output_dir}/extracted_R0{sim.designR0}.fits'
    hdu_list = fits.HDUList([fits.PrimaryHDU(),
                             fits.BinTableHDU(Table(spec), name='Spectrum'),
                             fits.BinTableHDU(Table(err_n), name='- Errors'),
                             fits.BinTableHDU(Table(err_p), name='+ Errors'),
                             fits.BinTableHDU(Table(lambda_pixel.to(u.Angstrom)), name='Wave Range')])
    hdu_list.writeto(fits_file, output_verify='ignore', overwrite=True)
    logging.info(f'The extracted spectrum with its errors has been saved to {fits_file}.')
    logging.info(f'\nTotal script runtime: {((time.time() - tic) / 60):.2f} min.')
    # ==================================================================================================================
    # OBSERVATION SPECTRUM EXTRACTION ENDS
    # ==================================================================================================================


    # ==================================================================================================================
    # DEBUGGING PLOTS
    # ==================================================================================================================
    spectrum = fits.open(fits_file)

    # plot the spectrum unblazed with the error band:
    fig2, ax2 = plt.subplots(2, 2, figsize=(15, 15), sharex=True)
    axes2 = ax2.ravel()
    for i in range(nord):
        spec_w_merr = np.array(spectrum[1].data[i]) - np.array(spectrum[2].data[i])
        spec_w_perr = np.array(spectrum[1].data[i]) + np.array(spectrum[3].data[i])
        spec_w_merr[spec_w_merr < 0] = 0
        axes2[i].grid()
        axes2[i].fill_between(detector.pixel_indices, spec_w_merr, spec_w_perr, edgecolor='r', facecolor='r',
                              linewidth=0.5)
        axes2[i].plot(detector.pixel_indices, spectrum[1].data[i])
        axes2[i].set_title(f'Order {7 - i}')
    axes2[-1].set_xlabel("Pixel Index")
    axes2[-2].set_xlabel("Pixel Index")
    axes2[0].set_ylabel('Photon Count')
    axes2[2].set_ylabel('Photon Count')
    plt.suptitle(f'{sim.type_spectra} Spectrum with error band')
    plt.tight_layout()
    plt.show()

    # plot the residual between model and observation:
    model = np.genfromtxt('Ar_flux_integrated.csv', delimiter=',')
    for n, i in enumerate(model[::-1]):
        plt.grid()
        model[::-1][n] /= np.max(i)
        normed = spectrum[1].data[n]/np.max(spectrum[1].data[n])
        plt.plot(detector.pixel_indices, model[::-1][n]-normed)
        plt.title(f'Residual between model and observation, Order {spectro.orders[::-1][n]}')
        plt.ylabel('Residual, both normalized to 1 at peak')
        plt.xlabel('Pixel Index')
        plt.show()

        plt.grid()
        plt.plot(detector.pixel_indices, normed, label='Observation')
        plt.plot(detector.pixel_indices, model[::-1][n], '--', label='Model')
        plt.title(f"Side-by-side comparison, Order {spectro.orders[::-1][n]}")
        plt.ylabel('Normalized Flux')
        plt.xlabel('Pixel Index')
        plt.legend()
        plt.show()
    pass
