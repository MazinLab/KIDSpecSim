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

from spectrograph import GratingSetup, SpectrographSetup
from detector import MKIDDetector
from engine import Engine
from plotting import quick_plot
from synphot.models import Box1D
from synphot import SourceSpectrum
from msf import MKIDSpreadFunction
from mkidpipeline.photontable import Photontable

"""
Extraction of an observation spectrum using the MSF products. The steps are:
-Open the calibration products: order bin edges and covariance matrices.
-Open the observation photon table and bin for orders.
-Calculate the errors on each point by multiplying the covariance matrices through the spectrum.
-Divide out the blaze function from the resulting spectrum and +/- errors and save to FITS.
-Show final spectrum as plot.

Potentially make spectrograph properties a file that is automatically imported from previous simulations?
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
    Extract the observation spectrum using the MKID Spread Function.
    --------------------------------------------------------------
    This program loads the observation photon table and use the MSF bins and covariance matrix to 
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
    # START LOGGING TO FILE
    # ==================================================================================================================
    now = dt.now()
    logging.basicConfig(filename=f'{args.output_dir}/extract_{now.strftime("%Y%m%d_%H%M%S")}.log',
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

    detector = MKIDDetector(sim.npix, sim.pixelsize, sim.designR0, sim.l0, R0s, resid_map)
    grating = GratingSetup(sim.l0, sim.m0, sim.m_max, sim.pixelsize, sim.npix, sim.focallength, sim.nolittrow)
    spectro = SpectrographSetup(grating, detector, sim.pixels_per_res_elem)
    eng = Engine(spectro)
    nord = spectro.nord
    lambda_pixel = spectro.pixel_wavelengths().to(u.nm)[::-1]  # flip order, extracts in ascending wavelengths

    # ==================================================================================================================
    # OBSERVATION SPECTRUM EXTRACTION STARTS
    # ==================================================================================================================
    # obtain the pixel-space blaze function:
    wave = np.linspace(sim.minwave.value - 100, sim.maxwave.value + 100, 10000) * u.nm
    blaze_no_int, _, _ = eng.blaze(wave, SourceSpectrum(Box1D, amplitude=1*u.photlam, x_0=(wave[-1]+wave[0])/2,
                                                        width=(wave[-1]-wave[0])))
    pix_leftedge = spectro.pixel_wavelengths(edge='left').to(u.nm).value
    blaze_shape = [eng.lambda_to_pixel_space(wave, blaze_no_int[i], pix_leftedge[i]) for i in range(nord)][::-1]
    blaze_shape /= np.max(blaze_shape)  # normalize max to 1
    blaze_shape[blaze_shape == 0] = 1  # prevent divide by 0 or very small num. issue

    spec = np.zeros([nord, sim.npix])
    for j in range(sim.npix):
        spec[:, j], _ = np.histogram(photons_pixel[j], bins=msf.bin_edges[:, j])  # binning photons by MSF bins edges

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
                       msf.cov_matrix[i, i, j] * spec[i, j] for j in range(sim.npix)] for i in range(nord)])
    err_n = np.array([[int(np.sum(msf.cov_matrix[i, :, j] * spec[:, j]) -
                           msf.cov_matrix[i, i, j] * spec[i, j]) for j in range(sim.npix)] for i in range(nord)])

    # saving extracted and unblazed spectrum to file TODO save to 1 file w/ errors
    fits_file = f'{args.output_dir}/{sim.type_spectra}_extracted_R0{sim.designR0}.fits'
    hdu_list = fits.HDUList([fits.PrimaryHDU(),
                             fits.BinTableHDU(Table(spec/blaze_shape), name='Spectrum'),
                             fits.BinTableHDU(Table(err_n/blaze_shape), name='- Errors'),
                             fits.BinTableHDU(Table(err_p/blaze_shape), name='+ Errors')])
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

    plt.grid()
    for i in range(nord):
        plt.fill_between(lambda_pixel[i].value, np.array(spectrum[1].data[i]) - np.array(spectrum[2].data[i]),
                         np.array(spectrum[1].data[i]) - np.array(spectrum[3].data[i]),
                         alpha=0.5, edgecolor='orange', facecolor='orange', linewidth=0.5)
        plt.plot(lambda_pixel[i], spectrum[1].data[i])
    plt.title(f"Unblazed & extracted {sim.type_spectra} spectrum")
    plt.ylabel('Total Photons')
    plt.xlabel('Wavelength (nm)')
    plt.tight_layout()
    plt.show()
