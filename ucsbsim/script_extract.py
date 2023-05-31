import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime as dt
import logging
import astropy.units as u
from astropy.table import QTable
from astropy.io import fits
from sorcery import dict_of

from mkidpipeline import photontable as pt
from spectrograph import GratingSetup, SpectrographSetup
from detector import MKIDDetector
from engine import Engine, quick_plot
from synphot.models import Box1D
from synphot import SourceSpectrum

"""
Extraction of an observation spectrum using the MSF products. The steps are:
-
"""

if __name__ == '__main__':
    tic = time.time()  # recording start time for script

    # ========================================================================
    # Constants which may or may not be needed in the future:
    # ========================================================================

    # ========================================================================
    # Assign spectrograph properties below if different from defaults. Check the module spectrograph for syntax.
    # ========================================================================
    # R0 = 8, etc.

    # ========================================================================
    # Spectrum extraction settings:
    # ========================================================================
    type_of_spectra = 'phoenix'

    # ========================================================================
    # Spectrum extraction begins:
    # ========================================================================
    now = dt.now()
    logging.basicConfig(filename=f'output_files/{type_of_spectra}/logging/extract_{now.strftime("%Y%m%d_%H%M%S")}.log',
                        format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info(f"The extraction of a {type_of_spectra} spectrum is recorded."
                 f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    # setting up engine and spectrograph, obtaining various needed properties:
    eng = Engine(SpectrographSetup(GratingSetup(), MKIDDetector()))
    R0 = eng.spectrograph.detector.design_R0
    nord = eng.spectrograph.nord
    npix = eng.spectrograph.detector.n_pixels
    resid_map = np.arange(npix, dtype=int) * 10 + 100  # TODO replace once known
    lambda_pixel = eng.spectrograph.pixel_center_wavelengths().to(u.nm)[::-1]

    # obtain the pixel-space blaze function:
    u.photlam = u.photon / u.s / u.cm ** 2 / u.AA  # photon flux per wavelength
    wave = np.linspace(eng.spectrograph.minimum_wave.value - 100, eng.spectrograph.detector.waveR0.value + 100,
                       10000) * u.nm
    blaze = eng.pixelspace_blaze(wave, SourceSpectrum(Box1D, amplitude=1*u.photlam, x_0=(wave[-1]+wave[0])/2,
                                                       width=(wave[-1]-wave[0])))[::-1]
    blaze /= np.max(blaze)

    # open relevant wavelength, calibration bins and covariance matrices
    bin_name = f'output_files/calibration/msf_bins_R0{R0}.fits'
    bins = fits.getdata(bin_name)
    logging.info(f'The bin edges for extraction were obtained from {bin_name}.')
    cov = np.zeros([nord, nord, npix])
    logging.info(f'The covariance matrices were obtained from:')
    for i in range(nord):
        name = f'output_files/calibration/msf_cov{i}_R0{R0}.fits'
        for j in range(nord):
            cov[j, i, :] = fits.getdata(name)[f'{eng.spectrograph.orders[j]}']
        logging.info(f'{name}')

    # opening the h5 file containing the photon table
    table = f'output_files/{type_of_spectra}/table_R0{R0}.h5'
    photons_pixel = eng.open_table(resid_map, table)
    logging.info(f'The photon table was obtained from {table}.')

    spec = np.zeros([nord, npix])
    for j in range(npix):
        spec[:, j], _ = np.histogram(photons_pixel[j], bins=bins[j])  # binning photons by MSF bins edges

    # to plot covariance as errors, must sum the counts "added" from other orders as well as "stolen" by other orders
    # v giving order, > receiving order [g_idx, r_idx, pixel]
    #      9   8   7   6   5
    # 9 [  1   #   #   #   #  ]  < multiply counts*cov in Order 9 to add to other orders
    # 8 [  #   1   #   #   #  ]
    # 7 [  #   #   1   #   #  ]
    # 6 [  #   #   #   1   #  ]
    # 5 [  #   #   #   #   1  ]
    #      ^ multiply counts*cov in other orders to add to Order 9

    err_p = np.array([[
        int(np.sum(cov[:, i, j] * spec[:, j])) - cov[i, i, j] * spec[i, j] for j in range(npix)] for i in range(nord)])
    err_n = np.array([[
        int(np.sum(cov[i, :, j] * spec[:, j]) - cov[i, i, j] * spec[i, j]) for j in range(npix)] for i in range(nord)])

    meta = dict_of(R0, eng.spectrograph.detector.waveR0.value, npix, nord, type_of_spectra)

    # saving extracted and unblazed spectrum to file
    fits = QTable(list(spec/blaze), names=[f'{i}' for i in eng.spectrograph.orders[::-1]],
                       meta={'name': 'Extracted Spectrum', **meta})
    fits_file = f'output_files/{type_of_spectra}/spectrum_R0{R0}.fits'
    fits.write(fits_file, format='fits', overwrite=True)
    logging.info(f'The extracted spectrum has been saved to {fits_file}.')

    # saving errors to fits file
    errors_p = QTable(list(err_p/blaze), names=[f'{i}' for i in eng.spectrograph.orders[::-1]],
                       meta={'name': 'Positive Errors on Spectrum', **meta})
    errp_file = f'output_files/{type_of_spectra}/errp_R0{R0}.fits'
    errors_p.write(errp_file, format='fits', overwrite=True)
    errors_n = QTable(list(err_n / blaze), names=[f'{i}' for i in eng.spectrograph.orders[::-1]],
                      meta={'name': 'Negative Errors on Spectrum', **meta})
    errn_file = f'output_files/{type_of_spectra}/errn_R0{R0}.fits'
    errors_n.write(errn_file, format='fits', overwrite=True)
    logging.info(f'The error bands on the spectrum have been saved to {errn_file} and {errp_file}.')
    logging.info(f'\nTotal script runtime: {((time.time() - tic) / 60):.2f} min.')
    # ========================================================================
    # Spectrum extraction ends
    # ========================================================================


    # ========================================================================
    # Plotting final spectrum with error band:
    # ========================================================================

    plt.grid()
    for i in range(nord):
        plt.fill_between(lambda_pixel[i].value, (spec[i] - err_n[i]) / blaze[i], (spec[i] + err_p[i]) / blaze[i],
                         alpha=0.5, edgecolor='orange', facecolor='orange', linewidth=0.5)
        plt.plot(lambda_pixel[i], spec[i] / blaze[i])
    plt.title(f"Unblazed & extracted {type_of_spectra} spectrum")
    plt.ylabel('Total Photons')
    plt.xlabel('Wavelength (nm)')
    plt.tight_layout()
    plt.show()
