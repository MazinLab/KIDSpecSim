import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import astropy.units as u
import argparse
from astropy.io import fits
from astropy.table import Table
import os

"""
Bringing it all together. This is the stage where we load and bring together:
- The observation spectrum separated into virtual pixels and the errors on each
- The wavecal solution from the emission spectrum
to recover a final spectrum.
This script may also:
- fit an empiric blaze function to be divided out
"""


def extract(
        obs_fits,
        wavecal_file,
        plot
):
    wavecal_file = np.load(f'{wavecal_file}')
    wavecal = wavecal_file['wave_result']
    orders = wavecal_file['orders']

    spectrum = fits.open(f'/home/kimc/pycharm/KIDSpecSim/ucsbsim/{obs_fits}', mode='update')
    obs_flux = np.array([np.array(spectrum[1].data[n]) for n, i in enumerate(orders)])
    guess_wave = np.array([np.array(spectrum[4].data[n]) for n, i in enumerate(orders)])
    hdu_list = fits.HDUList([spectrum[0],
                             spectrum[1],
                             spectrum[2],
                             spectrum[3],
                             fits.BinTableHDU(Table(wavecal), name='Wavecal')])
    hdu_list.writeto(obs_fits, output_verify='ignore', overwrite=True)
    logging.info(f'The FITS file has been updated with the wavecal: {obs_fits}.')

    if plot:
        # comparing the wavecal solution to the simulation wavelengths
        plt.grid()
        for n, i in enumerate(orders):
            plt.plot(guess_wave[n] / 10, obs_flux[n], 'r')
            plt.plot(wavecal[n] / 10, obs_flux[n], 'b', alpha=0.5)
        plt.title('Comparison between simulation (red) and wavecal (blue)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Photon Count')
        plt.show()


if __name__ == '__main__':

    arg_desc = '''
    Extract a spectrum using the uncalibrated spectrum and wavecal.
    --------------------------------------------------------------
    This program loads the observation spectrum on the phase grid and applies the wavecal solution to
    extract the spectrum. It will compare the simulation input to the final product if applicable.
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)

    # required extraction args:
    parser.add_argument('obs_fits',
                        metavar='FITS_FILE',
                        help='Directory/name of the FITS file containing the observation (str).')
    parser.add_argument('wavecal_file',
                        metavar='WAVECAL_FILE',
                        help='Directory/name of the wavelength calibration .npz file (str).')
    parser.add_argument('--plot', action='store_true', default=False, type=bool, help='If passed, plots will be shown.')

    # get arguments
    args = parser.parse_args()

    extract(obs_fits=args.obs_fits, wavecal_file=args.wavecal_file, plot=args.plot)


