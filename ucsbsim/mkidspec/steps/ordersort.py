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

from ucsbsim.mkidspec.spectrograph import GratingSetup, SpectrographSetup
from ucsbsim.mkidspec.detector import MKIDDetector, wave_to_phase
import ucsbsim.mkidspec.engine as engine
from ucsbsim.mkidspec.plotting import quick_plot
from synphot.models import BlackBodyNorm1D, ConstFlux1D
from synphot import SourceSpectrum
from ucsbsim.mkidspec.msf import MKIDSpreadFunction
from mkidpipeline.photontable import Photontable

"""
Application of the virtual pixel boundaries and errors on an spectrum using the MSF products. The steps are:
-Open the MSF products: order bin edges and covariance matrices.
-Open the observation/emission photon table and bin for virtual pixels/orders.
-Calculate the errors on each point by multiplying the covariance matrices through the spectrum.
-Save counts, errors, and estimate of wave range to FITS.
-Show final spectrum as plot.
"""


def ordersort(
        table: Photontable,
        filename: str,
        msf_file: str,
        outdir: str,
        plot: bool,
        resid_map=np.arange(2048, dtype=int) * 10 + 100,
):
    photons_pixel = engine.sorted_table(table=table, resid_map=resid_map)

    msf = MKIDSpreadFunction(filename=msf_file)
    sim = msf.sim_settings.item()
    logging.info(f'Obtained MKID Spread Function from {msf_file}.')

    # INSTANTIATE SPECTROGRAPH & DETECTOR:
    detector = MKIDDetector(
        n_pix=sim.npix,
        pixel_size=sim.pixelsize,
        design_R0=sim.designR0,
        l0=sim.l0,
        resid_map=resid_map
    )
    grating = GratingSetup(alpha=sim.alpha, delta=sim.delta, beta_center=sim.beta, groove_length=sim.groove_length)
    spectro = SpectrographSetup(
        order_range=sim.order_range,
        final_wave=sim.l0,
        pixels_per_res_elem=sim.pixels_per_res_elem,
        focal_length=sim.focallength,
        grating=grating,
        detector=detector
    )

    nord = spectro.nord
    lambda_pixel = spectro.pixel_wavelengths().to(u.nm)[::-1]

    spec = np.zeros([nord, sim.npix])
    for j in detector.pixel_indices:  # binning photons by MSF bins edges
        spec[:, j], _ = np.histogram(photons_pixel[j], bins=msf.bin_edges[:, j])

    err_p = np.array([[int(np.sum(msf.cov_matrix[:, i, j] * spec[:, j])) -
                       msf.cov_matrix[i, i, j] * spec[i, j] for j in detector.pixel_indices] for i in range(nord)])
    err_n = np.array([[int(np.sum(msf.cov_matrix[i, :, j] * spec[:, j]) -
                           msf.cov_matrix[i, i, j] * spec[i, j]) for j in detector.pixel_indices] for i in range(nord)])

    # saving extracted and unblazed spectrum to file
    fits_file = f'{outdir}/{filename}.fits'
    hdu_list = fits.HDUList([fits.PrimaryHDU(),
                             fits.BinTableHDU(Table(spec), name='Spectrum'),
                             fits.BinTableHDU(Table(err_n), name='- Errors'),
                             fits.BinTableHDU(Table(err_p), name='+ Errors'),
                             fits.BinTableHDU(Table(lambda_pixel.to(u.Angstrom)), name='Wave Range')])
    hdu_list.writeto(fits_file, output_verify='ignore', overwrite=True)
    logging.info(f'The extracted spectrum with its errors has been saved to {fits_file}.')

    if plot:
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
        plt.suptitle(f'Sorted spectrum with error band')
        plt.tight_layout()
        plt.show()

        # if filename == 'emission':
        #     # plot the residual between model and observation:
        #     model = np.genfromtxt('../../Ar_flux_integrated.csv', delimiter=',')
        #     for n, i in enumerate(model[::-1]):
        #         plt.grid()
        #         model[::-1][n] /= np.max(i)
        #         normed = spectrum[1].data[n] / np.max(spectrum[1].data[n])
        #         plt.plot(detector.pixel_indices, model[::-1][n] - normed)
        #         plt.title(f'Residual between model and observation, Order {spectro.orders[::-1][n]}')
        #         plt.ylabel('Residual, both normalized to 1 at peak')
        #         plt.xlabel('Pixel Index')
        #         plt.show()
        # 
        #         plt.grid()
        #         plt.plot(detector.pixel_indices, normed, label='Observation')
        #         plt.plot(detector.pixel_indices, model[::-1][n], '--', label='Model')
        #         plt.title(f"Side-by-side comparison, Order {spectro.orders[::-1][n]}")
        #         plt.ylabel('Normalized Flux')
        #         plt.xlabel('Pixel Index')
        #         plt.legend()
        #         plt.show()

    return fits_file


if __name__ == '__main__':
    tic = time.time()  # recording start time for script

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
    parser.add_argument('outdir', metavar='OUTPUT_DIRECTORY',
                        help='Directory for the output files (str).')
    parser.add_argument('msf_file',
                        metavar='MKID_SPREAD_FUNCTION_FILE',
                        help='Directory/name of the MKID Spread Function file (str).')
    parser.add_argument('obstable',
                        metavar='OBSERVATION_PHOTON_TABLE',
                        help='Directory/name of the observation spectrum photon table (str).')
    parser.add_argument('--plot', action='store_true', default=False, type=bool, help='If passed, plots will be shown.')

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
    logging.basicConfig(filename=f'{args.output_dir}/ordersort_{now.strftime("%Y%m%d_%H%M%S")}.log',
                        format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info(f"The extraction of an observed spectrum is recorded."
                 f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    obs_table = Photontable(file_name=args.obstable)
    sim = obs_table.query_header('sim_settings')
    spectra = sim.spectra_type

    # ==================================================================================================================
    # OBSERVATION SPECTRUM EXTRACTION STARTS
    # ==================================================================================================================
    ordersort(
        table=obs_table,
        filename=spectra,
        msf_file=args.msf_file,
        outdir=args.outdir,
        plot=args.plot
    )

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

    logging.info(f'\nTotal script runtime: {((time.time() - tic) / 60):.2f} min.')
    # ==================================================================================================================
    # OBSERVATION SPECTRUM EXTRACTION ENDS
    # ==================================================================================================================

