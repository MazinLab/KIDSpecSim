import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import sys
import os
import logging
import time
from datetime import datetime as dt
import argparse
from astropy import units as u
import scipy.interpolate as interp

from ucsbsim.spectrograph import SpectrographSetup
from ucsbsim.spectra import EmissionModel

sys.path.insert(1, '/home/kimc/pycharm/PyReduce/pyreduce')
from wavelength_calibration import WavelengthCalibration, WavelengthCalibrationInitialize, LineList

"""
Emission line wavelength calibration. The steps are:
-Load the emission line photon table
-
"""  # TODO

if __name__ == '__main__':
    tic = time.perf_counter()  # recording start time for script

    # ==================================================================================================================
    # CONSTANTS
    # ==================================================================================================================

    # ==================================================================================================================
    # PARSE ARGUMENTS
    # ==================================================================================================================
    arg_desc = '''
               Wavelength calibration from an emission line source spectrum.
               --------------------------------------------------------------
               This program loads the emission line photon table and allows user to
               recover a 2D wavecal solution.
               '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)

    # required wavecal args:
    parser.add_argument('output_dir',
                        metavar='OUTPUT_DIRECTORY',
                        help='Directory for the output files (str).')
    parser.add_argument('obs_file',
                        metavar='FITS_FILE',
                        help='Directory/name of the FITS file to be used for wavecal.')

    # optional wavecal args:
    parser.add_argument('-el', '--elem',
                        metavar='ELEMENT',
                        default='ne',
                        help="Calibration lamp element in use, i.e., 'ne' for Neon. (str)")
    parser.add_argument('-ord', '--orders',
                        metavar='ORDERS',
                        default=[7, 6, 5, 4],
                        help="Orders to be used. Useful if you only want to fit 1 order at a time. (list)")
    parser.add_argument('-deg', '--degree',
                        metavar='POLYNOMIAL_DEGREE',
                        default=4,
                        help="Polynomial degree to use in wavecal. (int)")
    parser.add_argument('-it', '--iterations',
                        metavar='ITERATIONS',
                        default=5,
                        help="Number of iterations to loop through for identifying and discarding lines. (int)")
    parser.add_argument('-mf', '--manualfit',
                        action='store_true',
                        default=False,
                        help="If passed, indicates user should click plot to align observation and linelist.")
    parser.add_argument('-rm', '--resid_max',
                        metavar='RESIDUAL_MAX',
                        default=85e3,
                        help="Maximum residual allowed between fit wavelength and atlas in m/s. (float)")
    parser.add_argument('-w', '--width',
                        metavar='PEAK_WIDTH',
                        default=3,
                        help="Width in pixels when searching for peaks in observation. (int)")
    parser.add_argument('-sw', '--shift_window',
                        metavar='ORDER_SHIFT_WINDOW',
                        default=0.05,
                        help="Fraction of columns to use in the alignment of individual orders, 0 to disable. (float)")
    parser.add_argument('-minw', '--minwave',
                        metavar='MINIMUM_WAVELENGTH_NM',
                        default=400,
                        help='The minimum wavelength of the spectrometer in nm. (float)')
    parser.add_argument('-maxw', '--maxwave',
                        metavar='MAXIMUM_WAVELENGTH_NM',
                        default=800,
                        help='The maximum wavelength of the spectrometer in nm. (float)')

    # extract arguments
    args = parser.parse_args()

    # ==================================================================================================================
    # CHECK AND/OR CREATE DIRECTORIES
    # ==================================================================================================================
    os.makedirs(f'{args.output_dir}/logging', exist_ok=True)

    # ==================================================================================================================
    # START LOGGING TO FILE
    # ==================================================================================================================
    now = dt.now()
    logging.basicConfig(filename=f'{args.output_dir}/logging/wavecal_{now.strftime("%Y%m%d_%H%M%S")}.log',
                        format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info("The process of wavelength calibration from an emission line spectrum is recorded."
                 f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    # ==================================================================================================================
    # OPEN FITS FILE AND LINE LIST
    # ==================================================================================================================
    # open observed spectrum and get wave/flux
    spectrum = fits.open(f'/home/kimc/pycharm/KIDSpecSim/ucsbsim/{args.obs_file}')
    obs_flux = np.array([np.array(spectrum[1].data[n]) for n, i in enumerate(args.orders)])
    obs_wave = np.array([np.array(spectrum[4].data[n]) for n, i in enumerate(args.orders)])

    # open entire line list:
    file = pd.read_csv(f'/home/kimc/pycharm/KIDSpecSim/ucsbsim/{args.elem}_linelist.csv', delimiter=',')
    line_wave = np.array([float(i[2:-1]) * 10 for i in file['obs_wl_air(nm)']])  # nm to Angstrom
    line_flux = np.array([float(i[2:-1]) for i in file['intens']])

    # this is the theoretical reference line spectrum:
    theo_lines = EmissionModel(f'{args.elem}_linelist.csv', minwave=args.minwave, maxwave=args.maxwave ,target_R=1000000)
    col1 = fits.Column(name='wave', format='E', array=theo_lines.waveset.value)
    col2 = fits.Column(name='spec', format='E', array=theo_lines(theo_lines.waveset).value)
    cols = fits.ColDefs([col1, col2])
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu_list = fits.HDUList([fits.PrimaryHDU(), hdu])
    hdu_list.writeto(f'/home/kimc/pycharm/PyReduce/pyreduce/wavecal/atlas/{args.elem}.fits',
                     overwrite=True)

    f = open(f'/home/kimc/pycharm/PyReduce/pyreduce/wavecal/atlas/{args.elem}_list.txt', 'w+')
    for i in range(len(line_wave)):
        f.write(f'{line_wave[i]} {args.elem.capitalize()}I\n')
    f.close()

    # ==================================================================================================================
    # WAVELENGTH CALIBRATION STARTS
    # ==================================================================================================================
    # linelist = LineList()
    # for n, i in enumerate(args.orders):
    #     # takes each order and conducts MCMC line matching
    #     module_init = WavelengthCalibrationInitialize(
    #         degree=args.degree,  # polynomial degree of the wavelength fit
    #         plot=True,
    #         plot_title=f'Deg. {args.degree} Poly./{10} Walkers/{50000} Steps/{args.resid_max} Max Resid./Element: {args.elem}',
    #         wave_delta=10,  # wavelength uncertainty on the initial guess in Angstrom
    #         nwalkers=100,  # number of walkers in the MCMC
    #         steps=50000,  # number of steps in the MCMC
    #         resid_delta=args.resid_max,
    #         # residual uncertainty allowed when matching observation with known lines (diff/wave*c)
    #         cutoff=10,  # minimum value in the spectrum to be considered a spectral line,
    #         # if the value is above (or equal 1) it defines the percentile of the spectrum
    #         smoothing=5,  # gaussian smoothing on wavecal spectrum before the MCMC in pixel scale, disable with 0
    #         element=args.elem,
    #         medium="vac"
    #     )
    #     ll = module_init.execute(np.array([obs_flux[n]]), [(obs_wave[n][0], obs_wave[n][-1])])
    #     ll = ll.data
    #     for li in ll:
    #         linelist.add_line(li['wlc'], n, li['posc'], li['width'], li['height'], li['flag'])

    # on an order by order basis, cut away lines that are too close together or too dim
    pixel_interp = [interp.interp1d(obs_wave[n], range(2048), bounds_error=False, fill_value=0) for n, i in
                    enumerate(args.orders)]
    posc = np.array([pixel_interp[n](line_wave) for n, i in enumerate(args.orders)])  # Pixel Position (before fit)

    # remove 10% of the least intense lines
    sort_idx = np.argsort(line_flux)
    percent = int(0.1 * len(sort_idx))
    sort_idx = sort_idx[percent:]
    cut_wave = line_wave[sort_idx]
    cut_flux = line_flux[sort_idx]

    # re-sort back to increasing wavelength
    sort_idx = np.argsort(cut_wave)
    cut_wave = cut_wave[sort_idx]
    cut_flux = cut_flux[sort_idx]

    # keep only the most intense lines within a certain range:
    min_dl = np.min(np.diff(obs_wave[0]))  # the smallest extent of a pixel [Ang]
    start_range = 0
    new_line_idx = []
    for n, l in enumerate(cut_wave[:-1]):
        diff = cut_wave[n + 1] - cut_wave[start_range]
        if diff > min_dl:
            stop_range = n + 1
            copy_flux = np.copy(cut_flux)
            copy_flux[:start_range] = 0
            copy_flux[stop_range:] = 0
            new_line_idx.append(np.argmax(copy_flux))
            start_range = stop_range

    new_line_wave = cut_wave[new_line_idx]
    new_line_flux = cut_flux[new_line_idx]

    # create PyReduce LineList object
    pixel_interp = [interp.interp1d(obs_wave[n], range(2048), bounds_error=False, fill_value=0) for n, i in
                    enumerate(args.orders)]
    posc = np.array([pixel_interp[n](new_line_wave) for n, i in enumerate(args.orders)])  # Pixel Position (before fit)
    linelist = LineList()
    for n, i in enumerate(args.orders):
        for m, (w, p, h) in enumerate(zip(new_line_wave, posc[n], new_line_flux)):
            if posc[n][m]:
                linelist.add_line(w, n, p, 2, h, True)

    # load wavecal module
    module = WavelengthCalibration(
        threshold=args.resid_max,  # Residual threshold in m/s above which to remove lines (diff/wave*c)
        degree=(args.degree, 2),  # polynomial degree of the wavelength fit in (pixel, order) direction
        iterations=args.iterations,  # Number of iterations in the remove residuals, auto id, loop
        dimensionality="2D",  # Whether to use 1d or 2d fit
        shift_window=args.shift_window,
        # Fraction of columns to use in the alignment of individual orders. 0 to disable
        manual=args.manualfit,  # Whether to manually align the reference instead of using cross correlation
        lfc_peak_width=args.width,  # Laser Frequency Peak width (for scipy.signal.find_peaks)
        element=args.elem,
        medium="vac",  # use vac if the linelist provided is in 'air' already, air will convert
        plot=True,
        plot_title=f'{args.degree}D Poly./{args.resid_max:.0e} Max Resid./Element: {args.elem.capitalize()}'
    )
    # combines the orders and returns 2D solution
    wave_result, coef, residuals, fit_lines = module.execute(obs_flux, linelist)

    rms = [np.sqrt(np.mean(residuals[fit_lines['order'] == n] ** 2)) for n, i in enumerate(args.orders)]
    print(f'RMS (km/s): {[round(i / 1000, 2) for i in rms]}, Avg: {np.sqrt(np.mean(residuals ** 2)) / 1000:.2f}')
    print(f"{np.sum(fit_lines['flag'])} used out of {len(fit_lines)}")

    np.savez(f'{args.output_dir}/wavecal.npz', wave_result=wave_result, coef=coef, rms=rms, linelist=linelist,
             orders=args.orders)
    logging.info(f'\nSaved wavecal solution and linelist to {args.output_dir}/wavecal.npz.')
    logging.info(f'\nTotal script runtime: {((time.perf_counter() - tic) / 60):.2f} min.')
    # ==================================================================================================================
    # MSF EXTRACTION ENDS
    # ==================================================================================================================

    # ==================================================================================================================
    # DEBUGGING PLOTS
    # ==================================================================================================================
    new_file = np.load(f'{args.output_dir}/wavecal.npz')

    for n, i in enumerate(args.orders):
        plt.grid()
        plt.ylim(bottom=0)
        use_idx = np.logical_and(fit_lines['wll'] > obs_wave[n][0], fit_lines['wll'] < obs_wave[n][-1])
        lines_use = fit_lines[use_idx]
        max_flux = np.max(lines_use['height'])
        for m, w in enumerate(lines_use):
            if not w['flag']:
                plt.axvline(w['wll'], color='black', ymin=0, ymax=w['height'] / max_flux)
            else:
                plt.axvline(w['wll'], color='red', alpha=0.5, ymin=0, ymax=w['height'] / max_flux)
        plt.plot(obs_wave[n], obs_flux[n] / np.max(obs_flux[n]), 'gray', label='Guess')
        plt.plot(new_file['wave_result'][n], obs_flux[n] / np.max(obs_flux[n]), 'blue', label=f'Fit Result')
        plt.title(f"Wavecal Order {i}. Used Lines in Red, Unused in Black\n"
                  f'Order {args.degree} Poly./Element: {args.elem.capitalize()}')
        plt.ylabel('Photon Count')
        plt.xlabel(r'Wavelength ($\AA$)')
        plt.legend()
        plt.show()
    pass
