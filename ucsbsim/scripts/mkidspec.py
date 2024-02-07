import numpy as np
from numpy.polynomial.legendre import Legendre
import matplotlib.pyplot as plt
import scipy.signal
import scipy
import scipy.interpolate as interp
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

import astropy.units as u
import time
from datetime import datetime as dt
import argparse
import logging
import os
from lmfit import Parameters, minimize

from mkidpipeline.photontable import Photontable
from ucsbsim.mkidspec.steps.fitmsf import fitmsf
from ucsbsim.mkidspec.steps.ordersort import ordersort
from ucsbsim.mkidspec.steps.wavecal import wavecal
from ucsbsim.mkidspec.steps.extract import extract
from ucsbsim.mkidspec.simsettings import SpecSimSettings


def parse():
    # read in command line arguments
    parser = argparse.ArgumentParser(description='MKID Spectrograph Data Reduction')

    # optional script args:
    parser.add_argument('--outdir', default='testfiles/outdir', type=str, help='Directory for the output files.')
    parser.add_argument('--plot', action='store_true', default=False, type=bool,
                        help='If passed, indicates that intermediate plots will be shown.')
    parser.add_argument('--resid_map', default=np.arange(2048, dtype=int) * 10 + 100,
                        help='Resonator IDs for the array.')

    # optional spectrograph simulation args which are necessary for msf guess:
    parser.add_argument('--minw', default=400, type=float, help='The minimum wavelength of the spectrograph in nm.')
    parser.add_argument('--maxw', default=800, type=float, help='The maximum wavelength of the spectrograph in nm.')
    parser.add_argument('--npix', default=2048, type=int, help='The number of pixels in the MKID detector.')
    parser.add_argument('--pixsize', default=20,
                        help='The length of the MKID pixel in the dispersion direction in um.')
    parser.add_argument('-R0', default=15, type=float, help='The spectral resolution at the maximum wavelength.')
    parser.add_argument('-l0', default=800,
                        help="The longest wavelength in nm. Can be float or 'same' to be equal to 'maxwave' arg.")
    parser.add_argument('--osamp', default=10,
                        help='The number of samples to use for the smallest pixel dlambda during convolution.')
    parser.add_argument('--nsig', default=3, type=float,
                        help='The number of sigma to use for Gaussian during convolution.')
    parser.add_argument('--alpha', default=28.3, type=float, help='Angle of incidence on the grating in degrees.')
    parser.add_argument('--beta', default=34.7, type=float, help='Reflectance angle at the central pixel in degrees. ')
    parser.add_argument('--delta', default=63, type=float, help='Blaze angle in degrees.')
    parser.add_argument('-d', '--groove_length', default=((1 / 316) * u.mm).to(u.nm).value, type=float,
                        help='The groove length d, or distance between slits, of the grating in nm.')
    parser.add_argument('--m0', default=4, type=int, help='The initial order, at the longer wavelength end.')
    parser.add_argument('--m_max', default=8, type=int, help='The final order, at the shorter wavelength end.')
    parser.add_argument('-ppre', '--pixels_per_res_elem', default=2.5, type=float,
                        help='Number of pixels per spectral resolution element for the spectrograph.')
    parser.add_argument('--focallength', default=300, type=float, help='The focal length of the detector in mm.')

    # optional MSF args:
    parser.add_argument('--msf', default='testfiles/flat.h5', type=str,
                        help='Directory/name of the flat/blackbody spectrum photon table .h5 file OR'
                             'Directory/name of the complete MKID Spread Function .npz file.'
                             'Pass "False" to disable this step.')
    parser.add_argument('--bin_range', default=(-1, 0), type=tuple,
                        help='Start and stop of range for phase histogram.')

    # optional wavecal args:
    parser.add_argument('--wavecal', default='testfiles/emission.h5', type=str,
                        help='Directory/name of the emission lamp spectrum photon table .h5 file OR'
                             'Directory/name of the order-sorted emission lamp spectrum FITS file OR'
                             'Directory/name of the complete wavelength calibration solution .npz file.'
                             'Pass "False" to disable this step.')
    parser.add_argument('--elem', default='hgar', type=str,
                        help="Emission lamp element in use, i.e., 'hgar' for Mercury-Argon.")
    parser.add_argument('--orders', default=[8, 7, 6, 5, 4], type=list,  # TODO which orders are actually incident
                        help="Orders to be used. Useful if you only want to fit 1 order at a time.")
    parser.add_argument('--degree', default=4, type=int, help="Polynomial degree to use in wavecal.")
    parser.add_argument('--iters', default=5, type=int,
                        help="Number of iterations to loop through for identifying and discarding lines.")
    parser.add_argument('--manual_fit', action='store_true', default=False, type=bool,
                        help="If passed, indicates user should click plot to align observation and linelist.")
    parser.add_argument('--resid_max', default=85e3, type=float,
                        help="Maximum residual allowed between fit wavelength and atlas in m/s. (float)")
    parser.add_argument('--width', default=3, type=int, help="Width in pixels when searching for matching peaks.")
    parser.add_argument('--shift_window', default=0.05, type=float,
                        help="Fraction of columns to use in the alignment of individual orders, 0 to disable.")
    parser.add_argument('--dim', default='1D', type=str,
                        help="Return a '1D' (pixel direction) or '2D' (pixel+order directions) fitting solution.")

    # optional observation args:
    parser.add_argument('--extract', default='testfiles/phoenix.h5', type=str,
                        help='Directory/name of the on-sky observation spectrum photon table .h5 file OR'
                             'Directory/name of the order-sorted observation spectrum FITS file.'
                             'Pass "False" to disable this step.')

    return parser.parse_args()


if __name__ == "__main__":

    now = dt.now()

    args = parse()

    os.makedirs(name=f'{args.outdir}', exist_ok=True)

    logging.basicConfig(filename=f'{args.outdir}/{now.strftime("%Y%m%d_%H%M%S")}.log',
                        format='%(levelname)s:%(message)s', level=logging.INFO)

    # packing up the simulation values into a class
    sim = SpecSimSettings(
        minwave_nm=args.minw,
        maxwave_nm=args.maxw,
        npix=args.npix,
        pixelsize_um=args.pixelsize,
        designR0=args.R0,
        l0_nm=args.l0,
        alpha_deg=args.alpha,
        delta_deg=args.delta,
        beta_deg=args.beta,
        groove_length_nm=args.groove_length,
        m0=args.m0,
        m_max=args.mmax,
        pixels_per_res_elem=args.pixels_per_res_elem,
        focallength_mm=args.focallength
    )

    steps = []  # list to append steps in use

    # MSF
    if args.msf.lower().endswith('.h5'):  # the MSF has yet to be fit
        msf_table = Photontable(file_name=args.msf)
        steps.append('msf')
    elif args.msf.lower().endswith('.npz'):  # the MSF file already exists
        msf_file = args.msf

    # wavecal
    if args.wavecal.lower().endswith('.h5'):  # the table is not order-sorted and wavecal has yet to be done
        wavecal_table = Photontable(file_name=args.wavecal)
        steps.append('wt_sort')
        steps.append('wavecal')
    elif args.wavecal.lower().endswith('.fits'):  # the wavecal has yet to be done
        wavecal_fits = args.wavecal
        steps.append('wavecal')
    elif args.wavecal.lower().endswith('.npz'):  # the wavecal file already exists
        wavecal_file = args.wavecal

    # extract
    if args.extract.lower().endswith('.h5'):  # the table is not order-sorted or extracted
        obs_table = Photontable(file_name=args.extract)
        steps.append('ot_sort')
        steps.append('extract')
    elif args.extract.lower().endswith('.fits'):  # the observation is awaiting extraction
        obs_fits = args.extract
        steps.append('extract')

    # execute the steps:
    for step in steps:
        if step == 'msf':
            msf_file = fitmsf(
                msf_table=msf_table,
                sim=sim,
                resid_map=args.resid_map,
                outdir=args.outdir,
                bin_range=args.bin_range,
                plot=args.plot
            )
        if step == 'wt_sort':
            wavecal_fits = ordersort(
                table=wavecal_table,
                filename='emission',
                msf_file=msf_file,
                resid_map=args.resid_map,
                outdir=args.outdir,
                plot=args.plot
            )
        if step == 'wavecal':
            wavecal_file = wavecal(
                wavecal_fits=wavecal_fits,
                orders=args.orders,
                elem=args.elem,
                minw=args.minw,
                maxw=args.maxw,
                resid_max=args.resid_max,
                degree=args.degree,
                iters=args.iters,
                dim=args.dim,
                shift_window=args.shift_window,
                manual_fit=args.manual_fit,
                width=args.width,
                outdir=args.outdir,
                plot=args.plot
            )
        if step == 'ot_sort':
            obs_fits = ordersort(
                table=obs_table,
                filename='observation',
                msf_file=msf_file,
                resid_map=args.resid_map,
                outdir=args.outdir,
                plot=args.plot
            )
        if step == 'extract':
            extract(
                obs_fits=obs_fits,
                wavecal_file=wavecal_file,
                plot=args.plot
            )
