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
from ucsbsim.spectra import get_spectrum, apply_bandpass, clip_spectrum, FineGrid
import ucsbsim.engine as engine
from ucsbsim.msf import MKIDSpreadFunction
from ucsbsim.spectrograph import GratingSetup, SpectrographSetup
from ucsbsim.detector import MKIDDetector, wave_to_phase
from ucsbsim.plotting import quick_plot
from ucsbsim.simsettings import SpecSimSettings


def parse():
    # read in command line arguments
    arg_desc = '''
                   Extract the MSF Spread Function from the calibration spectrum.
                   --------------------------------------------------------------
                   This program loads the calibration photon table and conducts non-linear least squares fits to determine
                   bin edges and the covariance matrix.
                   '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)

    # required args:
    parser.add_argument('output_dir',
                        metavar='OUTPUT_DIRECTORY',
                        help='Directory for the output files (str).')
    parser.add_argument('msftable',
                        metavar='MSF_PHOTON_TABLE',
                        help='Directory/name of the MSF photon table, either flat-field or blackbody.')


    # optional MSF args:
    parser.add_argument('-br', '--bin_range',
                        metavar='BIN_RANGE',
                        default=(-1, 0),
                        help='Tuple containing start and stop of range for histogram.')


    # optional script args:
    parser.add_argument('-pr', '--plotresults',
                        action='store_true',
                        default=False,
                        help='If passed, indicates that plots showing goodness-of-fit for each pixel will be shown.')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse()
    now = dt.now()
    logging.basicConfig(
        filename=f'{args.output_dir}/logging/msf_{now.strftime("%Y%m%d_%H%M%S")}.log',
        format='%(levelname)s:%(message)s',
        level=logging.INFO
    )
    logging.info(msg="The process of modeling the MKID Spread Function (MSF) is recorded."
                     f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")
