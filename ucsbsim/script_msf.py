import numpy as np
from numpy.polynomial.legendre import Legendre
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
import matplotlib.patches as mpatches
import matplotlib as mpl
import scipy.signal
import scipy
import scipy.interpolate as interp
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy
import numdifftools as ndt

import astropy.units as u
import time
from datetime import datetime as dt
import argparse
import logging
import os
from lmfit import Parameters, minimize, Minimizer
from synphot import SourceSpectrum
from synphot.models import BlackBodyNorm1D, ConstFlux1D

from mkidpipeline.photontable import Photontable
from ucsbsim.spectra import get_spectrum
import ucsbsim.engine as engine
from ucsbsim.msf import MKIDSpreadFunction
from ucsbsim.spectrograph import GratingSetup, SpectrographSetup
from ucsbsim.detector import MKIDDetector, wave_to_phase
from ucsbsim.plotting import quick_plot
from ucsbsim.simsettings import SpecSimSettings
from ucsbsim.lmfit_conf_revised import conf_interval

# TODO edit below
"""
Extraction of the MKID Spread Function from calibration spectrum. The steps are:
-Load the calibration photon table.
-Fit Gaussians to each pixel from polynomial evaluations.

-Use overlap points of the Gaussians to get bin edges for each order.
-Calculate fractional overlap between Gaussians and converts them into an n_ord x n_ord "covariance" matrix for each
 pixel. This matrix details what fraction of each order Gaussian was grouped into another order Gaussian due to binning.
 This will later be used to determine the error band of some extracted spectrum.
-Saves newly obtained bin edges and covariance matrices to files.
"""

'''
def init_params_new(
        x_phases,
        y_counts,
        n_bins,
        snr,
        avg_peak_sep,
        simulation,
        spectrograph,
        pix,
        miss_per=0.1,
        reg_per=0.15,
        degree=2,
        e_domain=[-1, 0]
):
    parameters = Parameters()

    peaks_idx, _ = scipy.signal.find_peaks(y_counts, distance=int(0.1 * n_bins), height=snr ** 2)
    sort_idx = np.argsort(y_counts[peaks_idx])[::-1][:nord]  # get rid of smallest peaks, if more than nord
    peaks_sort_idx = np.sort(peaks_idx[sort_idx])  # sorting the indices by phase
    peaks_phi = y_counts[peaks_sort_idx]  # the phases of the retrieved peaks

    if len(peaks_sort_idx) == nord:
        peaks_nord = peaks_sort_idx  # indices of final peaks
        miss_ord = None
    else:
        # let the missing orders decision tree begin
        n_missing = nord - len(sort_idx)  # there are n orders missing

        # calculate the peak separation and round to avg_sep units:
        peak_seps = np.diff(peaks_phi)
        n_peak_seps = np.round(peak_seps / avg_peak_sep)
        if np.all(n_peak_seps == n_peak_seps[0]):  # all equal
            # check if there is enough room for n orders prior to the first peak with that separation
            orders_before = peaks_phi[-1] + peak_seps[-1] * np.range(1, n_missing+1)
            if np.round(orders_before[-1]/avg_offset) >= 1:  # enough room to place n orders at front
                peaks_nord = np.concatenate((peaks_sort_idx, nearest_idx(x_phases, orders_before)))
                miss_ord = np.range(0, n_missing)
            else:  # not enough room
                # check if there is enough room for n orders after the last peak with that separation
                orders_after = peaks_phi[0] - peak_seps[0] * np.range(1, n_missing + 1)[::-1]
                if orders_after[0] > x_phases[0]:  # enough room to place n orders at end
                    peaks_nord = np.concatenate((nearest_idx(x_phases, orders_after), peaks_sort_idx))
                else:  # not enough room
                    # check if n is equal to number of separations
                    if len(n_missing) == len(n_peak_seps):  # it is
                        # place missing orders inbetween the found peaks
                        peaks_nord = np.insert(
                            peaks_sort_idx,  # the idx with missing orders
                            range(1, len(peaks_sort_idx)),  # the array locations to insert orders
                            peaks_sort_idx[:-1]+np.diff(peaks_sort_idx)/2  # the inbetween indices to add
                        )
                    # check if n is equal to the number of separations + 1
                    elif len(n_missing) == len(n_peak_seps)+1:  # it is
                        # check if there is enough room at the beginning for 1 order
                        one_order_before = peaks_phi[-1] + peak_seps[-1]
                        if np.round(one_order_before / avg_offset) >= 1:  # there is
                            # step 7: place the orders in between + at beginning, done
                        else:  # there is not
                            # check if there is enough room at end for 1 order
                            one_order_after = peaks_phi[0] - peak_seps[0]
                            if one_order_after > x_phases[0]:  # there is
                                # step 9: place the orders in between + at end, done
                            else:  # there is not
                                raise ValueError(f'No missing orders solution for Pixel {pix}.')
                    # check if n is equal to the number of separations + 2
                    elif len(n_missing) == len(n_peak_seps)+2:
                        # step 10: place the orders in between + at front + at end, done
                    else:  # it is not
                        raise ValueError(f'No missing orders solution for Pixel {pix}.')
        else:  # all are not equal
            # step 2: check if the number of ~2x seps is equal to n
                        # it is:
                            # step 3: place orders there, done
                        # it is not:
                            # step 3: check if the number ~2x seps + 1 is equal to n
                                # it is:
                                    # step 4: check if there is room for one before the first
                                        # there is:
                                            # step 5: place n-1 at the number of ~2x seps and one order at the front
                                        # there is not:
                                            # step 5: check if there is room for one at the end
                                                # there is:
                                                    # step 6: place n-1 at the number of ~2x seps and one at the end
                                                # there is not:
                                                    # FAILURE
                                # it is not:
                                    # step 4: check if the number of ~2x seps + 2 is equal to n
                                        # it is:
                                            # step 5: check if you can place an order at both ends
                                                # i can:
                                                    # step 6: do that + put the rest at the ~2x seps
                                                # i cant:
                                                    # FAILURE
                                        # it is not:
                                            # FAILURE

    # use Legendre polyfitting on the theoretical/peak-finding data to get guess coefs
    e_coefs = Legendre.fit(x=phi_guess, y=e_guess, deg=degree, domain=e_domain).coef

    s_coefs = Legendre.fit(x=e_guess, y=s_guess, deg=degree, domain=[e_guess[0], e_guess[-1]]).coef

    # add phi_0s to params object:
    parameters.add(name=f'phi_0', value=phi_guess[-1], min=phi_guess[-1] - 0.2, max=phi_guess[-1] + 0.2)

    # add amplitudes to params object:
    if miss_ord is None:  # no orders were identified as missing
        for i in range(len(amp_guess)):
            parameters.add(
                name=f'O{i}_amp',
                value=amp_guess[i],
                min=amp_guess[i] * (1 - reg_per),
                max=amp_guess[i] * (1 + reg_per)
            )
    else:
        for i in range(len(amp_guess)):
            if i in miss_ord:
                parameters.add(name=f'O{i}_amp', value=amp_guess[i], vary=False)
            else:
                parameters.add(
                    name=f'O{i}_amp',
                    value=amp_guess[i],
                    min=amp_guess[i] * (1 - reg_per),
                    max=amp_guess[i] * (1 + reg_per)
                )

    # add polynomial coefficients to params object:
    parameters.add(name=f'e0', value=e_coefs[0], min=0)  # , min=e0_bound[0], max=e0_bound[-1])
    parameters.add(name=f'e1', value=e_coefs[1], max=0)  # , min=e1_bound[0], max=e1_bound[-1])
    parameters.add(name=f'e2', value=e_coefs[2])  # , min=e2_bound[0], max=e2_bound[-1])  # the curvation factor is small

    parameters.add(name=f's0', value=s_coefs[0], min=0)  # , min=s0_bound[0], max=s0_bound[-1])
    parameters.add(name=f's1', value=s_coefs[1], min=0)  # , min=s1_bound[0], max=s1_bound[-1])
    parameters.add(name=f's2', value=s_coefs[2])  # , min=-0.01, max=0.01)  # the curvation coef is of order the others

    return None
    '''


def init_params(
        phi_guess,
        e_guess,
        s_guess,
        amp_guess,
        miss_ord=None,
        miss_per=0.3,
        reg_per=0.3,
        degree=2,
        e_domain=[-1, 0]
):
    """
    :param phi_guess: n_ord guess values for phases
    :param e_guess: n_ord guess values for energies of given phases
    :param s_guess: n_ord guess values for sigmas of given energies
    :param amp_guess: n_ord guess values for fit_amps
    :param miss_ord: as list, the orders that are probably missing/0
    :param miss_per: the percentage to add to the threshold for missing orders
    :param reg_per: the percentage to add to the threshold for existing orders
    :param degree: degree to use for polynomial fitting, default 2nd order
    :param e_domain: domain for the legendre polynomial
    :return: an lmfit Parameter object with the populated parameters and guess values
    """
    parameters = Parameters()

    # use Legendre polyfitting on the theoretical/peak-finding data to get guess coefs
    e_coefs = Legendre.fit(x=phi_guess, y=e_guess / e_guess[-1], deg=degree, domain=e_domain).coef

    s_coefs = Legendre.fit(x=e_guess / e_guess[-1], y=s_guess, deg=degree, domain=[e_guess[0] / e_guess[-1], 1]).coef

    parameters.add(name=f's0', value=s_coefs[0])
    parameters.add(name=f's1', value=s_coefs[1])
    parameters.add(name=f's2', value=s_coefs[2])

    # add phi_0s to params object:
    parameters.add(name=f'phi_0', value=phi_guess[-1], min=phi_guess[-1] - 0.2, max=phi_guess[-1] + 0.2)

    # add polynomial coefficients to params object:
    parameters.add(name=f'e1', value=e_coefs[1])
    parameters.add(name=f'e2', value=e_coefs[2])

    # add amplitudes to params object:
    if miss_ord is None:  # no orders were identified as missing
        for i in range(len(amp_guess)):
            parameters.add(
                name=f'O{i}_amp',
                value=amp_guess[i],
                # min=amp_guess[i] * (1 - reg_per),
                max=amp_guess[i] * (1 + reg_per)
            )
    else:
        for i in range(len(amp_guess)):
            if i in miss_ord:
                parameters.add(name=f'O{i}_amp', value=0, vary=False)
            else:
                parameters.add(
                    name=f'O{i}_amp',
                    value=amp_guess[i],
                    # min=amp_guess[i] * (1 - reg_per),
                    max=amp_guess[i] * (1 + reg_per)
                )

    return parameters


def fit_func(
        params,
        x_phases,
        y_counts=None,
        orders=None,
        nord=None,
        pix: int = None,
        leg_e=None,
        leg_s=None,
        degree=2,
        plot=False,
        to_sum=False
):
    # turn dictionary of param values into separate params:
    s0, s1, s2, phi_0, e1, e2, *amps = tuple(params.valuesdict().values())
    e0 = e0_from_params(e1, e2, phi_0)

    # pass coef parameters to energy/sigma polys:
    setattr(leg_e, 'coef', [e0, e1, e2])
    setattr(leg_s, 'coef', (s0, s1, s2))

    try:
        # calculate the other phi_m based on phi_0:
        phis = phis_from_grating_eq(orders, phi_0, leg=leg_e, coefs=[e0, e1, e2])

        # get sigmas at each phase center:
        sigs = leg_s(leg_e(phis))

        # put the phi, sigma, and fit_amps together for Gaussian model:
        gauss_i = np.nan_to_num(gauss(x_phases, phis[:, None], sigs[:, None], np.array(amps)[:, None]), nan=0)
        model = np.sum(gauss_i, axis=1).flatten()

        if y_counts is not None:
            if np.iscomplex(phis).any() or not np.isfinite(phis).any():
                residual = np.full_like(y_counts, np.max(y_counts) / np.sqrt(np.max(y_counts)))
            else:
                # get the residuals and weighted reduced chi^2:
                residual = np.divide(model - y_counts, np.sqrt(y_counts), where=y_counts != 0)
                residual[y_counts == 0] = model[y_counts == 0]

    except IndexError:
        if y_counts is not None:
            residual = np.full_like(y_counts, np.max(y_counts) / np.sqrt(np.max(y_counts)))
        else:
            print(f'Cannot retrieve model for pixel {pix}.')

    # WARNING, WILL PLOT HUNDREDS IF FIT SUCKS/CAN'T CONVERGE
    if plot:

        N_dof = len(residual) - len(params)
        red_chi2 = np.sum(residual ** 2) / N_dof

        fig = plt.figure(1)
        ax = fig.add_axes((.1, .3, .8, .6))
        ax.plot(x_phases, y_counts, label='Data')
        for n, i in enumerate(gauss_i.T):
            ax.plot(x_phases, i, label=f'O{7 - n}')
        ax.legend()
        ax.set_ylabel('Count')
        ax.set_xlim([-1, 0])
        ax.set_xticklabels([])
        ax.legend()
        res = fig.add_axes((.1, .1, .8, .2))
        res.grid()
        res.plot(x_phases, model - y_counts, '.', color='purple')
        res.set_ylabel('Residual')
        res.set_xlabel('Phase')
        res.set_xlim([-1, 0])
        res.text(-0.3, 10, f'Red. Chi^2={red_chi2:.1f}')
        plt.suptitle(f"Pixel {pix} Fitting Iteration")
        plt.show()

    if y_counts is None:
        if to_sum:
            return model
        else:
            return gauss_i
    else:
        return residual


def extract_params(params, n_ord, degree=2):
    """
    :param params: Parameter object
    :param n_ord: number of orders
    :param degree: the polynomial degree
    :return: the extracted parameters
    """
    phi_0 = params[f'phi_0'].value
    e_coefs = np.array([params[f'e{c}'].value for c in range(1, degree + 1)])
    s_coefs = np.array([params[f's{c}'].value for c in range(degree + 1)])
    amps = np.array([params[f'O{i}_amp'].value for i in range(n_ord)])

    return phi_0, e_coefs, s_coefs, amps


def gauss(x, mu, sig, A):
    """
    :param x: wavelength
    :param mu: mean
    :param sig: sigma
    :param A: amplitude
    :return: value of the Gaussian
    """
    return (A * np.exp(- (x - mu) ** 2. / (2. * sig ** 2.))).T


def quad_formula(a, b, c):
    """
    :return: quadratic formula results for ax^2 + bx + c = 0 or linear bx + c = 0
    """
    if a == 0:
        return [-c / b]
    else:
        return [((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)), ((-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))]


def gauss_intersect(mu, sig, A):
    """
    :param mu: 2 means of Gaussians
    :param sig: 2 sigmas of Gaussians
    :param A: 2 amplitudes of Gaussians
    :return: analytic calculation of the intersection point between 2 1D Gaussian functions
    """
    if A[1] == 0:
        return np.nan
    else:
        n = len(mu)
        if n != len(sig) or n != len(A):
            raise ValueError("mu, sig, and A must all be the same size.")
        a = 1 / sig[0] ** 2 - 1 / sig[1] ** 2
        b = 2 * mu[1] / sig[1] ** 2 - 2 * mu[0] / sig[0] ** 2
        c = (mu[0] / sig[0]) ** 2 - (mu[1] / sig[1]) ** 2 - 2 * np.log(A[0] / A[1])
        if a == 0:
            return quad_formula(a=a, b=b, c=c)[0]
        else:
            solp, soln = tuple(quad_formula(a=a, b=b, c=c))
            if mu[0] < solp < mu[1]:
                return solp
            elif mu[0] < soln < mu[1]:
                return soln
            else:
                return np.nan


def nearest_idx(array, value):
    return (np.abs(array - value)).argmin()


def sig_to_R(sig, lam):
    dlam = sig * 2 * np.sqrt(2 * np.log(2))
    R = lam / dlam
    return np.abs(R)


def e0_from_params(e1, e2, phi_0):
    return 1 - e2 * (1 / 2 * (3 * ((phi_0 + 0.5) * 2) ** 2 - 1)) - e1 * 2 * (phi_0 + 0.5)


def phis_from_grating_eq(orders, phi_0, leg, coefs=None, analytic=True):
    grating_eq = orders[1:][::-1] / orders[0] * leg(phi_0)

    phis = []
    for i in grating_eq:
        if analytic:
            # all weird because domain centering added extra terms
            # roots = np.array(quad_formula(6*coefs[2], 6*coefs[2]+2*coefs[1], coefs[2] + coefs[0] + coefs[1] - i))
            roots = np.roots([6 * coefs[2], 6 * coefs[2] + 2 * coefs[1], coefs[2] + coefs[0] + coefs[1] - i])
        else:
            roots = (leg_e - i).roots()
        if ~np.isfinite(roots).all():
            phis.append(np.nan)
        elif len(roots) == 1:
            phis.append(roots[0])
        elif all(roots < 0):
            phis.append(roots[(-1 < roots) & (roots < phi_0)][0])
        else:
            phis.append(roots.min())

    return np.append(np.array(phis), phi_0)


def cov_from_params(params, model, nord, edges_idx, dx, **kwargs):
    model_sum = np.array([np.sum(model[edges_idx[i]:edges_idx[i + 1]]) * dx for i in range(nord)])
    cov = np.empty([nord, nord])
    for k in range(nord):
        suppress_params = copy.deepcopy(params)
        suppress_params.add(f'O{k}_amp', value=0)
        suppress_model = fit_func(suppress_params, nord=nord, **kwargs)
        suppress_model_sum = np.array([np.sum(suppress_model[edges_idx[i]:edges_idx[i + 1]]) * dx for i in range(nord)])
        cov[:, k] = np.nan_to_num((model_sum - suppress_model_sum) / model_sum)
    return cov


if __name__ == '__main__':
    tic = time.perf_counter()  # recording start time for script

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
    parser.add_argument('-br', '--bin_range',
                        metavar='BIN_RANGE',
                        default=(-1, 0),
                        help='Tuple containing start and stop of range for histogram.')
    parser.add_argument('-pr', '--plotresults',
                        action='store_true',
                        default=False,
                        help='If passed, indicates that plots showing goodness-of-fit for each pixel will be shown.')

    # set arguments as variables
    args = parser.parse_args()

    # ==================================================================================================================
    # CHECK AND/OR CREATE DIRECTORIES
    # ==================================================================================================================
    os.makedirs(name=f'{args.output_dir}/logging', exist_ok=True)

    # ==================================================================================================================
    # START LOGGING TO FILE
    # ==================================================================================================================
    now = dt.now()
    logging.basicConfig(
        filename=f'{args.output_dir}/logging/msf_{now.strftime("%Y%m%d_%H%M%S")}.log',
        format='%(levelname)s:%(message)s',
        level=logging.INFO
    )
    logging.info(msg="The process of recovering the MKID Spread Function (MSF) is recorded."
                     f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    # ==================================================================================================================
    # OPEN PHOTON TABLE AND PULL NECESSARY DATA
    # ==================================================================================================================
    pt = Photontable(file_name=args.caltable)
    sim = pt.query_header(name='sim_settings')
    phases = pt.query(column='wavelength')
    resID = pt.query(column='resID')

    resid_map = np.arange(sim.npix, dtype=int) * 10 + 100  # TODO replace once known
    idx = [np.where(resID == resid_map[j]) for j in range(sim.npix)]
    photons_pixel = [phases[idx[j]].tolist() for j in range(sim.npix)]  # list of photons in each pixel

    num_pixel = [len(photons_pixel[j]) for j in range(sim.npix)]  # number of photons in all pixels
    sparse_pixel = int(np.min(num_pixel))  # number of photons in sparsest pixel

    # ==================================================================================================================
    # INSTANTIATE SPECTROGRAPH & DETECTOR
    # ==================================================================================================================
    # checking if the R0s file exists:
    if not os.path.exists(path=sim.R0s_file):
        IOError('File does not exist, check path and file name.')
    else:
        R0s = np.loadtxt(fname=sim.R0s_file, delimiter=',')
        logging.info(msg=f'\nThe individual R0s were imported from {sim.R0s_file}.')

    # creating the spectrometer objects:
    detector = MKIDDetector(
        n_pix=sim.npix,
        pixel_size=sim.pixelsize,
        design_R0=sim.designR0,
        l0=sim.l0,
        R0s=R0s,
        resid_map=resid_map
    )

    grating = GratingSetup(sim.alpha, sim.delta, sim.groove_length)
    spectro = SpectrographSetup(sim.m0, sim.m_max, sim.l0, sim.pixels_per_res_elem, sim.focallength, grating, detector)
    eng = engine.Engine(spectro)

    # shortening some long variable names:
    nord = spectro.nord
    pixels = detector.pixel_indices

    # converting and calculating the simulation phases, energies, sigmas ("true"):
    sim_phase = np.nan_to_num(
        wave_to_phase(spectro.pixel_wavelengths().to(u.nm)[::-1], minwave=sim.minwave, maxwave=sim.maxwave))
    # avg_peak_sep = np.average(np.diff(sim_phase, axis=0))

    energy = np.nan_to_num(engine.wave_to_eV(spectro.pixel_wavelengths().to(u.nm)[::-1]).value)
    sig_start = wave_to_phase(spectro.pixel_wavelengths().to(u.nm)[::-1] - (detector.mkid_resolution_width(
        spectro.pixel_wavelengths().to(u.nm)[::-1], pixels) / (2 * np.log(2))) / 2,
                              minwave=sim.minwave, maxwave=sim.maxwave)
    sig_end = wave_to_phase(spectro.pixel_wavelengths().to(u.nm)[::-1] + (detector.mkid_resolution_width(
        spectro.pixel_wavelengths().to(u.nm)[::-1], pixels) / (2 * np.log(2))) / 2,
                            minwave=sim.minwave, maxwave=sim.maxwave)
    sigmas = sig_end - sig_start
    # for all, flip order axis to be in ascending phase/lambda

    # ==================================================================================================================
    # MSF EXTRACTION STARTS
    # ==================================================================================================================
    # generating bins and building initial histogram for every pixel:
    n_bins = engine.n_bins(sparse_pixel, method="rice")  # bins the same for all, based on pixel with fewest photons
    bin_edges = np.linspace(args.bin_range[0], args.bin_range[1], n_bins + 1, endpoint=True)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    bin_counts = np.zeros([n_bins, sim.npix])
    for j in pixels:
        bin_counts[:, j], _ = np.histogram(photons_pixel[j], bins=bin_edges)

    # create lists/arrays to place loop values, param objects can only go in lists:
    red_chi2 = np.full([sim.npix], fill_value=1e6)
    missing_cov = []

    # create empty arrays to hold phi values for gaussian model:
    fit_phi = np.empty([nord, sim.npix])
    gausses = np.zeros([1000, sim.npix])  # the entire n_ord Gaussian model summed
    gausses_i = np.zeros([1000, nord, sim.npix])

    covariance = np.zeros([nord, nord, sim.npix])  # the overlap matrix

    # create empty array for order bin edges:
    order_edges = np.empty([nord + 1, sim.npix])
    order_edges[0, :] = -1

    # separate index for list indexing because param objects can only be in lists:
    n = 0

    # TODO change pixel indices here
    redchi_val = 5
    # cov_cutoff = 0.01
    max_evals = None  # 1500
    # xtol = 1e-5
    method = 'leastsq'
    pixels = [1735]#, 1743, 1836, 1964]
    #pixels = [1735, 1764, 1769, 1772, 1797, 1811, 1830, 1853, 1854, 1862, 1872]
    leg_e = Legendre(coef=(0, 0, 0), domain=np.array(args.bin_range))
    leg_e_pre = Legendre(coef=(0, 0, 0), domain=np.array(args.bin_range))
    snr = 3
    # do the lmfit:
    for p in pixels:
        print(p)
        leg_s = Legendre(coef=(0, 0, 0), domain=[energy[0, p] / energy[-1, p], 1])
        leg_s_pre = Legendre(coef=(0, 0, 0), domain=[energy[0, p] / energy[-1, p], 1])

        # TODO move into function, can be function of parameters and use avg distance from 0 + avg peak sep
        # TODO construct initial guess algo by ruling out different cases depending on peaks
        peaks, _ = scipy.signal.find_peaks(bin_counts[:, p], distance=int(0.1 * n_bins), height=snr ** 2)
        # use find peaks to get phi/amplitude guesses IF there are at least n_ord peaks,
        if len(peaks) >= nord:
            idx = np.argsort(bin_counts[peaks, p])[::-1][:nord]
            peaks_nord = np.sort(peaks[idx])
            miss_ord = None
        else:
            idx = np.argsort(bin_counts[peaks, p])[::-1]
            peaks_sort = np.sort(peaks[idx])
            if p <= 346 and len(peaks_sort) == 3:
                peaks_nord = np.array([
                    peaks_sort[0],
                    peaks_sort[1],
                    int((peaks_sort[1] + peaks_sort[2]) / 2),
                    peaks_sort[2]]
                )
                miss_ord = [2]
            elif p <= 1300 and len(peaks_sort) == 2:
                peaks_nord = np.array([
                    int(peaks_sort[0] - (peaks_sort[1] - peaks_sort[0]) / 2),
                    peaks_sort[0],
                    int((peaks_sort[0] + peaks_sort[1]) / 2),
                    peaks_sort[1]]
                )
                miss_ord = [0, 2]
            elif 346 < p <= 1300 and len(peaks_sort) == 3:
                peaks_nord = np.array([
                    int(peaks_sort[0] - (peaks_sort[1] - peaks_sort[0]) / 2),
                    peaks_sort[0],
                    peaks_sort[1],
                    peaks_sort[2]]
                )
                miss_ord = [0]
            elif p > 1300 and len(peaks_sort) == 3:
                peaks_nord = np.array([
                    peaks_sort[0],
                    int((peaks_sort[0] + peaks_sort[1]) / 2),
                    peaks_sort[1],
                    peaks_sort[2]]
                )
                miss_ord = [1]
            elif p > 1300 and len(peaks_sort) == 2:
                peaks_nord = np.array([
                    peaks_sort[0],
                    int((peaks_sort[0] + peaks_sort[1]) / 2),
                    peaks_sort[1],
                    int(peaks_sort[1] + (peaks_sort[1] - peaks_sort[0]) / 2)]
                )
                miss_ord = [1, 3]
            else:
                logging.info(f'Pixel {p} will default to simulation guess.')
                peaks_nord = [nearest_idx(bin_centers, sim_phase[i, p]) for i in range(nord)]
                miss_ord = None

        phi_init = bin_centers[peaks_nord]
        amp_init = bin_counts[peaks_nord, p]

        # obtain Parameter object:
        params = init_params(phi_guess=phi_init, e_guess=energy[:, p], s_guess=sigmas[:, p],
                             amp_guess=amp_init, miss_ord=miss_ord)

        opt_params = minimize(
            fit_func,
            params,  # params
            args=(
                bin_centers,  # x_phases
                bin_counts[:, p],  # y_counts
                spectro.orders,  # orders
                nord,  # number of orders
                p,  # pix
                leg_e,  # energy legendre poly object
                leg_s,  # sigma legendre poly object
                2,  # degree
                False,  # plot  # TODO CHANGE
            ),
            nan_policy='omit',
            max_nfev=max_evals,
            method=method,
            # xtol=xtol
        )
        # print(opt_params.message)
        red_chi2[p] = opt_params.redchi

        # extract the successfully fit parameters:
        fit_phi0, fit_e_coef, fit_s_coef, fit_amps = extract_params(opt_params.params, nord)
        fit_e0 = e0_from_params(fit_e_coef[0], fit_e_coef[1], fit_phi0)

        setattr(leg_e, 'coef', [fit_e0, fit_e_coef[0], fit_e_coef[1]])
        setattr(leg_s, 'coef', fit_s_coef)

        fit_phis = phis_from_grating_eq(spectro.orders, fit_phi0, leg=leg_e,
                                        coefs=[fit_e0, fit_e_coef[0], fit_e_coef[1]])
        fit_sigs = leg_s(leg_e(fit_phis))

        # get the order bin edges:
        idx = np.argwhere(fit_amps != 0).flatten()
        for h, i in enumerate(idx[:-1]):
            order_edges[i + 1, p] = gauss_intersect(fit_phis[[i, idx[h + 1]]], fit_sigs[[i, idx[h + 1]]],
                                                    fit_amps[[i, idx[h + 1]]])

        # store model to array:
        fine_phase_grid = np.linspace(-1, 0, 1000)
        gausses_i[:, :, p] = fit_func(params=opt_params.params, x_phases=fine_phase_grid,
                                      orders=spectro.orders, nord=nord, pix=p, leg_e=leg_e, leg_s=leg_s)
        gausses[:, p] = np.sum(gausses_i[:, :, p], axis=1)

        # generate more robust errors on parameters:
        if opt_params.params['phi_0'].stderr is None:
            for par in opt_params.params:
                opt_params.params[par].stderr = abs(opt_params.params[par].value * 0.1)

        mini = Minimizer(fit_func, opt_params.params, fcn_args=(
            bin_centers,  # x_phases
            bin_counts[:, p],  # y_counts
            spectro.orders,  # orders
            nord,  # number of orders
            p,  # pix
            leg_e,  # energy legendre poly object
            leg_s,  # sigma legendre poly object
            2,  # degree
            False,))

        ci = conf_interval(mini, opt_params, sigmas=[1], maxiter=400)
        vary_params = list(ci.keys())

        # find order-bleeding covar using CI errors:

        # put this into definition:
        dx = fine_phase_grid[1] - fine_phase_grid[0]
        edges_idx = [nearest_idx(fine_phase_grid, e) for e in order_edges[:, p]]

        original_cov = cov_from_params(opt_params.params, gausses[:, p], nord, edges_idx, dx,
                                       x_phases=fine_phase_grid,
                                       orders=spectro.orders,
                                       pix=p,
                                       leg_e=leg_e,
                                       leg_s=leg_s,
                                       to_sum=True
                                       )

        per_leg_e = Legendre(coef=(0, 0, 0), domain=np.array(args.bin_range))
        per_leg_s = Legendre(coef=(0, 0, 0), domain=[energy[0, p] / energy[-1, p], 1])
        perturb_params = copy.deepcopy(opt_params.params)
        for n_p, par in enumerate(vary_params):
            original_val = opt_params.params[par].value
            p_err = ci[par][2][1]-ci[par][1][1]
            m_err = ci[par][1][1]-ci[par][0][1]

            perturb_params.add(par, value=original_val + p_err)
            perturb_model = fit_func(params=perturb_params, x_phases=fine_phase_grid, orders=spectro.orders,
                                     nord=nord, pix=p, leg_e=per_leg_e, leg_s=per_leg_s, to_sum=True)
            cov_p = cov_from_params(perturb_params, perturb_model, nord, edges_idx, dx,
                                    x_phases=fine_phase_grid,
                                    orders=spectro.orders,
                                    pix=p,
                                    leg_e=per_leg_e,
                                    leg_s=per_leg_s,
                                    to_sum=True
                                    )

            # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
            # axes = axes.ravel()
            # ax1 = axes[0]
            # ax2 = axes[1]
            # ax1.grid()
            # ax1.set_title(f'Parameter "{par}":\n'
            #               f'Optimized: {original_val:.3e} PLUS Uncert: {p_err:.3e}')
            # ax1.set_ylabel('Photon Count')
            # ax1.set_xlabel(r'Phase $\times 2\pi$')
            # ax1.plot(bin_centers, bin_counts[:, p], 'k', label='Data')
            # ax1.fill_between(bin_centers, 0, bin_counts[:, p], color='k')
            # ax1.plot(fine_phase_grid, fit_func(opt_params.params, fine_phase_grid, orders=spectro.orders, nord=nord,
            #                                    pix=p, leg_e=leg_e, leg_s=leg_s, to_sum=True), label='Optimized')
            # ax1.plot(fine_phase_grid, fit_func(perturb_params, fine_phase_grid, orders=spectro.orders, nord=nord, pix=p,
            #                                    leg_e=per_leg_e, leg_s=per_leg_s, to_sum=True), label='Perturbed')
            # ax1.legend()

            perturb_params.add(par, value=original_val - m_err)
            perturb_model = fit_func(params=perturb_params, x_phases=fine_phase_grid, orders=spectro.orders,
                                     nord=nord, pix=p, leg_e=per_leg_e, leg_s=per_leg_s, to_sum=True)
            cov_n = cov_from_params(perturb_params, perturb_model, nord, edges_idx, dx,
                                    x_phases=fine_phase_grid,
                                    orders=spectro.orders,
                                    pix=p,
                                    leg_e=per_leg_e,
                                    leg_s=per_leg_s,
                                    to_sum=True
                                    )

            # ax2.grid()
            # ax2.set_title(f'Parameter "{par}":\n'
            #               f'Optimized: {original_val:.3e} MINUS Uncert: {m_err:.3e}')
            # ax2.set_ylabel('Photon Count')
            # ax2.set_xlabel(r'Phase $\times 2\pi$')
            # ax2.plot(bin_centers, bin_counts[:, p], 'k', label='Data')
            # ax2.fill_between(bin_centers, 0, bin_counts[:, p], color='k')
            # ax2.plot(fine_phase_grid, fit_func(opt_params.params, fine_phase_grid, orders=spectro.orders, nord=nord,
            #                                    pix=p, leg_e=leg_e, leg_s=leg_s, to_sum=True), label='Optimized')
            # ax2.plot(fine_phase_grid, fit_func(perturb_params, fine_phase_grid, orders=spectro.orders, nord=nord, pix=p,
            #                                    leg_e=per_leg_e, leg_s=per_leg_s, to_sum=True), label='Perturbed')
            # ax2.legend()
            # plt.show()

            if ((original_cov - cov_p) ** 2).sum() > ((original_cov - cov_n) ** 2).sum():
                covariance[:, :, p] = cov_p
                perturb_params.add(par, value=original_val + p_err)
            else:
                covariance[:, :, p] = cov_n
                perturb_params.add(par, value=original_val - m_err)

        if (np.abs(covariance[:, :, p] - original_cov) > 0.01).any():
            bleed = np.nanmax(np.abs(covariance[:, :, p] - original_cov))
            print(f'The max change in covariance was {bleed:.3f} for Pixel {p}.')

        # plot the individual pixels:
        if red_chi2[p] > redchi_val:
            param_vals = list(opt_params.params.valuesdict().values())

            # get the post residuals and weighted reduced chi^2:
            opt_residual = fit_func(
                params=opt_params.params,
                x_phases=bin_centers,
                y_counts=bin_counts[:, p],
                orders=spectro.orders,
                nord=nord,
                pix=p,
                leg_e=leg_e,
                leg_s=leg_s
            )

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
            axes = axes.ravel()
            ax1 = axes[0]
            ax2 = axes[1]

            plt.suptitle(f'Pixel {p}')

            size1 = '30%'
            size2 = '100%'

            divider1 = make_axes_locatable(ax1)
            divider2 = make_axes_locatable(ax2)

            res1 = divider1.append_axes("top", size=size1, pad=0)
            res2 = divider1.append_axes("bottom", size=size1, pad=0)

            ax2_2 = divider2.append_axes("bottom", size=size2, pad=0)

            ax1.figure.add_axes(res1)
            ax1.figure.add_axes(res2)
            # TODO make the middle figure log scale
            ax2.figure.add_axes(ax2_2)

            # get the pre initial guess
            pre_gauss = fit_func(params, fine_phase_grid, orders=spectro.orders, nord=nord, pix=p, leg_e=leg_e_pre,
                                 leg_s=leg_s_pre, to_sum=True)

            # get the pre residuals and weighted reduced chi^2:
            pre_residual = fit_func(params, bin_centers, bin_counts[:, p], spectro.orders, nord, p, leg_e_pre,
                                    leg_s_pre)
            N = len(pre_residual)
            N_dof2 = N - len(param_vals)
            pre_red_chi2 = np.sum(pre_residual ** 2) / N_dof2

            ax1.grid()
            ax1.plot(bin_centers, bin_counts[:, p], 'k', label='Data')
            ax1.fill_between(bin_centers, 0, bin_counts[:, p], color='k')
            ax1.plot(fine_phase_grid, pre_gauss, color='gray', label='Init. Guess')
            quick_plot(ax1, [fine_phase_grid for i in range(nord)], gausses_i[:, :, p].T,
                       labels=[f'Order {i}' for i in spectro.orders[::-1]],
                       title=f'Least-Squares Fit', xlabel=r'Phase $\times 2\pi$', ylabel='Photon Count')
            for b in order_edges[:-1, p]:
                ax1.axvline(b, linestyle='--', color='black')
            ax1.axvline(order_edges[-1, p], linestyle='--', color='black', label='Order Edges')
            ax1.set_xlim([-1, 0])
            # ax1.set_yscale(matplotlib.scale.FuncScale(ax1, (lambda e: e**0.5, lambda e: e**2)))
            ax1.legend()

            res1.grid()
            quick_plot(res1, [bin_centers], [pre_residual], marker='.', linestyle='None',
                       labels=[r'Pre Red. $\chi^2=$'f'{pre_red_chi2:.1f}'], color='red', ylabel='Weight. Resid.')
            res1.set_xlim([-1, 0])

            res2.grid()
            quick_plot(res2, [bin_centers], [opt_residual], marker='.', color='purple', linestyle='None',
                       ylabel='Weight. Resid.', labels=[r'Post Red. $\chi^2=$'f'{red_chi2[p]:.1f}'],
                       xlabel=r'Phase $\times 2\pi$')
            res2.set_xlim([-1, 0])
            for b in order_edges[:, p]:
                res2.axvline(b, linestyle='--', color='black')

            # second figure with polynomials:
            if not np.isnan(fit_phis[0]) and not np.isnan(fit_phis[-1]):
                new_x = np.linspace(fit_phis[0] - 0.01, fit_phis[-1] + 0.01, 1000)
            elif not np.isnan(fit_phis[0]):
                new_x = np.linspace(fit_phis[0] - 0.01, fit_phis[-2] + 0.01, 1000)
            elif not np.isnan(fit_phis[-1]):
                new_x = np.linspace(fit_phis[1] - 0.01, fit_phis[-1] + 0.01, 1000)

            ax2.grid()
            ax2.set_ylabel('Deviation from Linear (nm)')


            def e_poly_linear(x):
                b = leg_e(fit_phis[0]) - fit_phis[0] * (leg_e(fit_phis[-1]) - leg_e(fit_phis[0])) / (
                        fit_phis[-1] - fit_phis[0])
                return (leg_e(fit_phis[-1]) - leg_e(fit_phis[0])) / (fit_phis[-1] - fit_phis[0]) * x + b


            masked_reg = engine.eV_to_wave(leg_e(new_x) * energy[-1, p] * u.eV)
            masked_lin = engine.eV_to_wave(e_poly_linear(new_x) * energy[-1, p] * u.eV)
            deviation = masked_reg - masked_lin
            ax2.plot(new_x, deviation, color='k')
            for m, i in enumerate(fit_phis):
                ax2.plot(i, engine.eV_to_wave(leg_e(i) * energy[-1, p] * u.eV) - engine.eV_to_wave(
                    e_poly_linear(i) * energy[-1, p] * u.eV), '.',
                         markersize=10, label=f'Order {spectro.orders[::-1][m]}')
            ax2.legend()

            ax2_2.grid()
            ax2_2.set_ylabel('R')
            ax2_2.set_xlabel(r'Phase $\times 2\pi$')
            s_eval = leg_s(leg_e(new_x))
            R = sig_to_R(s_eval, new_x)
            ax2_2.plot(new_x, R, color='k')
            for m, i in enumerate(fit_phis):
                ax2_2.plot(i, sig_to_R(fit_sigs[m], i), '.', markersize=10, label=f'Order {spectro.orders[::-1][m]}')
            ax2.set_title(
                r'$E(\varphi)=$'f'{fit_e_coef[1]:.2e}P_2+{fit_e_coef[0]:.2f}P_1+{fit_e0:.2f}P_0\n'
                r'$\sigma(E)=$'f'{fit_s_coef[2]:.2e}P_2+{fit_s_coef[1]:.2e}P_1+{fit_s_coef[0]:.2e}P_0'
            )

            ax1.set_xticks([])  # removes axis labels
            ax2.set_xticks([])
            res1.set_xticks([])

            plt.show()

            fit_phi[:, p] = fit_phis

    # plot all decent pixel models together:
    idxs = np.where(red_chi2 > redchi_val)
    print(f'Number of pixels with red-chi2 less than {redchi_val}: {sim.npix - len(idxs[0])}')

    gausses[:, [idxs]] = 0
    gausses[gausses < 0.01] = 0.01  # TODO change back
    plt.imshow(gausses[::-1], extent=[1, sim.npix, bin_centers[0], bin_centers[-1]], aspect='auto', norm=LogNorm())
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Photon Count')
    plt.title("Fit MSF Model")
    plt.xlabel("Pixel Index")
    plt.ylabel(r"Phase ($\times \pi /2$)")
    plt.tight_layout()
    plt.show()

    # TODO DELETE
    # gausses_i[gausses_i < 0.01] = 0.01
    # #C0 = colorConverter.to_rgba('C0')
    # #C1 = colorConverter.to_rgba('C1')
    # #C2 = colorConverter.to_rgba('C2')
    # #C3 = colorConverter.to_rgba('C3')
    # 
    # cm0 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap', ['none', 'C0'],256)
    # cm1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['none', 'C1'],256)
    # cm2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap3',['none', 'C2'],256)
    # cm3 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap4',['none', 'C3'],256)
    # 
    # cm0._init()
    # cm1._init()
    # cm2._init()
    # cm3._init()
    # 
    # alphas = np.linspace(0, 0.8, cm0.N + 3)
    # cm0._lut[:, -1] = alphas
    # cm1._lut[:, -1] = alphas
    # cm2._lut[:, -1] = alphas
    # cm3._lut[:, -1] = alphas
    # 
    # img0 = plt.imshow(gausses_i[:, 0, :], interpolation='nearest', cmap=cm0, origin='lower', aspect='auto', norm=LogNorm())#, extent=[1, sim.npix, bin_centers[0], bin_centers[-1]])
    # img1 = plt.imshow(gausses_i[:, 1, :], interpolation='nearest', cmap=cm1, origin='lower', aspect='auto', norm=LogNorm())#, extent=[1, sim.npix, bin_centers[0], bin_centers[-1]])
    # img2 = plt.imshow(gausses_i[:, 2, :], interpolation='nearest', cmap=cm2, origin='lower', aspect='auto', norm=LogNorm())#, extent=[1, sim.npix, bin_centers[0], bin_centers[-1]])
    # img3 = plt.imshow(gausses_i[:, 3, :], interpolation='nearest', cmap=cm3, origin='lower', aspect='auto', norm=LogNorm())#, extent=[1, sim.npix, bin_centers[0], bin_centers[-1]])
    # 
    # plt.yticks([0, 200, 400, 600, 800, 1000],[-1, -0.8, -0.6, -0.4, -0.2, 0])
    # plt.title("Virtual Pixel Model")
    # plt.xlabel("Pixel Index")
    # plt.ylabel(r"Phase ($\times \pi /2$)")
    # 
    # c3 = mpatches.Patch(color='C3', label='Order 4')
    # c2 = mpatches.Patch(color='C2', label='Order 5')
    # c1 = mpatches.Patch(color='C1', label='Order 6')
    # c0 = mpatches.Patch(color='C0', label='Order 7')
    # 
    # plt.legend(handles=[c3, c2, c1, c0])
    # 
    # plt.tight_layout()
    # plt.show()

    plt.grid()
    plt.plot(detector.pixel_indices[red_chi2 < 1e6], red_chi2[red_chi2 < 1e6], '.', markersize=3)
    plt.axhline(redchi_val, color='red', label='Cut-off')
    plt.semilogy()
    plt.title(r'Weighted Reduced $\chi^2$ for all pixels')
    plt.ylabel(r'Weighted Red. $\chi^2$')
    plt.xlabel('Pixel Index')
    plt.legend()
    plt.show()

    logging.info(f"Finished fitting all pixels across all orders.")

    # assign bin edges, covariance matrices, bin centers, and simulation settings to MSF class and save:
    covariance = np.nan_to_num(covariance)
    msf = MKIDSpreadFunction(bin_edges=order_edges, cov_matrix=covariance, waves=fit_phi, sim_settings=sim)
    msf_file = f'{args.output_dir}/msf_R0{sim.designR0}_{sim.pixellim}.npz'
    msf.save(msf_file)
    logging.info(f'\nSaved MSF bin edges and covariance matrix to {msf_file}.')
    logging.info(f'\nTotal script runtime: {((time.perf_counter() - tic) / 60):.2f} min.')
    # ==================================================================================================================
    # MSF EXTRACTION ENDS
    # ==================================================================================================================

    # ==================================================================================================================
    # DEBUGGING PLOTS
    # ==================================================================================================================
    # retrieving the blazed calibration spectrum shape:
    # spectra = get_spectrum(sim.type_spectra)
    # blazed_spectrum, _, _ = eng.blaze(wave, spectra)
    # blazed_spectrum = np.nan_to_num(blazed_spectrum)
    # pix_leftedge = spectro.pixel_wavelengths(edge='left').to(u.nm).value
    # blaze_shape = [eng.lambda_to_pixel_space(wave, blazed_spectrum[i], pix_leftedge[i]) for i in range(nord)][::-1]
    # blaze_shape = np.nan_to_num(blaze_shape)
    # blaze_shape /= np.max(blaze_shape)  # normalize max to 1

    # # plot unblazed, unshaped calibration spectrum (should be mostly flat line):
    # fig2, ax2 = plt.subplots(1, 1)
    # quick_plot(ax2, centers_nm, amp_counts / blaze_shape, labels=[f'O{i}' for i in spectro.orders[::-1]],
    #            first=True, title='Calibration Spectrum (Blaze Divided Out)', xlabel='Phase',
    #            ylabel='Photon Count', linestyle='None', marker='.')
    # ax2.set_yscale('log')
    plt.show()
