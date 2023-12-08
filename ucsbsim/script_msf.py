import numpy as np
from numpy.polynomial.legendre import Legendre
import matplotlib.pyplot as plt
import scipy.signal
import scipy
import scipy.interpolate as interp
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import astropy.units as u
import time
from datetime import datetime as dt
import argparse
import logging
import os
from lmfit import Parameters, minimize
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
        miss_per=0.1,
        reg_per=0.15,
        degree=2,
        e_domain=[-1, 0]
):
    """
    :param phi_guess: n_ord guess values for phases
    :param e_guess: n_ord guess values for energies of given phases
    :param s_guess: n_ord guess values for sigmas of given energies
    :param amp_guess: n_ord guess values for amps
    :param miss_ord: as list, the orders that are probably missing/0
    :param miss_per: the percentage to add to the threshold for missing orders
    :param reg_per: the percentage to add to the threshold for existing orders
    :param degree: degree to use for polynomial fitting, default 2nd order
    :param e_domain: domain for the legendre polynomial
    :return: an lmfit Parameter object with the populated parameters and guess values
    """
    parameters = Parameters()

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
    parameters.add(name=f'e0', value=e_coefs[0])# , min=2, max=3)  # , min=e0_bound[0], max=e0_bound[-1])
    parameters.add(name=f'e1', value=e_coefs[1])#, min=-3, max=-1.4)  # , min=e1_bound[0], max=e1_bound[-1])
    parameters.add(name=f'e2', value=e_coefs[2])#, min=-1e-5, max=1e-5)  # , min=e2_bound[0], max=e2_bound[-1])  # the curvation factor is small

    parameters.add(name=f's0', value=s_coefs[0])#, min=0.015, max=0.04)  # , min=s0_bound[0], max=s0_bound[-1])
    parameters.add(name=f's1', value=s_coefs[1])#, min=1e-15, max=1e-3)  # , min=s1_bound[0], max=s1_bound[-1])
    parameters.add(name=f's2', value=s_coefs[2])#, min=-1e-3, max=1e-3)  # , min=-0.01, max=0.01)  # the curvation coef is of order the others

    return parameters, e_coefs, s_coefs


def fit_func(
        params,
        x_phases,
        y_counts,
        orders,
        pix,
        res_mask=None,
        degree=2,
        e_domain=[-1, 0],
        s_domain=None,
        evaluate=None,
        plot=False
):
    have_roots = True
    # turn dictionary of param values into list:
    param_vals = list(params.valuesdict().values())

    # by how I ordered it, the first param is always the phi_0 value
    phi_0 = param_vals[0]

    # the 1 through n_ord + 1 params are the amplitudes:
    amps = np.array(param_vals[1:nord + 1])

    # the next poly_degree+1 params are the e coefficients:
    e0, e1, e2 = param_vals[nord + 1:nord + 2 + degree]

    # pass coef parameters to energy poly and get energies at each phase center:
    e_poly = Legendre((e0, e1, e2), domain=e_domain)

    # calculate the other phi_m based on phi_0:
    m0 = orders[0]
    other_orders = orders[1:][::-1]  # flipping so that orders are in ascending *lambda* (like the rest)
    grating_eq = other_orders / m0 * e_poly(phi_0)
    try:
        phis = []
        for i in grating_eq:
            roots = (e_poly - i).roots()
            if all(roots < 0):
                phis.append(roots[(-1 < roots) & (roots < phi_0)][0])
            else:
                phis.append(roots.min())
    except IndexError:
        have_roots = False
        residual = np.full_like(y_counts, np.max(y_counts) / np.sqrt(np.max(y_counts)))

    if np.iscomplex(phis).any():
        residual = np.full_like(y_counts, np.max(y_counts) / np.sqrt(np.max(y_counts)))
        model = np.zeros_like(x_phases)

    elif have_roots:
        phis = np.array(phis)
        phis = np.append(phis, phi_0)
        # the last poly_degree+1 params are the sigma coefficients:
        s0, s1, s2 = param_vals[nord + 2 + degree:]

        # pass coef parameters to sigma poly and get sigmas at each phase center:
        sigs = Legendre((s0, s1, s2), domain=s_domain)(e_poly(phis))

        # put the phi, sigma, and amps together for Gaussian model:
        gauss_i = np.nan_to_num(gauss(x=x_phases, mu=phis[:, None], sig=sigs[:, None], A=amps[:, None]), nan=0)
        model = np.array([np.sum(gauss_i, axis=1)]).flatten()

        # get the residuals and weighted reduced chi^2:
        residual = np.divide(
            model - y_counts,
            np.sqrt(y_counts),
            where=y_counts != 0)  # TODO change back
        residual[y_counts == 0] = 0

    #if res_mask is not None:
    #    residual[~res_mask] = np.nan

    # WARNING, WILL PLOT HUNDREDS IF FIT SUCKS/CAN'T CONVERGE
    if plot:
        mask = np.isfinite(residual)
        N = mask.sum()
        N_dof = N - len(param_vals)
        red_chi2 = np.sum(residual[mask] ** 2) / N_dof

        fig = plt.figure(1)
        ax = fig.add_axes((.1, .3, .8, .6))
        ax.plot(x_phases, y_counts, label='Data')
        ax.plot(x_phases, model, label='Model')
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

    if evaluate is not None:
        return gauss(x=evaluate, mu=phis[:, None], sig=sigs[:, None], A=amps[:, None])
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
    e_coefs = np.array([params[f'e{c}'].value for c in range(degree + 1)])
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
    :return: quadratic formula results for ax^2 + bx + c = 0
    """
    return ((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)), ((-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))


def gauss_intersect(mu, sig, A):
    """
    :param mu: 2 means of Gaussians
    :param sig: 2 sigmas of Gaussians
    :param A: 2 amplitudes of Gaussians
    :return: analytic calculation of the intersection point between 2 1D Gaussian functions
    """
    n = len(mu)
    if n != len(sig) or n != len(A):
        raise ValueError("mu, sig, and A must all be the same size.")
    a = 1 / sig[0] ** 2 - 1 / sig[1] ** 2
    b = 2 * mu[1] / sig[1] ** 2 - 2 * mu[0] / sig[0] ** 2
    c = (mu[0] / sig[0]) ** 2 - (mu[1] / sig[1]) ** 2 - 2 * np.log(A[0] / A[1])
    solp, soln = quad_formula(a=a, b=b, c=c)
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
    #avg_peak_sep = np.average(np.diff(sim_phase, axis=0))

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
    e_init = np.zeros([3, sim.npix])
    s_init = np.zeros([3, sim.npix])
    phi_init = np.zeros([4, sim.npix])
    amp_init = np.zeros([nord, sim.npix])

    # create empty arrays to hold phi, sigma, and amp values for gaussian model:
    fit_e = np.empty([3, sim.npix])  # the e coefficients
    phis = np.empty([nord, sim.npix])  # the evaluated order phase centers
    fit_sig = np.empty([3, sim.npix])  # the sig coefficients
    sigs = np.empty([nord, sim.npix])  # the evaluated order sigmas
    amps = np.empty([nord, sim.npix])  # the amplitudes of each order
    gausses = np.empty([1000, sim.npix])  # the entire n_ord Gaussian model summed

    # for use in recovery of other phis:
    m0 = spectro.orders[0]
    other_orders = spectro.orders[1:][::-1]

    # create empty array for order bin edges:
    order_edges = np.empty([nord + 1, sim.npix])
    order_edges[0, :] = -1

    # separate index for list indexing because param objects can only be in lists:
    n = 0

    # TODO change pixel indices here
    #pixels = [107, 1635]
    snr = 2
    # do the lmfit:
    for p in pixels:
        print(p)
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

        phi_init[:, p] = bin_centers[peaks_nord]
        amp_init[:, p] = bin_counts[peaks_nord, p]

        # obtain Parameter object:
        params, e_coefs, s_coefs = init_params(phi_guess=phi_init[:, p], e_guess=energy[:, p], s_guess=sigmas[:, p],
                                               amp_guess=amp_init[:, p], miss_ord=miss_ord)
        e_init[:, p] = e_coefs
        s_init[:, p] = s_coefs

        opt_params = minimize(
                fit_func,
                params,  # params
                args=(
                    bin_centers,  # x_phases
                    bin_counts[:, p],  # y_counts
                    spectro.orders,  # orders
                    p,  # pix
                    bin_counts[:, p] > snr ** 2,  # res_mask
                    2,  # degree
                    [-1, 0],  # e_domain
                    [energy[0, p], energy[-1, p]],  # s_domain
                    None,  # evaluate
                    False,  # plot
                ),
                nan_policy='omit'
            )
        red_chi2[p] = opt_params.redchi

        # extract the successfully fit parameters:
        phis[-1, p], fit_e[:, p], fit_sig[:, p], amps[:, p] = extract_params(opt_params.params, nord)

        e_poly = Legendre(fit_e[:, p], domain=[-1, 0], window=[-1,1])
        grating_eq = other_orders / m0 * e_poly(phis[-1, p])
        phis_i = []
        for i in grating_eq:
            roots = (e_poly - i).roots()
            if all(roots < 0):
                phis_i.append(roots[(-1 < roots) & (roots < phis[-1, p])][0])
            else:
                phis_i.append(roots.min())
        phis[:-1, p] = phis_i

        s_poly = Legendre(fit_sig[:, p], domain=[energy[0, p], energy[-1, p]], window=[-1,1])
        sigs[:, p] = s_poly(e_poly(phis[:, p]))

        # get the order bin edges:
        idx = np.argwhere(amps[:, p] != 0)
        for h, i in enumerate(idx[:-1]):
            order_edges[i + 1, p] = gauss_intersect(phis[[i, idx[h + 1]], p], sigs[[i, idx[h + 1]], p],
                                                    amps[[i, idx[h + 1]], p])

        # store model to array:
        x = np.linspace(-1, 0, 1000)
        gausses_i = fit_func(params=opt_params.params, x_phases=bin_centers, y_counts=bin_counts[:, p],
                             s_domain=[energy[0, p], energy[-1, p]], orders=spectro.orders, pix=p, evaluate=x)
        #gausses_i[:, amps[:, p] < 25] = 0
        gausses[:, p] = np.sum(gausses_i, axis=1)

        # plot the individual pixels:
        if red_chi2[p] > 1e6:  # TODO change back
            param_vals = list(opt_params.params.valuesdict().values())

            # get the post residuals and weighted reduced chi^2:
            residual = fit_func(
                params=opt_params.params,
                x_phases=bin_centers,
                y_counts=bin_counts[:, p],
                s_domain=[energy[0, p], energy[-1, p]],
                orders=spectro.orders,
                pix=p,
                res_mask=bin_counts[:, p] > snr ** 2
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

            # first figure with model and pre/post residual:
            e_p = Legendre(e_init[:, p], domain=[-1, 0], window=[-1,1])

            grating_eq = other_orders / m0 * e_p(phi_init[-1, p])
            phis_p = []
            for i in grating_eq:
                roots = (e_p - i).roots()
                if all(roots < 0):
                    phis_p.append(roots[(-1 < roots) & (roots < phi_init[-1, p])][0])
                else:
                    phis_p.append(roots.min())
            phis_p = np.array(phis_p)
            phis_p = np.append(phis_p, phi_init[-1, p])

            sigs_p = Legendre(s_init[:, p], domain=[energy[0, p], energy[-1, p]], window=[-1,1])(e_p(phis_p))

            amps_p = amp_init[:, p]

            initial = np.sum(gauss(bin_centers, phis_p[:, None], sigs_p[:, None], amps_p[:, None]), axis=1)

            # get the pre residuals and weighted reduced chi^2:
            pre_mask = bin_counts[:, p] > 0  # TODO change back
            pre_residual = np.divide(
                initial - bin_counts[:, p],
                np.sqrt(bin_counts[:, p]),
                where=pre_mask)
            pre_residual[~pre_mask] = 0  # TODO change back
            N = np.isfinite(pre_residual).sum()
            N_dof2 = N - len(param_vals)
            N_dof2 = 1 if N_dof2 == 0 else N_dof2
            pre_red_chi2 = np.sum(pre_residual[np.isfinite(pre_residual)] ** 2) / N_dof2

            ax1.grid()
            ax1.plot(bin_centers, bin_counts[:, p], 'k', label='Data')
            ax1.fill_between(bin_centers, 0, bin_counts[:, p], color='k')
            quick_plot(ax1, [x for i in range(nord)], gauss(x, phis_p[:, None], sigs_p[:, None], amps_p[:, None]).T,
                       labels=['Init. Guess'] + ['_nolegend_' for o in spectro.orders[:-1]], color='gray')
            quick_plot(ax1, [x for i in range(nord)], gausses_i.T, labels=[f'Order {i}' for i in spectro.orders[::-1]],
                       title=f'Least-Squares Fit', xlabel=r'Phase $\times 2\pi$', ylabel='Photon Count')
            for b in order_edges[:-1, p]:
                ax1.axvline(b, linestyle='--', color='black')
            ax1.axvline(order_edges[-1, p], linestyle='--', color='black', label='Order Edges')
            ax1.set_xlim([-1, 0])
            #ax1.set_yscale(matplotlib.scale.FuncScale(ax1, (lambda e: e**0.5, lambda e: e**2)))
            #ax1.set_ylim(bottom=25)
            ax1.legend()

            res1.grid()
            quick_plot(res1, [bin_centers], [pre_residual], marker='.', linestyle='None',
                       labels=[r'Pre Red. $\chi^2=$'f'{pre_red_chi2:.1f}'], color='red', ylabel='Weight. Resid.')
            res1.set_xlim([-1, 0])

            res2.grid()
            quick_plot(res2, [bin_centers], [residual], marker='.', color='purple', linestyle='None',
                       ylabel='Weight. Resid.', labels=[r'Post Red. $\chi^2=$'f'{red_chi2[p]:.1f}'],
                       xlabel=r'Phase $\times 2\pi$')
            res2.set_xlim([-1, 0])
            for b in order_edges[:, p]:
                res2.axvline(b, linestyle='--', color='black')

            # ax1.set_yscale('log')
            # ax1.set_ylim(bottom=0.1)

            # second figure with polynomials:
            if not np.isnan(phis[0, p]) and not np.isnan(phis[-1, p]):
                new_x = np.linspace(phis[0, p] - 0.01, phis[-1, p] + 0.01, 1000)
            elif not np.isnan(phis[0, p]):
                new_x = np.linspace(phis[0, p] - 0.01, phis[-2, p] + 0.01, 1000)
            elif not np.isnan(phis[-1, p]):
                new_x = np.linspace(phis[1, p] - 0.01, phis[-1, p] + 0.01, 1000)

            ax2.grid()
            ax2.set_ylabel('Deviation from Linear (nm)')


            def e_poly_linear(x):
                b = e_poly(phis[0, p]) - phis[0, p] * (e_poly(phis[-1, p]) - e_poly(phis[0, p])) / (
                        phis[-1, p] - phis[0, p])
                return (e_poly(phis[-1, p]) - e_poly(phis[0, p])) / (phis[-1, p] - phis[0, p]) * x + b


            masked_reg = engine.eV_to_wave(e_poly(new_x) * u.eV)
            masked_lin = engine.eV_to_wave(e_poly_linear(new_x) * u.eV)
            deviation = masked_reg - masked_lin
            ax2.plot(new_x, deviation, color='k')
            for m, i in enumerate(phis[:, p]):
                ax2.plot(i, engine.eV_to_wave(e_poly(i) * u.eV) - engine.eV_to_wave(e_poly_linear(i) * u.eV), '.',
                         markersize=10, label=f'Order {spectro.orders[::-1][m]}')
            ax2.legend()

            ax2_2.grid()
            ax2_2.set_ylabel('R')
            ax2_2.set_xlabel(r'Phase $\times 2\pi$')
            s_eval = s_poly(e_poly(new_x))
            R = sig_to_R(s_eval, new_x)
            ax2_2.plot(new_x, R, color='k')
            for m, i in enumerate(phis[:, p]):
                ax2_2.plot(i, sig_to_R(sigs[m, p], i), '.', markersize=10, label=f'Order {spectro.orders[::-1][m]}')
            ax2.set_title(
                r'$E(\varphi)=$'f'{fit_e[2, p]:.2e}P_2+{fit_e[1, p]:.1e}P_1+{fit_e[0, p]:.2f}P_0\n'
                r'$\sigma(\varphi)=$'f'{fit_sig[2, p]:.2e}P_2+{fit_sig[1, p]:.2e}P_1+{fit_sig[0, p]:.2e}P_0'
            )

            ax1.set_xticks([])  # removes axis labels
            ax2.set_xticks([])
            res1.set_xticks([])

            plt.show()

    # plot all decent pixel models together:
    idxs = np.where(red_chi2 > 10)
    print(r'Number of pixels with reduced $\chi^2$ less than 10:', sim.npix - len(idxs[0]))

    gausses[:, [idxs]] = 0
    gausses[gausses < 4] = 0.1  # TODO change back
    plt.imshow(gausses[::-1], extent=[1, sim.npix, bin_centers[0], bin_centers[-1]], aspect='auto', norm=LogNorm())
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Photon Count')
    plt.title("Fit MSF Model")
    plt.xlabel("Pixel Index")
    plt.ylabel(r"Phase ($\times \pi /2$)")
    plt.tight_layout()
    plt.show()

    plt.grid()
    plt.plot(detector.pixel_indices, red_chi2, '.', markersize=3)
    plt.axhline(10, color='red', label='Cut-off')
    plt.semilogy()
    plt.title(r'Weighted Reduced $\chi^2$')
    plt.ylabel(r'Weighted Red. $\chi^2$')
    plt.xlabel('Pixel Index')
    plt.legend()
    plt.show()

    # initializing empty arrays for msf products:
    covariance = np.zeros([nord, nord, sim.npix])
    phase_grid = np.linspace(-1, 0, 10000)
    for j in pixels:
        # get the covariance matrices:
        gauss_sum = np.sum(
            gauss(phase_grid, phis[:, j, None], sigs[:, j, None], amps[:, j, None]), axis=0
        ) * (phase_grid[1] - phase_grid[0])

        gauss_int = [
            interp.InterpolatedUnivariateSpline(
                x=phase_grid,
                y=gauss(phase_grid, phis[i, j], sigs[i, j], amps[i, j]),
                k=1,
                ext=1
            ) for i in range(nord)
        ]

        covariance[:, :, j] = [[
            gauss_int[i].integral(order_edges[k, j], order_edges[k + 1, j]) / gauss_sum[i] for k in range(nord)
        ] for i in range(nord)]

    logging.info(f"Finished fitting all pixels across all orders.")

    # assign bin edges, covariance matrices, bin centers, and simulation settings to MSF class and save:
    covariance = np.nan_to_num(covariance)
    msf = MKIDSpreadFunction(bin_edges=order_edges, cov_matrix=covariance, waves=phis, sim_settings=sim)
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
