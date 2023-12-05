import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy
import scipy.interpolate as interp
from matplotlib import colors
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool

# import scipy.interpolate as interp
import astropy.units as u
import time
from datetime import datetime as dt
import argparse
import logging
import os
from lmfit import Parameters, minimize, Model
from synphot import SourceSpectrum
from synphot.models import BlackBodyNorm1D, ConstFlux1D
import itertools

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
-Load the calibration photon table and change wavelengths to phase space.
-Fit Gaussians to each pixel. Some have n_ord peaks, some n_ord-1 (bandpass).
-If calibration spectrum type is known, the relative amplitudes of the spectrum shape multiplied with the blaze, all
 converted to pixel space, are generated.
-Divide this calibration-blaze shape out of the Gaussian amplitude fit to make a flux-normalized spectrum.
 These normalized Gaussian fits are known as the MKID spread function (MSF).
-Use overlap points of the Gaussians to get bin edges for each order.
-Calculate fractional overlap between Gaussians and converts them into an n_ord x n_ord "covariance" matrix for each
 pixel. This matrix details what fraction of each order Gaussian was grouped into another order Gaussian due to binning.
 This will later be used to determine the error band of some extracted spectrum.
-Saves newly obtained bin edges and covariance matrices to files.
"""


def bound_finder(poly, x, y, domain=None, window=None):
    if poly == 'e':
        # coef 2: the curvature term must be small
        e2 = [-1e-5, 1e-5]
        # coef 1: the slope given the curvature term is 0
        (_, e1_min) = np.polynomial.legendre.Legendre.fit(x=x, y=y * 0.5, deg=1, domain=domain, window=window).coef
        (_, e1_max) = np.polynomial.legendre.Legendre.fit(x=x, y=y * 1.5, deg=1, domain=domain, window=window).coef
        e1 = [e1_min, e1_max]
        # coef 0: the y intercept minus the part wrapped up in L_2
        (e0_reg, _) = np.polynomial.legendre.Legendre.fit(x=x, y=y, deg=1, domain=domain, window=window).coef
        e0 = [e0_reg-1, e0_reg+1]
        return e0, e1, e2
    elif poly == 'sig':
        # coef 2: the curvature term, ignore for now
        s2 = None
        # coef 1: the slope given the curvature term is 0
        (_, s1_min, _) = np.polynomial.legendre.Legendre.fit(x=x, y=y * 0.5, deg=2, domain=domain, window=window).coef
        (_, s1_max, _) = np.polynomial.legendre.Legendre.fit(x=x, y=y * 1.5, deg=2, domain=domain, window=window).coef
        s1 = [s1_min, s1_max]
        # coef 0: the y intercept minus the part wrapped up in L_2
        (s0_reg, _, _) = np.polynomial.legendre.Legendre.fit(x=x, y=y, deg=2, domain=domain, window=window).coef
        s0 = [s0_reg-1000, s0_reg+1000]
        return s0, s1, s2
    else:
        raise ValueError('Other polynomials not currently supported.')


def sig_to_R(sig, lam):
    dlam = sig * 2 * np.sqrt(2 * np.log(2))
    R = lam / dlam
    return np.abs(R)


def param_guess_all(phi_guess, energy_guess, sigma_guess, amp_guess, pixels, nord):
    """
    :param phi_guess: n_ord guess values for phases
    :param energy_guess: n_ord guess values for energies of given phases
    :param sigma_guess: n_ord guess values for sigmas of given energies
    :param amp_guess: n_ord guess values for amps
    :param pixels: all pixel indices
    :param nord: number of orders
    :return: an lmfit Parameter object
    """
    params = Parameters()
    # params being added separate in loops to make it easier to parse later:
    for j in pixels:
        params.add(f'P{j}_phi_0', phi_guess[-1, j], min=-1, max=0)

    for i in range(nord):
        amp_coefs = np.polynomial.Legendre.fit(pixels, amp_guess[i], 3, domain=[pixels[0], pixels[-1]]).coef
        for c in range(4):
            params.add(f'O{i}_amp{c}', amp_coefs[c])

    # use polyfit with the desired degrees on the theoretical data to get guess coefs
    for j in pixels:
        energy_coefs = np.polynomial.Polynomial.fit(
            phi_guess[:, j],
            energy_guess[:, j],
            2,
            domain=[phi_guess[0, j], phi_guess[-1, j]]
        ).coef
        for c in range(3):
            params.add(f'P{j}_energy{c}', energy_coefs[c])

    for j in pixels:
        sigma_coefs = np.polynomial.Polynomial.fit(
            energy_guess[:, j],
            sigma_guess[:, j],
            2,
            domain=[energy_guess[0, j], energy_guess[-1, j]]
        ).coef
        for c in range(3):
            params.add(f'P{j}_sigma{c}', sigma_coefs[c])

    return params


def param_guess_1pix(
        phi_guess,
        e_guess,
        s_guess,
        amp_guess,
        miss_ord=None,
        thresh=2,
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
    :param thresh: the threshold of the minimum # of counts
    :param miss_per: the percentage to add to the threshold for missing orders
    :param reg_per: the percentage to add to the threshold for existing orders
    :param degree: degree to use for polynomial fitting, default 2nd order
    :param e_domain: domain for the legendre polynomial
    :return: an lmfit Parameter object with the populated parameters and guess values
    """
    parameters = Parameters()

    # use Legendre polyfitting on the theoretical/peak-finding data to get guess coefs
    e_coefs = np.polynomial.legendre.Legendre.fit(
        x=phi_guess,
        y=e_guess,
        deg=degree,
        domain=e_domain,
        window=[e_guess[0], e_guess[-1]]
    ).coef
    (e0_bound, e1_bound, e2_bound) = bound_finder('e', phi_guess, e_guess, domain=e_domain,
                                                  window=[e_guess[0], e_guess[-1]])

    s_coefs = np.polynomial.legendre.Legendre.fit(
        x=e_guess,
        y=s_guess,
        deg=degree,
        domain=[e_guess[0], e_guess[-1]],
        window=[s_guess[0], s_guess[-1]]
    ).coef
    (s0_bound, s1_bound, s2_bound) = bound_finder('sig', e_guess, s_guess, domain=[e_guess[0], e_guess[-1]],
                                                  window=[s_guess[0], s_guess[-1]])

    # add phi_0s to params object:
    parameters.add(name=f'phi_0', value=phi_guess[-1], min=phi_guess[-1] - 0.2, max=phi_guess[-1] + 0.2)

    # add amplitudes to params object:
    if miss_ord is None:  # no orders were identified as 0 flux
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
                # try:
                parameters.add(name=f'O{i}_amp', value=amp_guess[i], vary=False)
                # except ValueError:
                #    parameters.add(name=f'O{i}_amp', value=0, vary=False)
            else:
                parameters.add(
                    name=f'O{i}_amp',
                    value=amp_guess[i],
                    min=amp_guess[i] * (1 - reg_per),
                    max=amp_guess[i] * (1 + reg_per)
                )

    # add polynomial coefficients to params object:
    parameters.add(name=f'e0', value=e_coefs[0])#, min=e0_bound[0], max=e0_bound[-1])
    parameters.add(name=f'e1', value=e_coefs[1])#, min=e1_bound[0], max=e1_bound[-1])
    parameters.add(name=f'e2', value=e_coefs[2])#, min=e2_bound[0], max=e2_bound[-1])  # the curvation factor is small

    parameters.add(name=f's0', value=s_coefs[0])#, min=s0_bound[0], max=s0_bound[-1])
    parameters.add(name=f's1', value=s_coefs[1])#, min=s1_bound[0], max=s1_bound[-1])
    parameters.add(name=f's2', value=s_coefs[2])  # , min=-0.01, max=0.01)  # the curvation coef is of order the others

    return parameters, e_coefs, s_coefs


def fit_func_all(params, x_phases, y_counts, orders, pixels):
    """
    :param params: lmfit Parameter object
    :param x_phases: phases of the bins
    :param y_counts: counts of the bins
    :param orders: orders in spectrograph
    :param pixels: pixels indices in array
    :return: residuals between data and model
    """
    phi_0, amp_coefs, e_coefs, s_coefs = param_extract(params, len(pixels), len(orders))

    # calculate the other order centers based on the m0:
    phis = phi_from_m0(orders[0], orders[1:], (e_coefs[0], e_coefs[1], e_coefs[2]))

    s0, s1, s2 = s_coefs[0], s_coefs[1], s_coefs[2]
    sigs = [np.polynomial.legendre.Legendre((s0[j], s1[j], s2[j]))(phis[:, j]) for j in pixels].T

    a0, a1, a2, a3 = amp_coefs[0], amp_coefs[1], amp_coefs[2], amp_coefs[3]
    amps = [np.polynomial.legendre.Legendre((a0[i], a1[i], a2[i], a3[i]))(pixels) for i in range(len(orders))]

    model = np.empty([len(x_phases), len(pixels)])
    gauss_i = [gauss(x_phases, phis[:, j, None], sigs[:, j, None], amps[:, j, None]) for j in pixels]
    model = np.array([np.sum(gauss, axis=1)])

    return model.flatten() - y_counts.flatten()


def fit_func_1pix(
        params,
        x_phases,
        y_counts,
        orders,
        pix,
        res_mask=None,
        degree=2,
        e_domain=[-1, 0],
        e_window=None,
        s_window=None,
        evalu=None,
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
    e_poly = np.polynomial.legendre.Legendre((e0, e1, e2), domain=e_domain, window=e_window)

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

    elif have_roots:
        phis = np.array(phis)
        phis = np.append(phis, phi_0)
        # the last poly_degree+1 params are the sigma coefficients:
        s0, s1, s2 = param_vals[nord + 2 + degree:]

        # pass coef parameters to sigma poly and get sigmas at each phase center:
        sigs = np.polynomial.legendre.Legendre((s0, s1, s2), domain=e_window, window=s_window)(e_poly(phis))

        # put the phi, sigma, and amps together for Gaussian model:
        gauss_i = np.nan_to_num(gauss(x=x_phases, mu=phis[:, None], sig=sigs[:, None], A=amps[:, None]), nan=0)
        model = np.array([np.sum(gauss_i, axis=1)]).flatten()

        # get the residuals and weighted reduced chi^2:
        residual = np.divide(
            model - y_counts,
            np.sqrt(y_counts),
            where=res_mask)

    if res_mask is not None:
        residual[~res_mask] = np.nan

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
        plt.tight_layout()
        plt.show()

    if evalu is not None:
        return gauss(x=evalu, mu=phis[:, None], sig=sigs[:, None], A=amps[:, None])
    else:
        return residual


def param_extract_1pix(params, n_ord, degree=2):
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


def param_extract(params, npix, nord):
    """
    :param params: Parameter object
    :param npix: number of pixels
    :param nord: number of orders
    :return: the extracted parameters
    """
    # turn dictionary of param values into list:
    param_vals = list(params.valuesdict().values())

    # the m0 order center phases are the first n_pix params:
    phi_0 = np.array(param_vals[:npix])

    # amp coefs are the next 4*nord params:
    amp_coefs = np.array(param_vals[npix:npix + 4 * nord]).reshape([4, nord])

    # e coefs are the next 3*npix params:
    e_coefs = np.array(param_vals[npix + 4 * nord:npix + 4 * nord + 3 * npix]).reshape([3, npix])

    # s coefs are the remaining 3*npix params:
    s_coefs = np.array(param_vals[npix + 4 * nord + 3 * npix:]).reshape([3, npix])

    return phi_0, amp_coefs, e_coefs, s_coefs


def gauss(x, mu, sig, A):
    """
    :param x: wavelength
    :param mu: mean
    :param sig: sigma
    :param A: amplitude
    :return: value of the Gaussian
    """
    return (A * np.exp(- (x - mu) ** 2. / (2. * sig ** 2.))).T


def _quad_formula(a, b, c):
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
    solp, soln = _quad_formula(a=a, b=b, c=c)
    if mu[0] < solp < mu[1]:
        return solp
    elif mu[0] < soln < mu[1]:
        return soln
    else:
        return np.nan


def nearest_idx(array, value):
    return (np.abs(array - value)).argmin()


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
    energy = np.nan_to_num(engine.wave_to_eV(spectro.pixel_wavelengths().to(u.nm)[::-1]).value)
    sig_start = wave_to_phase(spectro.pixel_wavelengths().to(u.nm)[::-1] - (detector.mkid_resolution_width(
        spectro.pixel_wavelengths().to(u.nm)[::-1], pixels) / (2 * np.log(2))) / 2,
                              minwave=sim.minwave, maxwave=sim.maxwave)
    sig_end = wave_to_phase(spectro.pixel_wavelengths().to(u.nm)[::-1] + (detector.mkid_resolution_width(
        spectro.pixel_wavelengths().to(u.nm)[::-1], pixels) / (2 * np.log(2))) / 2,
                            minwave=sim.minwave, maxwave=sim.maxwave)
    sigmas = sig_end - sig_start  # for all, flip order axis to be in ascending phase/lambda

    # ==================================================================================================================
    # MSF EXTRACTION STARTS
    # ==================================================================================================================
    # generating bins and building initial histogram for every pixel:
    n_bins = engine.n_bins(sparse_pixel, method="rice")
    bin_edges = np.linspace(args.bin_range[0], args.bin_range[1], n_bins + 1, endpoint=True)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    bin_counts = np.zeros([n_bins, sim.npix])
    for j in pixels:
        bin_counts[:, j], _ = np.histogram(photons_pixel[j], bins=bin_edges)

    # create lists/arrays to place loop values, param objects can only go in lists:
    used_pix = []
    params = []
    opt_params = []
    red_chi2 = np.full([sim.npix], fill_value=1e6)
    e_init = np.zeros([3, sim.npix])
    s_init = np.zeros([3, sim.npix])
    phi_init = np.zeros([4, sim.npix])
    amp_init = np.zeros([nord, sim.npix])

    # separate index for list indexing because param objects can only be in lists:
    n = 0

    # TODO change pixel indices here
    #pixels = [96]
    #pixels = range(600,610)
    thresh = 25
    # do the lmfit:
    for p in pixels:
        # print(p)
        # use find peaks to get phi/amplitude guesses IF there are at least n_ord peaks,
        # use default simulation values if there are less than n_ord peaks:
        peaks, _ = scipy.signal.find_peaks(bin_counts[:, p], distance=int(0.1 * n_bins), height=thresh)
        if len(peaks) >= nord:
            idx = np.argsort(bin_counts[peaks, p])[::-1][:nord]
            peaks_nord = np.sort(peaks[idx])
            miss_ord = None
        else:
            idx = np.argsort(bin_counts[peaks, p])[::-1]
            peaks_order = np.sort(peaks[idx])
            if p <= 350 and len(peaks_order) == 3:
                peaks_nord = np.array([
                    peaks_order[0],
                    peaks_order[1],
                    int((peaks_order[1] + peaks_order[2]) / 2),
                    peaks_order[2]]
                )
                miss_ord = [2]
            elif p <= 1300 and len(peaks_order) == 2:
                peaks_nord = np.array([
                    int(peaks_order[0] - (peaks_order[1] - peaks_order[0]) / 2),
                    peaks_order[0],
                    int((peaks_order[0] + peaks_order[1]) / 2),
                    peaks_order[1]]
                )
                miss_ord = [0, 2]
            elif 350 < p <= 1300 and len(peaks_order) == 3:
                peaks_nord = np.array([
                    int(peaks_order[0] - (peaks_order[1] - peaks_order[0]) / 2),
                    peaks_order[0],
                    peaks_order[1],
                    peaks_order[2]]
                )
                miss_ord = [0]
            elif p > 1300 and len(peaks_order) == 3:
                peaks_nord = np.array([
                    peaks_order[0],
                    int((peaks_order[0] + peaks_order[1]) / 2),
                    peaks_order[1],
                    peaks_order[2]]
                )
                miss_ord = [1]
            elif p > 1300 and len(peaks_order) == 2:
                peaks_nord = np.array([
                    peaks_order[0],
                    int((peaks_order[0] + peaks_order[1]) / 2),
                    peaks_order[1],
                    int(peaks_order[1] + (peaks_order[1] - peaks_order[0]) / 2)]
                )
                miss_ord = [1, 3]
            else:
                logging.info(f'Pixel {p} will default to simulation guess.')
                peaks_nord = [nearest_idx(bin_centers, sim_phase[i, p]) for i in range(nord)]
                miss_ord = None

        phi_init[:, p] = bin_centers[peaks_nord]
        amp_init[:, p] = bin_counts[peaks_nord, p]

        # obtain Parameter object:
        pars, e_coefs, s_coefs = param_guess_1pix(
            phi_guess=phi_init[:, p],
            e_guess=energy[:, p],
            s_guess=sigmas[:, p],
            amp_guess=amp_init[:, p],
            miss_ord=miss_ord,
            thresh=thresh
        )
        e_init[:, p] = e_coefs
        s_init[:, p] = s_coefs

        # attempt to optimize fit, pass if fails:
        try:
            # with Pool(16) as pool:
            #     results = pool.apply(
            #         minimize, kwds=fcn,
            #         (fit_func_1pix,params[n], (bin_centers,bin_counts[:, pix],spectro.orders, pix,2,False), 'omit'))
            # opt_params.append(results)
            opt_params.append(
                minimize(
                    fit_func_1pix,
                    pars,
                    args=(
                        bin_centers,
                        bin_counts[:, p],
                        spectro.orders,
                        p,
                        bin_counts[:, p] > thresh,
                        2,
                        [-1, 0],
                        [energy[0, p], energy[-1, p]],
                        [sigmas[0, p], sigmas[-1, p]],
                        None,
                        False,
                    ),
                    nan_policy='omit'
                )
            )
            red_chi2[p] = opt_params[n].redchi
            # print(opt_params[n].message)

            # record which pixels were fit "successfully":
            used_pix.append(p)

            # increase index for lists
            n += 1
            params.append(pars)
        except TypeError as e:
            # prints why the fit failed
            logging.info(f'#{p} Failure - {e}')

    # create empty arrays to hold phi, sigma, and amp values for gaussian model:
    fit_e = np.empty([3, sim.npix])
    phis = np.empty([nord, sim.npix])
    fit_sig = np.empty([3, sim.npix])
    sigs = np.empty([nord, sim.npix])
    amps = np.empty([nord, sim.npix])
    gausses = np.empty([1000, sim.npix])

    # for use in recovery of other phis:
    m0 = spectro.orders[0]
    other_orders = spectro.orders[1:][::-1]

    # create empty array for order bin edges:
    order_edges = np.zeros([nord + 1, sim.npix])
    order_edges[0, :] = -1

    for n, j in enumerate(used_pix):
        # extract the successfully fit parameters:
        phis[-1, j], fit_e[:, j], fit_sig[:, j], amps[:, j] = param_extract_1pix(opt_params[n].params, nord)

        e_poly = np.polynomial.legendre.Legendre(fit_e[:, j], domain=[-1, 0], window=[energy[0, j], energy[-1, j]])
        grating_eq = other_orders / m0 * e_poly(phis[-1, j])
        phis_i = []
        for i in grating_eq:
            roots = (e_poly - i).roots()
            if all(roots < 0):
                phis_i.append(roots[(-1 < roots) & (roots < phis[-1, j])][0])
            else:
                phis_i.append(roots.min())
        phis[:-1, j] = phis_i

        s_poly = np.polynomial.legendre.Legendre(fit_sig[:, j], domain=[energy[0, j], energy[-1, j]],
                                                 window=[sigmas[0, j], sigmas[-1, j]])
        sigs[:, j] = s_poly(e_poly(phis[:, j]))

        # get the order bin edges:
        for k, a in enumerate(amps[:-1, j]):
            order_edges[k + 1, j] = gauss_intersect(phis[[k, k + 1], j], sigs[[k, k + 1], j], amps[[k, k + 1], j])

        # store model to array:
        x = np.linspace(-1, 0, 1000)
        gausses_i = fit_func_1pix(
            params=opt_params[n].params,
            x_phases=bin_centers,
            y_counts=bin_counts[:, j],
            orders=spectro.orders,
            pix=j,
            e_window=[energy[0, j], energy[-1, j]],
            s_window=[sigmas[0, j], sigmas[-1, j]],
            evalu=x
        )
        gausses[:, j] = np.sum(gausses_i, axis=1)

        param_vals = list(opt_params[n].params.valuesdict().values())

        # get the post residuals and weighted reduced chi^2:
        residual = fit_func_1pix(
            params=opt_params[n].params,
            x_phases=bin_centers,
            y_counts=bin_counts[:, j],
            orders=spectro.orders,
            pix=j,
            e_window=[energy[0, j], energy[-1, j]],
            s_window=[sigmas[0, j], sigmas[-1, j]],
            res_mask=bin_counts[:, j] > thresh
        )
        # mask = np.isfinite(residual)
        # N = mask.sum()
        # N_dof = N - len(param_vals)
        # red_chi2 = np.sum(residual[~np.isnan(residual)] ** 2) / N_dof

        # plot the individual pixels:
        if red_chi2[j] > 10:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
            axes = axes.ravel()
            ax1 = axes[0]
            ax2 = axes[1]

            plt.suptitle(f'Pixel {j}')

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
            e_j = np.polynomial.legendre.Legendre(e_init[:, j], domain=[-1, 0], window=[energy[0, j], energy[-1, j]])

            grating_eq = other_orders / m0 * e_j(phi_init[-1, j])
            phis_j = []
            for i in grating_eq:
                roots = (e_j - i).roots()
                if all(roots < 0):
                    phis_j.append(roots[(-1 < roots) & (roots < phi_init[-1, j])][0])
                else:
                    phis_j.append(roots.min())
            phis_j = np.array(phis_j)
            phis_j = np.append(phis_j, phi_init[-1, j])

            sigs_j = np.polynomial.legendre.Legendre(s_init[:, j], domain=[energy[0, j], energy[-1, j]],
                                                     window=[sigmas[0, j], sigmas[-1, j]])(e_j(phis_j))

            amps_j = amp_init[:, j]

            initial = np.sum(gauss(bin_centers, phis_j[:, None], sigs_j[:, None], amps_j[:, None]), axis=1)

            # get the pre residuals and weighted reduced chi^2:
            pre_mask = bin_counts[:, j] > 25
            pre_residual = np.divide(
                initial - bin_counts[:, j],
                np.sqrt(bin_counts[:, j]),
                where=pre_mask)
            pre_residual[~pre_mask] = np.nan
            N = np.isfinite(pre_residual).sum()
            N_dof2 = N - len(param_vals)
            N_dof2 = 1 if N_dof2 == 0 else N_dof2
            pre_red_chi2 = np.sum(pre_residual[np.isfinite(pre_residual)] ** 2) / N_dof2

            ax1.grid()
            ax1.plot(bin_centers, bin_counts[:, j], 'k', label='Data')
            ax1.fill_between(bin_centers, 0, bin_counts[:, j], color='k')
            quick_plot(ax1, [x for i in range(nord)], gauss(x, phis_j[:, None], sigs_j[:, None], amps_j[:, None]).T,
                       labels=['Init. Guess'] + ['_nolegend_' for o in spectro.orders[:-1]], color='gray')
            quick_plot(ax1, [x for i in range(nord)], gausses_i.T, labels=[f'Order {i}' for i in spectro.orders[::-1]],
                       title=f'Least-Squares Fit', xlabel=r'Phase $\times 2\pi$', ylabel='Photon Count')
            for b in order_edges[:-1, j]:
                ax1.axvline(b, linestyle='--', color='black')
            ax1.axvline(order_edges[-1, j], linestyle='--', color='black', label='Order Edges')
            ax1.set_xlim([-1, 0])
            ax1.set_ylim(bottom=25)
            ax1.legend()

            res1.grid()
            quick_plot(res1, [bin_centers], [pre_residual], marker='.', linestyle='None',
                       labels=[r'Pre Red. $\chi^2=$'f'{pre_red_chi2:.1f}'], color='red', ylabel='Weight. Resid.')
            res1.set_xlim([-1, 0])

            res2.grid()
            quick_plot(res2, [bin_centers], [residual], marker='.', color='purple', linestyle='None',
                       ylabel='Weight. Resid.', labels=[r'Post Red. $\chi^2=$'f'{red_chi2[j]:.1f}'],
                       xlabel=r'Phase $\times 2\pi$')
            res2.set_xlim([-1, 0])
            for b in order_edges[:, j]:
                res2.axvline(b, linestyle='--', color='black')

            # ax1.set_yscale('log')
            # ax1.set_ylim(bottom=0.1)

            # second figure with polynomials:
            if not np.isnan(phis[0, j]) and not np.isnan(phis[-1, j]):
                new_x = np.linspace(phis[0, j] - 0.01, phis[-1, j] + 0.01, 1000)
            elif not np.isnan(phis[0, j]):
                new_x = np.linspace(phis[0, j] - 0.01, phis[-2, j] + 0.01, 1000)
            elif not np.isnan(phis[-1, j]):
                new_x = np.linspace(phis[1, j] - 0.01, phis[-1, j] + 0.01, 1000)

            ax2.grid()
            ax2.set_ylabel('Deviation from Linear (nm)')


            def e_poly_linear(x):
                b = e_poly(phis[0, j]) - phis[0, j] * (e_poly(phis[-1, j]) - e_poly(phis[0, j])) / (
                        phis[-1, j] - phis[0, j])
                return (e_poly(phis[-1, j]) - e_poly(phis[0, j])) / (phis[-1, j] - phis[0, j]) * x + b


            masked_reg = engine.eV_to_wave(e_poly(new_x) * u.eV)
            masked_lin = engine.eV_to_wave(e_poly_linear(new_x) * u.eV)
            deviation = masked_reg - masked_lin
            ax2.plot(new_x, deviation, color='k')
            for m, i in enumerate(phis[:, j]):
                ax2.plot(i, engine.eV_to_wave(e_poly(i) * u.eV) - engine.eV_to_wave(e_poly_linear(i) * u.eV), '.',
                         markersize=10, label=f'Order {spectro.orders[::-1][m]}')
            ax2.legend()

            ax2_2.grid()
            ax2_2.set_ylabel('R')
            ax2_2.set_xlabel(r'Phase $\times 2\pi$')
            s_eval = s_poly(e_poly(new_x))
            R = sig_to_R(s_eval, new_x)
            ax2_2.plot(new_x, R, color='k')
            for m, i in enumerate(phis[:, j]):
                ax2_2.plot(i, sig_to_R(sigs[m, j], i), '.', markersize=10, label=f'Order {spectro.orders[::-1][m]}')
            ax2.set_title(
                r'$E(\varphi)=$'f'{fit_e[2, j]:.2e}P_2+{fit_e[1, j]:.1e}P_1+{fit_e[0, j]:.2f}P_0\n'
                r'$\sigma(\varphi)=$'f'{fit_sig[2, j]:.2e}P_2+{fit_sig[1, j]:.2e}P_1+{fit_sig[0, j]:.2e}P_0'
            )

            ax1.set_xticks([])  # removes axis labels
            ax2.set_xticks([])
            res1.set_xticks([])

            plt.show()

    # plot all decent pixel models together:
    idxs = np.where(red_chi2 > 10)
    print(r'Number of pixels with reduced $\chi^2$ less than 10:', sim.npix - len(idxs[0]))

    gausses[:, [idxs]] = 0
    gausses[gausses < 0.1] = 0.1
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
    # plt.show()
