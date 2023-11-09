import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy
import scipy.interpolate as interp
from matplotlib import colors
from matplotlib.colors import LogNorm

# import scipy.interpolate as interp
import astropy.units as u
import time
from datetime import datetime as dt
import argparse
import logging
import os
from lmfit import Parameters, minimize
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
        amp_coefs = np.polynomial.Legendre.fit(pixels, amp_guess[i], 3, window=[pixels[0], pixels[-1]]).coef
        for c in range(4):
            params.add(f'O{i}_amp{c}', amp_coefs[c])

    # use polyfit with the desired degrees on the theoretical data to get guess coefs
    for j in pixels:
        energy_coefs = np.polynomial.Polynomial.fit(
            phi_guess[:, j],
            energy_guess[:, j],
            2,
            window=[phi_guess[0, j], phi_guess[-1, j]]
        ).coef
        for c in range(3):
            params.add(f'P{j}_energy{c}', energy_coefs[c])

    for j in pixels:
        sigma_coefs = np.polynomial.Polynomial.fit(
            energy_guess[:, j],
            sigma_guess[:, j],
            2,
            window=[energy_guess[0, j], energy_guess[-1, j]]
        ).coef
        for c in range(3):
            params.add(f'P{j}_sigma{c}', sigma_coefs[c])

    return params


def param_guess_1pix(phi_guess, energy_guess, sigma_guess, amp_guess, degree=2):
    """
    :param phi_guess: n_ord guess values for phases
    :param energy_guess: n_ord guess values for energies of given phases
    :param sigma_guess: n_ord guess values for sigmas of given energies
    :param amp_guess: n_ord guess values for amps
    :param degree: degree to use for polynomial fitting, default 2nd order
    :return: an lmfit Parameter object with the populated parameters and guess values
    """
    params = Parameters()

    # use Legendre polyfitting on the theoretical/peak-finding data to get guess coefs
    energy_coefs = np.polynomial.Legendre.fit(
        x=phi_guess,
        y=energy_guess,
        deg=degree,
        window=[phi_guess[0], phi_guess[-1]]
    ).coef

    sigma_coefs = np.polynomial.Legendre.fit(
        x=phi_guess,
        y=sigma_guess,
        deg=degree,
        window=[phi_guess[0], phi_guess[-1]]
    ).coef

    # add phi_0s to params object:
    params.add(f'phi_0', value=phi_guess[-1])

    # add amplitudes to params object:
    for i in range(len(amp_guess)):
        params.add(f'O{i}_amp',
                   value=amp_guess[i]*(sigma_guess[i]*np.sqrt(2*np.pi)), min=0)

    # add polynomial coefficients to params object:
    for c in range(degree):
        params.add(f'energy{c}', energy_coefs[c])
    # force x^2 coefficients to not be 0, that is causing nan in quadratic equation later,
    # TODO, figure out how to force parameter to be anything /but/ 0 (so it can be negative as well):
    params.add(f'energy{degree}', energy_coefs[degree], min=1e-5)
    for c in range(degree):
        params.add(f'sigma{c}', sigma_coefs[c])
    params.add(f'sigma{degree}', sigma_coefs[degree], min=0)

    return params


def fit_func_all(params, x_phases, y_counts, orders, pixels):
    """
    :param params: lmfit Parameter object
    :param x_phases: phases of the bins
    :param y_counts: counts of the bins
    :param orders: orders in spectrograph
    :param pixels: pixels indices in array
    :return: residuals between data and model
    """
    phi_0, amp_coefs, e_coefs, sig_coefs = param_extract(params, len(pixels), len(orders))

    # calculate the other order centers based on the m0:
    phis = phi_from_m0(orders[0], orders[1:], (e_coefs[0], e_coefs[1], e_coefs[2]))

    q0, q1, q2 = sig_coefs[0], sig_coefs[1], sig_coefs[2]
    sigs = [np.polynomial.legendre.Legendre((q0[j], q1[j], q2[j]))(phis[:, j]) for j in pixels].T

    a0, a1, a2, a3 = amp_coefs[0], amp_coefs[1], amp_coefs[2], amp_coefs[3]
    amps = [np.polynomial.legendre.Legendre((a0[i], a1[i], a2[i], a3[i]))(pixels) for i in range(len(orders))]

    model = np.empty([len(x_phases), len(pixels)])
    gauss_i = [gauss(x_phases, phis[:, j, None], sigs[:, j, None], amps[:, j, None]) for j in pixels]
    model = np.array([np.sum(gauss, axis=1)])

    return model.flatten() - y_counts.flatten()


def fit_func_1pix(params, x_phases, y_counts, orders, pix, degree=2, plot=False):
    # turn dictionary of param values into list:
    param_vals = list(params.valuesdict().values())

    # by how I ordered it, the first param is always the phi_0 value
    phi_0 = param_vals[0]

    # the next n_ord params are the amplitudes:
    amps = np.array(param_vals[1:nord + 1])

    # the next poly_degree+1 params are the energy coefficients:
    s0, s1, s2 = param_vals[nord + 1:nord + 2 + degree]

    # the last poly_degree+1 params are the sigma coefficients:
    q0, q1, q2 = param_vals[nord + 2 + degree:]

    # calculate the other phi_m based on phi_0:
    # derivation is in Overleaf doc, specific to Legendre polys
    m0 = orders[0]
    other_orders = orders[1:][::-1]  # flipping so that orders are in ascending *lambda* (like the rest)
    const = s0 - s2 / 2 - (other_orders / m0) * (s0 - s2 / 2 + s1 * phi_0 + 3 / 2 * s2 * phi_0 ** 2)
    phis = np.append((-s1 - np.sqrt(s1 ** 2 - 6 * s2 * const)) / (3 * s2), phi_0)  # solving the quadratric equation

    # pass coef parameters to sigma poly and get sigmas at each phase center:
    sigs = np.polynomial.legendre.Legendre((q0, q1, q2))(phis)

    # put the phi, sigma, and amps together for Gaussian model:
    gauss_i = gauss(x_phases, phis[:, None], sigs[:, None], amps[:, None])
    model = np.array([np.sum(gauss_i, axis=1)]).flatten()
    reduced_chi2 = np.sum((y_counts - model) ** 2) / (len(model) - len(param_vals))

    # WARNING, WILL PLOT HUNDREDS IF FIT SUCKS/CAN'T CONVERGE
    if plot:
        fig = plt.figure(1)
        ax = fig.add_axes((.1,.3,.8,.6))
        ax.plot(x_phases, y_counts, label='Data')
        ax.plot(x_phases, model, label='Model')
        ax.legend()
        ax.set_ylabel('Count')
        ax.set_xlim([-1,0])
        ax.set_xticklabels([])
        ax.legend()
        res = fig.add_axes((.1,.1,.8,.2))
        res.grid()
        res.plot(x_phases, model-y_counts, '.', color='purple')
        res.set_ylabel('Residual')
        res.set_xlabel('Phase')
        res.set_xlim([-1,0])
        res.text(-0.3, 10, f'Red. Chi^2={reduced_chi2:.1f}')
        plt.suptitle(f"Pixel {pix} Fitting Iteration")
        plt.tight_layout()
        plt.show()

    # use np.divide so divide by 0 is prevented:
    weighted_residual = np.divide(model-y_counts, np.sqrt(y_counts), out=np.zeros_like(y_counts), where=np.sqrt(y_counts) != 0)
    return weighted_residual


def param_extract_1pix(params, n_ord, degree=2):
    """
    :param params: Parameter object
    :param n_ord: number of orders
    :param degree: polynomial degree used
    :return: the extracted parameters
    """
    phi_0 = params[f'phi_0'].value
    energy_coefs = [params[f'energy{c}'].value for c in range(degree + 1)]
    sigma_coefs = [params[f'sigma{c}'].value for c in range(degree + 1)]
    amps = [params[f'O{i}_amp'].value for i in range(n_ord)]

    return phi_0, energy_coefs, sigma_coefs, amps


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
    amp_coefs = np.array(param_vals[npix:npix+4*nord]).reshape([4, nord])

    # e coefs are the next 3*npix params:
    e_coefs = np.array(param_vals[npix+4*nord:npix+4*nord+3*npix]).reshape([3, npix])

    # sig coefs are the remaining 3*npix params:
    sig_coefs = np.array(param_vals[npix+4*nord+3*npix:]).reshape([3, npix])

    return phi_0, amp_coefs, e_coefs, sig_coefs


def gauss(x, mu, sig, A):
    """
    :param x: wavelength
    :param mu: mean
    :param sig: sigma
    :param A: amplitude
    :return: value of the Gaussian
    """
    return (A/(sig*np.sqrt(2*np.pi)) * np.exp(- (x - mu) ** 2. / (2. * sig ** 2.))).T


def _quad_formula(a, b, c):
    """
    :return: positive quadratic formula result for ax^2 + bx + c = 0
    """
    return ((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)), ((-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))


def gauss_intersect(mu, sig, A):
    """
    :param mu: means of Gaussians in order
    :param sig: sigmas of Gaussians in order
    :param A: amplitudes of Gaussians in order
    :return: analytic calculation of the intersection point between 1D Gaussian functions
    """
    n = len(mu)
    if n != len(sig) or n != len(A):
        raise ValueError("mu, sig, and A must all be the same size.")
    a = [1 / (2*sig[i+1] ** 2) - 1 / (2*sig[i] ** 2) for i in range(n - 1)]
    b = [mu[i] / (sig[i] ** 2) - mu[i+1] / (sig[i+1] ** 2) for i in range(n - 1)]
    c = [(mu[i+1]**2 / (2*sig[i+1] ** 2)) - (mu[i]**2 / (2*sig[i] ** 2)) + np.log((A[i]*sig[i+1])/(A[i + 1]*sig[i])) for i in range(n - 1)]
    solp, soln = _quad_formula(np.array(a), np.array(b), np.array(c))
    for i in range(n-1):
        if mu[i] < solp[i] < mu[i+1]:
            return solp
        else:
            return soln


def nearest_idx(array, value):
    return (np.abs(array - value)).argmin()


if __name__ == '__main__':
    tic = time.time()  # recording start time for script

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
    os.makedirs(f'{args.output_dir}/logging', exist_ok=True)

    # ==================================================================================================================
    # START LOGGING TO FILE
    # ==================================================================================================================
    now = dt.now()
    logging.basicConfig(
        filename=f'{args.output_dir}/logging/msf_{now.strftime("%Y%m%d_%H%M%S")}.log',
        format='%(levelname)s:%(message)s',
        level=logging.INFO
    )
    logging.info("The process of recovering the MKID Spread Function (MSF) is recorded."
                 f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    # ==================================================================================================================
    # OPEN PHOTON TABLE AND PULL NECESSARY DATA
    # ==================================================================================================================
    pt = Photontable(args.caltable)
    sim = pt.query_header('sim_settings')
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
    if not os.path.exists(sim.R0s_file):
        IOError('File does not exist, check path and file name.')
    else:
        R0s = np.loadtxt(sim.R0s_file, delimiter=',')
        logging.info(f'\nThe individual R0s were imported from {sim.R0s_file}.')

    # creating the spectrometer objects:
    detector = MKIDDetector(sim.npix, sim.pixelsize, sim.designR0, sim.l0, R0s, None, resid_map)
    grating = GratingSetup(sim.alpha, sim.delta, sim.groove_length)
    spectro = SpectrographSetup(sim.m0, sim.m_max, sim.l0, sim.pixels_per_res_elem, sim.focallength, grating, detector)
    eng = engine.Engine(spectro)

    # shortening some long variable names:
    nord = spectro.nord
    pixels = detector.pixel_indices

    # converting and calculating the simulation phases, energies, sigmas ("true"):
    sim_phase = np.nan_to_num(wave_to_phase(spectro.pixel_wavelengths().to(u.nm)[::-1], minwave=sim.minwave, maxwave=sim.maxwave))
    energy = np.nan_to_num(engine.wave_to_eV(spectro.pixel_wavelengths().to(u.nm)[::-1]).value)
    sig_start = wave_to_phase(spectro.pixel_wavelengths().to(u.nm)[::-1]-(detector.mkid_resolution_width(
        spectro.pixel_wavelengths().to(u.nm)[::-1], pixels) / (2 * np.log(2)))/2,
                              minwave=sim.minwave, maxwave=sim.maxwave)
    sig_end = wave_to_phase(spectro.pixel_wavelengths().to(u.nm)[::-1]+(detector.mkid_resolution_width(
        spectro.pixel_wavelengths().to(u.nm)[::-1], pixels) / (2 * np.log(2)))/2,
                            minwave=sim.minwave, maxwave=sim.maxwave)
    sigmas = sig_end-sig_start  # for all, flip order axis to be in ascending phase/lambda

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

    # IGNORE BELOW manual fitting pixel 0:
    # s0, s1, s2 = 0.9967, -2.7707, -4e-2
    # q0, q1, q2 = 0.023, 3.3589e-3, 1.40176e-5
    # phi_0 = -0.274
    # As = np.array([6.5, 15, 0.88, 67])

    # m0 = 4
    # other_ord = spectro.orders[1:]
    # const = s0 - s2 / 2 - (other_ord / m0) * (s0 - s2 / 2 + s1 * phi_0 + 3 / 2 * s2 * phi_0 ** 2)
    # other_phis = (-s1 - np.sqrt(s1 ** 2 - 6 * s2 * const)) / (3 * s2)  # solving the quadratric euqation

    # ps = np.append(phi_0, other_phis)[::-1]  # put all the phis together
    # es = np.polynomial.Legendre((s0, s1, s2))(ps)
    # sigs = np.polynomial.Legendre((q0, q1, q2))(ps)
    # gs = np.sum(gauss(bin_centers, ps[:, None], sigs[:, None], As[:, None]), axis=1)
    # plt.plot(bin_centers, bin_counts[:, 0], label='data')
    # plt.plot(bin_centers, gs, label='model')
    # plt.title('Manual Fitting of Pixel 0')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # IGNORE ABOVE

    # create lists/arrays to place loop values, param objects can only go in lists:
    used_pix = []
    params = []
    opt_params = []
    reduced_chi2 = np.empty([sim.npix])

    # separate index for list indexing because param objects can only be in lists:
    n = 0

    # do the lmfit:
    for pix in pixels:
        # use find peaks to get phi/amplitude guesses IF there are at least n_ord peaks,
        # use default simulation values if there are less than n_ord peaks:
        peaks, _ = scipy.signal.find_peaks(bin_counts[:, pix], distance=int(0.1 * n_bins))
        if len(peaks) >= nord:
            idx = np.argsort(bin_counts[peaks, pix])[:nord]
            peaks_nord = np.sort(peaks[idx])
        else:
            peaks_nord = [nearest_idx(bin_centers, sim_phase[i, pix]) for i in range(nord)]

        # obtain Parameter object populated with guesses:
        params.append(
            param_guess_1pix(
                phi_guess=bin_centers[peaks_nord],
                energy_guess=energy[:, pix],
                sigma_guess=sigmas[:, pix],
                amp_guess=bin_counts[peaks_nord, pix]
            )
        )

        # attempt to optimize fit, pass if fails:
        try:
            opt_params.append(
                minimize(
                    fit_func_1pix,
                    params[n],
                    args=(bin_centers, bin_counts[:, pix], spectro.orders, pix)
                )
            )
            reduced_chi2[pix] = opt_params[n].redchi

            # record which pixels were fit "successfully":
            used_pix.append(pix)

            # increase index for lists
            n += 1
        except ValueError as v:
            if pix == pixels[-1]:
                # prints why the fit failed
                print(v)
            pass

    # create empty arrays to hold phi, sigma, and amp values for gaussian model:
    phis = np.empty([nord, sim.npix])
    sigs = np.empty([nord, sim.npix])
    amps = np.empty([nord, sim.npix])
    gausses = np.empty([n_bins, sim.npix])

    # for use in recovery of other phis:
    m0 = spectro.orders[0]
    other_orders = spectro.orders[1:][::-1]

    # create empty array for order bin edges:
    order_edges = np.zeros([nord + 1, sim.npix])
    order_edges[0, :] = -1

    for n, j in enumerate(used_pix):
        # extract the successfully fit parameters:
        phis[-1, j], (s0, s1, s2), (q0, q1, q2), amps[:, j] = param_extract_1pix(opt_params[n].params, nord)

        # evaluate the parameters to get phi, sigma, and amp for Gaussian model:
        const = s0 - s2 / 2 - (other_orders / m0) * (s0 - s2 / 2 + s1 * phis[-1, j] + 3 / 2 * s2 * phis[-1, j] ** 2)
        phis[:-1, j] = (-s1 - np.sqrt(s1 ** 2 - 6 * s2 * const)) / (3 * s2)
        sigs[:, j] = np.polynomial.legendre.Legendre((q0, q1, q2))(phis[:, j])

        # get the order bin edges:
        order_edges[1:-1, j] = gauss_intersect(phis[:, j], sigs[:, j], amps[:, j])

        # put together the Gaussian model and store to array:
        gausses[:, j] = np.sum(gauss(bin_centers, phis[:, j, None], sigs[:, j, None], amps[:, j, None]), axis=1)

        # plot the individual pixels:
        fig = plt.figure(1)

        # plotting the model and data on top of each other:
        ax = fig.add_axes((.1,.3,.8,.6))
        ax.plot(bin_centers, bin_counts[:, j], 'k', label='Data')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Photon Count')
        ax.fill_between(bin_centers, 0, bin_counts[:, j], color='k')
        x = [np.linspace(-1, 0, 1000) for i in range(nord)]
        quick_plot(ax, x, gauss(x, phis[:, j, None], sigs[:, j, None], amps[:, j, None]).T,
                   labels=[f'Order {i}' for i in spectro.orders[::-1]],
                   title=f'Least-Squares Fit for Pixel {j}', xlabel='Wavelength (nm)', ylabel='Photon Count')
        for b in order_edges[:-1, j]:
            ax.axvline(b, linestyle='--', color='black')
        ax.axvline(order_edges[-1, j], linestyle='--', color='black', label='Order Edges')
        ax.set_xlim([-1,0])
        ax.set_xticklabels([])
        ax.legend()

        # plotting the residual between model and data below:
        res = fig.add_axes((.1,.1,.8,.2))
        res.grid()
        quick_plot(res, [bin_centers], [bin_counts[:, j]-gausses[:, j]],
                   marker='.', color='purple', linestyle='None', labels=[None], ylabel='Residual', xlabel='Phase')
        res.set_xlim([-1,0])
        res.text(-0.3,10, f'Red. Chi^2={opt_params[n].redchi:.1f}')
        for b in order_edges[:, j]:
            res.axvline(b, linestyle='--', color='black')
        plt.tight_layout()
        plt.show()

    # plot all pixel models together:
    plt.imshow(gausses, extent=[1, sim.npix, bin_centers[0], bin_centers[-1]], aspect='auto')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Photon Count')
    plt.title("Fit MSF Model")
    plt.xlabel("Pixel Index")
    plt.ylabel(r"Phase ($\times \pi /2$)")
    plt.tight_layout()
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
