import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy
import scipy.interpolate as interp
from matplotlib import colors
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


def param_guess(phi_guess, energy_guess, sigma_guess, amp_guess, pixels, nord):
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
        params.add(f'P{j}_phi_m0', phi_guess[-1, j], min=-1, max=0)

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


def param_guess_1pix(phi_guess, energy_guess, sigma_guess, amp_guess):
    """
    :param phi_guess: n_ord guess values for phases
    :param energy_guess: n_ord guess values for energies of given phases
    :param sigma_guess: n_ord guess values for sigmas of given energies
    :param amp_guess: n_ord guess values for amps
    :return: an lmfit Parameter object
    """
    params = Parameters()

    # use Legendre polyfitting on the theoretical data to get guess coefs
    energy_coefs = np.polynomial.Legendre.fit(
        phi_guess,
        energy_guess,
        2,
        window=[phi_guess[0], phi_guess[-1]]
    ).coef

    sigma_coefs = np.polynomial.Legendre.fit(
        phi_guess,
        sigma_guess,
        2,
        window=[phi_guess[0], phi_guess[-1]]
    ).coef

    # add each item to params object:
    params.add(f'phi_m0', value=phi_guess[-1])

    for i in range(len(amp_guess)):
        params.add(f'O{i}_amp',
                   value=amp_guess[i]*(sigma_guess[i]*np.sqrt(2*np.pi)))

    for c in range(3):
        params.add(f'energy{c}', energy_coefs[c])
    for c in range(3):
        params.add(f'sigma{c}', sigma_coefs[c])

    return params


def fit_func(params, x_phases, y_counts, orders, pixels):
    """
    :param params: lmfit Parameter object
    :param x_phases: phases of the bins
    :param y_counts: counts of the bins
    :param orders: orders in spectrograph
    :param pixels: pixels indices in array
    :return: residuals between data and model
    """
    phi_m0, amp_coefs, e_coefs, sig_coefs = param_extract(params, len(pixels), len(orders))

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


def fit_func_1pix(params, x_phases, y_counts, orders, pix, plot=False):
    param_vals = list(params.valuesdict().values())  # turn dictionary of param values into list
    phi_m0 = param_vals[0]  # the m0 order center phase is the 1st param
    amps = np.array(param_vals[1:nord + 1])  # amplitudes are the next n_ord params
    s_0, s_1, s_2 = param_vals[nord + 1:nord + 4]  # energy poly coefs are the next 3 params
    q_0, q_1, q_2 = param_vals[nord + 4:]  # sigma poly coefs are the 3 last params

    # calculate the other order centers based on m0:
    m0 = orders[0]
    other_ord = orders[1:]
    const = s_0 - s_2 / 2 - (other_ord / m0) * (s_0 - s_2 / 2 + s_1 * phi_m0 + 3 / 2 * s_2 * phi_m0 ** 2)
    other_phis = (-s_1 - np.sqrt(s_1 ** 2 - 6 * s_2 * const)) / (3 * s_2)  # solving the quadratric euqation
    all_phis = np.append(phi_m0, other_phis)[::-1]  # put all the phis together

    # pass coef params to sigma poly and get sigmas at each phase center:
    sigs = np.polynomial.legendre.Legendre((q_0, q_1, q_2))(all_phis)

    gauss_i = gauss(x_phases, all_phis[:, None], sigs[:, None], amps[:, None])
    model = np.array([np.sum(gauss_i, axis=1)]).flatten()
    redchi = np.sum((y_counts-model)**2)/(len(model)-len(param_vals))

    if plot:  # WARNING, WILL PLOT HUNDREDS, MAYBE THOUSANDS IF FIT SUCKS/CANT CONVERGE
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
        quick_plot(res, [x_phases], [y_counts-model],
                   marker='.', color='purple', linestyle='None', labels=[None], ylabel='Residual', xlabel='Phase')
        res.set_xlim([-1,0])
        res.text(-0.3,10, f'Red. Chi^2={redchi:.1f}')
        plt.suptitle(f"Pixel {pix} Fitting Iteration")
        plt.tight_layout()
        plt.show()

    return (model - y_counts)/np.nan_to_num(np.sqrt(y_counts), nan=1)


def param_extract_1pix(params, n_ord):
    """
    :param params: Parameter object
    :param n_ord: number of orders
    :return: the extracted parameters
    """
    phi_m0 = params[f'phi_m0'].value
    energy_coefs = [params[f'energy{c}'].value for c in range(3)]
    sigma_coefs = [params[f'sigma{c}'].value for c in range(3)]
    amp_coefs = [params[f'O{i}_amp'].value for i in range(n_ord)]

    return phi_m0, energy_coefs, sigma_coefs, amp_coefs


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
    phi_m0 = np.array(param_vals[:npix])

    # amp coefs are the next 4*nord params:
    amp_coefs = np.array(param_vals[npix:npix+4*nord]).reshape([4, nord])

    # e coefs are the next 3*npix params:
    e_coefs = np.array(param_vals[npix+4*nord:npix+4*nord+3*npix]).reshape([3, npix])

    # sig coefs are the remaining 3*npix params:
    sig_coefs = np.array(param_vals[npix+4*nord+3*npix:]).reshape([3, npix])

    return phi_m0, amp_coefs, e_coefs, sig_coefs


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
    sigmas = sig_end-sig_start  # for all,flip order axis, ascending phase

    # ==================================================================================================================
    # MSF EXTRACTION STARTS
    # ==================================================================================================================
    # generating bins and building initial histogram for every pixel:
    n_bins = engine.n_bins(sparse_pixel, method="rice")
    bin_edges = np.linspace(args.bin_range[0], args.bin_range[1], n_bins + 1, endpoint=True)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    bin_counts = np.zeros([n_bins, sim.npix])
    for j in detector.pixel_indices:
        bin_counts[:, j], _ = np.histogram(photons_pixel[j], bins=bin_edges)

    # try manually fitting pixel 0:
    s0, s1, s2 = 0.9967, -2.7707, -4e-2
    q0, q1, q2 = 0.023, 3.3589e-3, 1.40176e-5
    phi_m0 = -0.274
    As = np.array([6.5, 15, 0.88, 67])

    m0 = 4
    other_ord = spectro.orders[1:]
    const = s0 - s2 / 2 - (other_ord / m0) * (s0 - s2 / 2 + s1 * phi_m0 + 3 / 2 * s2 * phi_m0 ** 2)
    other_phis = (-s1 - np.sqrt(s1 ** 2 - 6 * s2 * const)) / (3 * s2)  # solving the quadratric euqation

    ps = np.append(phi_m0, other_phis)[::-1]  # put all the phis together
    es = np.polynomial.Legendre((s0, s1, s2))(ps)
    ss = np.polynomial.Legendre((q0, q1, q2))(ps)
    gs = np.sum(gauss(bin_centers, ps[:, None], ss[:, None], As[:, None]), axis=1)
    plt.plot(bin_centers, bin_counts[:, 0], label='data')
    plt.plot(bin_centers, gs, label='model')
    plt.title('Manual Fitting of Pixel 0')
    plt.legend()
    plt.tight_layout()
    plt.show()

    params = []
    opt_params = []
    redchi = np.empty([sim.npix])
    # do the least squares fit:
    pixels = [0]
    for pix in pixels:
        plot = True if pix in [0,10,100,1000,2000] else False

        peaks, _ = scipy.signal.find_peaks(bin_counts[:, pix], distance=int(0.1 * n_bins))
        if len(peaks) >= nord:
            idx = np.argsort(bin_counts[peaks, pix])[:nord]
            peaks_nord = np.sort(peaks[idx])
        else:
            peaks_nord = [nearest_idx(bin_centers, sim_phase[i, pix]) for i in range(nord)]

        params.append(
            param_guess_1pix(bin_centers[peaks_nord], energy[:, pix], sigmas[:, pix], bin_counts[peaks_nord, pix]
                             )
        )

        opt_params.append(
            minimize(fit_func_1pix, params[0], args=(bin_centers, bin_counts[:, pix], spectro.orders, pix, plot)
                     , nan_policy='omit')
        )
        redchi[0] = opt_params[0].redchi

        #opt_params = minimize(fit_func, params, args=(bin_centers, bin_counts, spectro.orders, pixels))

    # extract the new parameters and evaluate at each order and pixel:
    phi_m0, (s_0, s_1, s_2), sig_coefs, amps = param_extract_1pix(opt_params[0].params, nord)

    # TODO USE BREAKPOINT BEFORE THIS calculate the other order centers based on m0 by solving the quadratic equation:
    m0 = spectro.orders[0]
    other_ord = spectro.orders[1:]
    const = s_0 - s_2 / 2 - (other_ord / m0) * (s_0 - s_2 / 2 + s_1 * phi_m0 + 3 / 2 * s_2 * phi_m0 ** 2)
    other_phis = (-s_1 - np.sqrt(s_1 ** 2 - 6 * s_2 * const)) / (3 * s_2)
    all_phis = np.append(phi_m0, other_phis)

    all_phis = all_phis[::-1]
    sig = np.polynomial.legendre.Legendre(sig_coefs)(all_phis)
    amps = np.array(amps)

    photon_bins = np.zeros([nord + 1, sim.npix])
    photon_bins[0, :] = -1
    # get the order bin edges:
    photon_bins[1:-1, pix] = gauss_intersect(all_phis, sig, amps)

    fig = plt.figure(1)
    ax = fig.add_axes((.1,.3,.8,.6))
    quick_plot(ax, [bin_centers], [bin_counts[:, pix]], labels=['Data'], first=True, color='k', xlabel='Phase',
               ylabel='Photon Count')
    ax.fill_between(bin_centers, 0, bin_counts[:, pix], color='k')
    x = [np.linspace(-1, 0, 1000) for i in range(nord)]
    quick_plot(ax, x, gauss(x, all_phis[:, None], sig[:, None], amps[:, None]).T,
               labels=[f'Order {i}' for i in spectro.orders[::-1]],
               title=f'Least-Squares Fit for Pixel {pix}', xlabel='Wavelength (nm)', ylabel='Photon Count')
    for b in photon_bins[:-1, pix]:
        ax.axvline(b, linestyle='--', color='black')
    ax.axvline(photon_bins[-1, pix], linestyle='--', color='black', label='Order Bin Edges')
    ax.set_xlim([-1,0])
    ax.set_xticklabels([])
    plt.legend()
    res = fig.add_axes((.1,.1,.8,.2))
    res.grid()
    quick_plot(res, [bin_centers],
               [bin_counts[:, pix]-np.sum(gauss(bin_centers, all_phis[:, None], sig[:, None], amps[:, None]).T, axis=0)],
               marker='.', color='purple', linestyle='None', labels=[None], ylabel='Residual', xlabel='Phase')
    res.set_xlim([-1,0])
    res.text(-0.3,10, f'Red. Chi^2={opt_params[0].redchi:.1f}')
    for b in photon_bins[:, pix]:
        res.axvline(b, linestyle='--', color='black')
    plt.tight_layout()
    plt.show()

    # plot the final fit for each pixel TODO NEEDS WORK, USE BREAKPOINT BEFORE THIS:
    if args.plotresults:
        for n, j in enumerate(pixels):
            fig, ax = plt.subplots(1, 1)
            quick_plot(ax, [bin_centers], [bin_counts[:, pix]], labels=['Data'], first=True, color='k', xlabel='Phase',
                       ylabel='Photon Count')
            ax.fill_between(bin_centers, 0, bin_counts[:, pix], color='k')
            x = [
                np.linspace(
                    engine.eV_to_wave(
                        E_of_phi(opt_params.params, args.bin_range[0], j)
                    ),
                    engine.eV_to_wave(
                        E_of_phi(opt_params.params, args.bin_range[-1], j)
                    ),
                    1000
                ) for i in range(nord)
            ]
            ax2 = ax.twin()
            quick_plot(ax2, x, engine.gauss(x,
                                            centers_nm[:, n][:, None],
                                            sigmas_nm[:, n][:, None],
                                            amp_counts[:, n][:, None]
                                            ).T,
                       labels=[f'Order {i}' for i in spectro.orders[::-1]], twin='red',
                       title=f'Least-Squares Fit for Pixel {j}', xlabel='Wavelength (nm)', ylabel='Photon Count')
            for b in photon_bins[:, j]:
                ax2.axvline(b, linestyle='--', color='black')
            plt.legend()
            plt.show()

    # initializing empty arrays for msf products:
    covariance = np.zeros([nord, nord, sim.npix])
    for n, j in enumerate(pixels):
        # get the covariance matrices:
        gauss_sum = np.sum(
            engine.gauss(
                wave,
                centers_nm[:, n][:, None],
                sigmas_nm[:, n][:, None],
                amp_counts[:, n][:, None]
            ),
            axis=0
        ) * (wave[1] - wave[0])

        gauss_int = [
            interp.InterpolatedUnivariateSpline(
                wave,
                engine.gauss(freq_domain, phase_centers[i, n], phase_sigmas[i, n], phase_counts[i, n]), k=1, ext=1
            ) for i in range(nord)
        ]

        covariance[:, :, j] = [[
            gauss_int[i].integral(
                photon_bins[k, j],
                photon_bins[k + 1, j]
            ) / gauss_sum[i] for k in range(nord)
        ] for i in range(nord)]

    logging.info(f"Finished fitting all pixels across all orders.")

    # assign bin edges, covariance matrices, bin centers, and simulation settings to MSF class and save:
    covariance = np.nan_to_num(covariance)
    msf = MKIDSpreadFunction(bin_edges=photon_bins, cov_matrix=covariance, waves=centers_nm, sim_settings=sim)
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
    spectra = get_spectrum(sim.type_spectra)
    wave = np.linspace(sim.minwave.to(u.nm).value - 100, sim.maxwave.to(u.nm).value + 100, 10000) * u.nm
    blazed_spectrum, _, _ = eng.blaze(wave, spectra)
    blazed_spectrum = np.nan_to_num(blazed_spectrum)
    pix_leftedge = spectro.pixel_wavelengths(edge='left').to(u.nm).value
    blaze_shape = [eng.lambda_to_pixel_space(wave, blazed_spectrum[i], pix_leftedge[i]) for i in range(nord)][::-1]
    blaze_shape = np.nan_to_num(blaze_shape)
    blaze_shape /= np.max(blaze_shape)  # normalize max to 1
    
    # plot unblazed, unshaped calibration spectrum (should be mostly flat line):
    fig2, ax2 = plt.subplots(1, 1)
    quick_plot(ax2, centers_nm, amp_counts / blaze_shape, labels=[f'O{i}' for i in spectro.orders[::-1]],
               first=True, title='Calibration Spectrum (Blaze Divided Out)', xlabel='Phase',
               ylabel='Photon Count', linestyle='None', marker='.')
    ax2.set_yscale('log')
    plt.show()
