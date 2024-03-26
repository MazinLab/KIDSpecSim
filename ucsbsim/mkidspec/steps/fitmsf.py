import numpy as np
from numpy.polynomial.legendre import Legendre
import matplotlib.pyplot as plt
import scipy.signal as sig
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
from ucsbsim.mkidspec.spectra import get_spectrum, apply_bandpass, clip_spectrum, FineGrid
import ucsbsim.mkidspec.engine as engine
from ucsbsim.mkidspec.msf import MKIDSpreadFunction
from ucsbsim.mkidspec.spectrograph import GratingSetup, SpectrographSetup
from ucsbsim.mkidspec.detector import MKIDDetector, wave_to_phase
from ucsbsim.mkidspec.plotting import quick_plot
import ucsbsim.mkidspec.utils.general as gen

"""
Development of the MKID Spread Function (MSF) from a calibration (flat-field or known blackbody) spectrum.
The steps are:
-Load the calibration photon table.
-Fit model function to each pixel.
-Use Gaussian function intersections to obtain virtual pixel bin edges for each order.
-Calculate fractional bleed between orders and converts them into an n_ord x n_ord "covariance" matrix for each
 pixel. This matrix how much of each order's flux was potentially grouped into another order.
 This will later be used to determine the error on the follow-on extracted spectra.
-Saves newly obtained bin edges and covariance matrices to files.
"""

'''
# TODO a new parameter initial guess generator that doesnt use assumptions about which orders are missing
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

    peaks_idx, _ = sig.find_peaks(y_counts, distance=int(0.1 * n_bins), height=snr ** 2)
    sort_idx = np.argsort(y_counts[peaks_idx])[::-1][:nord]  # get rid of smallest peaks, if more than nord
    peaks_sort_idx = np.sort(peaks_idx[sort_idx])  # sorting the indices by phase
    peaks_phi = y_counts[peaks_sort_idx]  # the phases of the retrieved peaks

    if len(peaks_sort_idx) == nord:
        peaks_nord = peaks_sort_idx  # indices of final peaks
        missing_ord = None
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
                missing_ord = np.range(0, n_missing)
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
    if missing_ord is None:  # no orders were identified as missing
        for i in range(len(amp_guess)):
            parameters.add(
                name=f'O{i}_amp',
                value=amp_guess[i],
                min=amp_guess[i] * (1 - reg_per),
                max=amp_guess[i] * (1 + reg_per)
            )
    else:
        for i in range(len(amp_guess)):
            if i in missing_ord:
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
        missing_ord=None,
        percent=0.3,
        degree=2,
        e_domain=[-1, 0]
):
    """
    :param phi_guess: n_ord guess values for phases
    :param e_guess: n_ord guess values for energies of given phases
    :param s_guess: n_ord guess values for sigmas of given energies
    :param amp_guess: n_ord guess values for fit_amps
    :param missing_ord: as list, the orders that are probably missing/0
    :param percent: the percentage to add to the amp threshold
    :param degree: degree to use for polynomial fitting, default 2nd order
    :param e_domain: domain for the legendre polynomial
    :return: an lmfit Parameter object with the populated parameters and guess values
    """
    parameters = Parameters()

    # use Legendre polyfitting on the theoretical/peak-finding data to get guess coefs
    e_coefs = Legendre.fit(x=phi_guess, y=e_guess / e_guess[-1], deg=degree, domain=e_domain).coef
    s_coefs = Legendre.fit(x=e_guess / e_guess[-1], y=s_guess, deg=degree, domain=[e_guess[0] / e_guess[-1], 1]).coef

    # add the sigma coefs to params object:
    parameters.add(name=f's0', value=s_coefs[0], min=1e-5)
    parameters.add(name=f's1', value=s_coefs[1], max=0)
    parameters.add(name=f's2', value=s_coefs[2], min=-1e-2, max=1e-2)

    # add phi_0s to params object:
    parameters.add(name=f'phi_0', value=phi_guess[-1], min=phi_guess[-1] - 0.1, max=phi_guess[-1] + 0.1)

    # add energy coefs to params object:
    parameters.add(name=f'e1', value=e_coefs[1], max=0.01)
    parameters.add(name=f'e2', value=e_coefs[2])#, min=-1e-2, max=1e-2)

    # add amplitudes to params object:
    if missing_ord is None:  # no orders were identified as missing
        for i in range(len(amp_guess)):
            if amp_guess[i] == 0:
                parameters.add(
                    name=f'O{i}_amp',
                    value=amp_guess[i],
                    min=0
                )
            else:
                parameters.add(
                    name=f'O{i}_amp',
                    value=amp_guess[i],
                    min=amp_guess[i] * (1 - percent),
                    max=amp_guess[i] * (1 + percent)
                )
    else:  # at least one order was flagged as missing
        for i in range(len(amp_guess)):
            if i in missing_ord:
                parameters.add(name=f'O{i}_amp', value=0, vary=False)
            else:
                if amp_guess[i] == 0:
                    parameters.add(
                        name=f'O{i}_amp',
                        value=amp_guess[i],
                        min=0
                    )
                else:
                    parameters.add(
                        name=f'O{i}_amp',
                        value=amp_guess[i],
                        min=amp_guess[i] * (1 - percent),
                        max=amp_guess[i] * (1 + percent)
                    )

    return parameters


def fit_func(
        params: Parameters,
        x_phases,
        y_counts=None,
        orders=None,
        pix: int = None,
        legendre_e=None,
        legendre_s=None,
        degree: int = 2,
        plot: bool = False,
        to_sum: bool = False
):
    """
    :param params: Parameter object containing all params with initial guesses
    :param x_phases: the grid of phases to be sampled along
    :param y_counts: the histogram data details photon count in each bin, must be same shape as x_phases
    :param orders: a list of the orders incident on the detector, in ascending order
    :param pix: the index for the pixel being fit for, only needed for plotting
    :param legendre_e: the Legendre poly object with the proper phase domain
    :param legendre_s: the Legendre poly object with the proper energy domain
    :param degree: number of polynomial degrees to use for energy and sigma polys
    :param bool plot: whether to show the plot of the fit
    :param bool to_sum: whether to sum the fitting function or leave the individual Gaussians
    :return: residuals, fitting function separated, or fitting function summed
    """

    # turn dictionary of param values into separate params:
    s0, s1, s2, phi_0, e1, e2, *amps = tuple(params.valuesdict().values())

    # obtain the 0th order coef
    e0 = e0_from_params(e1, e2, phi_0)

    # pass coef parameters to polys:
    setattr(legendre_e, 'coef', (e0, e1, e2))
    setattr(legendre_s, 'coef', (s0, s1, s2))

    try:
        # calculate the other phi_m based on phi_0:
        phis = phis_from_grating_eq(orders, phi_0, leg=legendre_e, coefs=[e0, e1, e2])

        # get sigmas at each phase center:
        sigs = legendre_s(legendre_e(phis))

        # put the phi, sigma, and fit_amps together for Gaussian model:
        gauss_i = np.nan_to_num(gen.gauss(x_phases, phis[:, None], sigs[:, None], np.array(amps)[:, None]))
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


def extract_params(params, nord, degree=2):
    """
    :param params: Parameter object
    :param nord: number of orders
    :param degree: the polynomial degree
    :return: the extracted parameters
    """
    phi_0 = params[f'phi_0'].value
    e_coefs = np.array([params[f'e{c}'].value for c in range(1, degree + 1)])
    s_coefs = np.array([params[f's{c}'].value for c in range(degree + 1)])
    amps = np.array([params[f'O{i}_amp'].value for i in range(nord)])

    return phi_0, e_coefs, s_coefs, amps


def e0_from_params(e1, e2, phi_0):
    """
    :param e1: the 1st order coef
    :param e2: the 2nd order coef
    :param phi_0: the initial order phase center
    :return: the Legendre poly solved for the 0th order coef given dimensionless energy
    """
    return 1 - e2 * (1 / 2 * (3 * ((phi_0 + 0.5) * 2) ** 2 - 1)) - e1 * 2 * (phi_0 + 0.5)


def phis_from_grating_eq(orders, phi_0, leg, coefs=None):
    """
    :param orders: orders of the spectrograph in ascending order
    :param phi_0: the phase center of the initial order
    :param leg: the energy legendre poly object
    :param coefs: the coefs of the energy poly in ascending order
    :return: the phase centers of the other orders constrained by the grating equation
    """
    grating_eq = orders[1:][::-1] / orders[0] * leg(phi_0)

    phis = []
    for i in grating_eq:
        roots = np.roots([6 * coefs[2], 6 * coefs[2] + 2 * coefs[1], coefs[2] + coefs[0] + coefs[1] - i])

        if len(roots) == 1:  # when there is only 1 root, use it
            phis.append(roots[0])
        elif ~np.isfinite(roots).all():  # if both roots are invalid, raise error
            raise ValueError("All calculated roots are not valid, check equation.")
        else:  # if there are 2 valid roots, use the one in the proper range
            phis.append(roots[(-1.5 < roots) & (roots < phi_0)][0])

    return np.append(np.array(phis), phi_0)


def cov_from_params(params, model, nord, order_edges, valid_idx, x_phases, **fit_kwargs):
    """
    :param params: Parameter object
    :param model: the model function
    :param nord: the number of orders
    :param order_edges: the indices of the virtual pixel edges
    :param valid_idx: the valid, non-zero order indices
    :param x_phases: phase grid to be used
    :param fit_kwargs: kwargs to pass to fit_func
    :return: the covariance between orders as nord x nord array, diagonal should be near 1
    """

    # v giving order, > receiving order [g_idx, r_idx, pixel]
    #      9   8   7   6   5
    # 9 [  1   #   #   #   #  ]  < how much of order 9 is in every other other as a fraction of order 9
    # 8 [  #   1   #   #   #  ]
    # 7 [  #   #   1   #   #  ]
    # 6 [  #   #   #   1   #  ]
    # 5 [  #   #   #   #   1  ]
    #      ^ how much of other orders is in order 9, every column must be a fraction of every orders

    model = interp.InterpolatedUnivariateSpline(x=x_phases, y=model, k=1, ext=1)
    model_sum = np.array([model.integral(order_edges[i], order_edges[i + 1]) for i in range(len(order_edges) - 1)])
    cov = np.zeros([nord, nord])
    for k in valid_idx:
        suppress_params = copy.deepcopy(params)
        suppress_params.add(f'O{k}_amp', value=0)
        suppress_model = fit_func(suppress_params, x_phases, **fit_kwargs)
        suppress_model = interp.InterpolatedUnivariateSpline(x=x_phases, y=suppress_model, k=1, ext=1)
        suppress_model_sum = np.array([
            suppress_model.integral(order_edges[i], order_edges[i + 1]) for i in range(len(order_edges) - 1)])
        cov[valid_idx, k] = np.nan_to_num((model_sum - suppress_model_sum) / model_sum)
        if np.array_equal(cov[valid_idx, k], np.zeros([nord])):
            cov[k, k] = 1
    cov[cov < 0] = 0
    return cov


def fitmsf(
        msf_table,
        sim,
        outdir: str,
        resid_map=np.arange(2048, dtype=int) * 10 + 100,
        bin_range: tuple = (-1, 0),
        bins=50,  # hardcode since 1 peak ~ 0.1 phase, 10 points across peak
        missing_order_pix=[[(0, 346), [2]], [(0, 1300), [0, 2]], [(346, 1300), [0]], [(1300, 2047), [1]], [(1300, 2047), [1, 3]]],
        # each item is (pix start, pix end), [list of missing orders]
        snr=3,
        plot: bool = False
):
    photons_pixel = engine.sorted_table(table=msf_table, resid_map=resid_map)

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
    eng = engine.Engine(spectrograph=spectro)

    # shortening some longer variable names:
    nord = spectro.nord
    pixels = detector.pixel_indices
    pix_waves = spectro.pixel_wavelengths().to(u.nm)[::-1]  # flip order axis to be in ascending phase/lambda:

    # converting and calculating the simulation phases, energies, and sigmas
    sim_phase = np.nan_to_num(wave_to_phase(pix_waves, minwave=sim.minwave, maxwave=sim.maxwave))

    energy = np.nan_to_num(engine.wave_to_eV(pix_waves).value)
    sig_start = wave_to_phase(pix_waves - (detector.mkid_resolution_width(pix_waves, pixels) / (2 * np.log(2))) / 2,
                              minwave=sim.minwave, maxwave=sim.maxwave)
    sig_end = wave_to_phase(pix_waves + (detector.mkid_resolution_width(pix_waves, pixels) / (2 * np.log(2))) / 2,
                            minwave=sim.minwave, maxwave=sim.maxwave)
    sigmas = sig_end - sig_start  # approximate sigmas given starting point to end point conversion

    # generating # of bins and building initial histogram for every pixel:
    num_pixel = [len(photons_pixel[j]) for j in detector.pixel_indices]  # number of photons in all pixels
    sparse_pixel = int(np.min(num_pixel))  # number of photons in sparsest pixel
    #n_bins = gen.n_bins(n_data=sparse_pixel, method="rice")  # bins the same for all, based on pixel with fewest photons

    bin_edges = np.linspace(bin_range[0], bin_range[1], bins + 1, endpoint=True)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    bin_counts = np.zeros([bins, sim.npix])
    for j in pixels:
        bin_counts[:, j], _ = np.histogram(photons_pixel[j], bins=bin_edges)

    # create lists/arrays to place loop values:
    red_chi2 = np.full(sim.npix, fill_value=1e6)

    # create empty arrays to hold phi values for gaussian model:
    all_fit_phi = np.empty([nord, sim.npix])
    all_fit_sig = np.empty([nord, sim.npix])
    fine_phase_grid = np.linspace(bin_range[0], bin_range[1], 1000)
    gausses = np.zeros([1000, sim.npix])  # the entire n_ord Gaussian model summed
    gausses_i = np.zeros([1000, nord, sim.npix])  # model separated by orders

    # create empty arrays for the order-bleeding covariance and errors:
    covariance = np.zeros([nord, nord, sim.npix])
    p_err = np.zeros([nord, sim.npix])
    m_err = np.zeros([nord, sim.npix])

    # create empty array for order bin edges and virtual pixel counts:
    order_edges = np.zeros([nord + 1, sim.npix])
    order_edges[0, :] = -1.5
    ord_counts = np.zeros([nord, sim.npix])
    inval_idx = []
    full_pix = []
    opt_param_all = []

    redchi_val = 30  # TODO remove after debug
    xtol = 1e-4
    #pixels = [1499]

    leg_e = Legendre(coef=(0, 0, 0), domain=np.array(bin_range))

    # do the non-linear least squares fit:
    for p in pixels:
        #print(p)
        leg_s = Legendre(coef=(0, 0, 0), domain=[energy[0, p] / energy[-1, p], 1])

        peaks, _ = sig.find_peaks(x=bin_counts[:, p], distance=int(0.095 * bins), height=snr**2)
        missing_ord = None

        # use find peaks to get phi/amplitude guesses:
        # TODO make into method by ruling out different cases based on peaks, use avg distance from 0 + avg peak sep
        if len(peaks) >= nord:  # all peaks are present
            idx = np.argsort(bin_counts[peaks, p])[::-1][:nord]
            peaks_nord = np.sort(peaks[idx])
        else:
            idx = np.argsort(bin_counts[peaks, p])[::-1]
            peaks_sort = np.sort(peaks[idx])
            for item in missing_order_pix:
                if item[0][0] <= p <= item[0][1] and len(peaks_sort) == (nord - len(item[1])):
                    peaks_nord = peaks_sort
                    for o in item[1]:
                        if o == 0:
                            idx = int(peaks_sort[0] - (peaks_sort[1] - peaks_sort[0]) / 2)
                            if idx >= 0:
                                peaks_nord = np.insert(peaks_nord, o, idx)
                            else:
                                peaks_nord = np.insert(peaks_nord, o, 0)
                        elif o == nord-1:
                            idx = int(peaks_sort[-1] + (peaks_sort[-1] - peaks_sort[-2]) / 2)
                            if idx >= bins-1:
                                peaks_nord = np.insert(peaks_nord, o, bins-1)
                            else:
                                peaks_nord = np.insert(peaks_nord, o, idx)
                        else:
                            peaks_nord = np.insert(peaks_nord, o, int((peaks_sort[o-1] + peaks_sort[o])/2))
                    missing_ord = item[1]
                    # if p <= 346 and len(peaks_sort) == 3:  # 1 order missing
                    #     peaks_nord = np.array([
                    #         peaks_sort[0],
                    #         peaks_sort[1],
                    #         int((peaks_sort[1] + peaks_sort[2]) / 2),
                    #         peaks_sort[2]]
                    #     )
                    #     missing_ord = [2]
                    # elif p <= 1300 and len(peaks_sort) == 2:  # 2 orders missing
                    #     peaks_nord = np.array([
                    #         int(peaks_sort[0] - (peaks_sort[1] - peaks_sort[0]) / 2),
                    #         peaks_sort[0],
                    #         int((peaks_sort[0] + peaks_sort[1]) / 2),
                    #         peaks_sort[1]]
                    #     )
                    #     missing_ord = [0, 2]
                    # elif 346 < p <= 1300 and len(peaks_sort) == 3:  # 1 order missing
                    #     peaks_nord = np.array([
                    #         int(peaks_sort[0] - (peaks_sort[1] - peaks_sort[0]) / 2),
                    #         peaks_sort[0],
                    #         peaks_sort[1],
                    #         peaks_sort[2]]
                    #     )
                    #     missing_ord = [0]
                    # elif 1680 < p < 1800 and len(peaks_sort) == 3:
                    #     # weird zone with flip flop between order 4 and 5
                    #     peaks, _ = sig.find_peaks(bin_counts[:, p], distance=int(0.25 * n_bins), height=snr**2)
                    #     idx = np.argsort(bin_counts[peaks, p])[::-1]
                    #     peaks_sort = np.sort(peaks[idx])
                    #     peaks_nord = np.array([
                    #         peaks_sort[0],
                    #         int((peaks_sort[0] + peaks_sort[1]) / 2),
                    #         peaks_sort[1],
                    #         int(peaks_sort[1] + (peaks_sort[1] - peaks_sort[0]) / 2)]
                    #     )
                    #     missing_ord = [1, 3]
                    # elif p > 1300 and len(peaks_sort) == 3:  # 1 order missing
                    #     peaks_nord = np.array([
                    #         peaks_sort[0],
                    #         int((peaks_sort[0] + peaks_sort[1]) / 2),
                    #         peaks_sort[1],
                    #         peaks_sort[2]]
                    #     )
                    #     missing_ord = [1]
                    # elif p > 1300 and len(peaks_sort) == 2:  # 2 orders missing
                    #     peaks_nord = np.array([
                    #         peaks_sort[0],
                    #         int((peaks_sort[0] + peaks_sort[1]) / 2),
                    #         peaks_sort[1],
                    #         int(peaks_sort[1] + (peaks_sort[1] - peaks_sort[0]) / 2)]
                    #     )
                    #     missing_ord = [1, 3]
            if missing_ord is None:  # e.g.: only 1 peak was found
                logging.info(f'Pixel {p} will default to simulation guess.')
                peaks_nord = [gen.nearest_idx(bin_centers, sim_phase[i, p]) for i in range(nord)]
                missing_ord = None

        phi_init = bin_centers[peaks_nord]
        amp_init = bin_counts[peaks_nord, p]

        # obtain Parameter object:
        params = init_params(phi_guess=phi_init, e_guess=energy[:, p], s_guess=sigmas[:, p], amp_guess=amp_init,
                             missing_ord=missing_ord)

        opt_params = minimize(
            fcn=fit_func,
            params=params,  # params
            args=(
                bin_centers,  # x_phases
                bin_counts[:, p],  # y_counts
                spectro.orders,  # orders
                p,  # pix
                leg_e,  # energy legendre poly object
                leg_s,  # sigma legendre poly object
                2,  # degree
                False,  # plot
            ),
            nan_policy='omit',
            xtol=xtol
        )

        opt_param_all.append(opt_params)
        # log which pixels failed to fit:
        if not opt_params.success:
            logging.warning(f'Pixel {p} failed to converge/fit.')
        red_chi2[p] = opt_params.redchi

        # extract the successfully fit parameters:
        fit_phi0, fit_e_coef, fit_s_coef, fit_amps = extract_params(params=opt_params.params, nord=nord)
        fit_e0 = e0_from_params(e1=fit_e_coef[0], e2=fit_e_coef[1], phi_0=fit_phi0)
        setattr(leg_e, 'coef', [fit_e0, fit_e_coef[0], fit_e_coef[1]])
        setattr(leg_s, 'coef', fit_s_coef)
        fit_phis = phis_from_grating_eq(orders=spectro.orders, phi_0=fit_phi0, leg=leg_e,
                                        coefs=[fit_e0, fit_e_coef[0], fit_e_coef[1]])
        fit_sigs = leg_s(leg_e(fit_phis))

        # get the order bin edges (requires explicit mu, sig, A):
        valid_idx = range(nord) if missing_ord is None else np.delete(range(nord), missing_ord)  # ords with non-0 amp
        if missing_ord is None:
            full_pix.append(p)
        inval_idx.append(missing_ord)
        for h, i in enumerate(valid_idx[:-1]):
            order_edges[i + 1, p] = gen.gauss_intersect(
                fit_phis[[i, valid_idx[h + 1]]],
                fit_sigs[[i, valid_idx[h + 1]]],
                fit_amps[[i, valid_idx[h + 1]]]
            )
        order_edges[:-1, p][order_edges[:-1, p] == 0] = np.nan  # the rest of the positions are invalid

        # re-histogram the photon table using the virtual pixel edges:
        ord_counts[valid_idx, p], _ = np.histogram(photons_pixel[p],
                                                   bins=order_edges[np.isfinite(order_edges[:, p]), p])

        # store model to array:
        gausses_i[:, :, p] = fit_func(params=opt_params.params, x_phases=fine_phase_grid, orders=spectro.orders,
                                      legendre_e=leg_e, legendre_s=leg_s)
        gausses[:, p] = np.sum(gausses_i[:, :, p], axis=1)

        if missing_ord is None:
            # find order-bleeding covar:
            covariance[:, :, p] = cov_from_params(params=opt_params.params, model=gausses[:, p], nord=nord,
                                                  order_edges=order_edges[np.isfinite(order_edges[:, p]), p],
                                                  valid_idx=valid_idx, x_phases=fine_phase_grid, orders=spectro.orders,
                                                  legendre_e=leg_e, legendre_s=leg_s, to_sum=True)

            # to plot covariance as errors, must sum the counts "added" from other orders as well as "stolen" by other orders
            # v giving order, > receiving order [g_idx, r_idx, pixel]
            #      9   8   7   6   5
            # 9 [  1   #   #   #   #  ]  < multiply counts*cov in Order 9 to add to other orders
            # 8 [  #   1   #   #   #  ]
            # 7 [  #   #   1   #   #  ]
            # 6 [  #   #   #   1   #  ]
            # 5 [  #   #   #   #   1  ]
            #      ^ multiply counts*cov in other orders to add to Order 9

            # apply the cov to data and extract errors:
            p_err[:, p] = np.array([int(np.sum(covariance[:, i, p] * ord_counts[:, p]) -
                                        covariance[i, i, p] * ord_counts[i, p]) for i in range(nord)])
            m_err[:, p] = np.array([int(np.sum(covariance[i, :, p] * ord_counts[:, p]) -
                                        covariance[i, i, p] * ord_counts[i, p]) for i in range(nord)])

        all_fit_phi[:, p] = fit_phis
        all_fit_sig[:, p] = fit_sigs

        # plot the individual pixels:
        if red_chi2[p] > redchi_val:
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
            ax2.figure.add_axes(ax2_2)

            # get the initial guess
            pre_gauss = fit_func(params, fine_phase_grid, orders=spectro.orders, legendre_e=leg_e,
                                 legendre_s=leg_s, to_sum=True)

            # get the initial residuals and weighted reduced chi^2:
            pre_residual = fit_func(params, bin_centers, y_counts=bin_counts[:, p], orders=spectro.orders,
                                    legendre_e=leg_e, legendre_s=leg_s)
            N = len(pre_residual)
            N_dof2 = N - opt_params.nvarys
            pre_red_chi2 = np.sum(pre_residual ** 2) / N_dof2

            # get the post residuals:
            opt_residual = fit_func(params=opt_params.params, x_phases=bin_centers, y_counts=bin_counts[:, p],
                                    orders=spectro.orders, legendre_e=leg_e, legendre_s=leg_s)

            ax1.grid()
            ax1.plot(bin_centers, bin_counts[:, p], 'k', label='Data')
            ax1.fill_between(bin_centers, 0, bin_counts[:, p], color='k')
            ax1.plot(fine_phase_grid, pre_gauss, color='gray', label='Init. Guess')
            quick_plot(ax1, [fine_phase_grid for i in range(nord)], gausses_i[:, :, p].T,
                       labels=[f'Order {i}' for i in spectro.orders[::-1]],
                       title=f'Least-Squares Fit', xlabel=r'Phase $\times 2\pi$', ylabel='Photon Count')
            ax1.plot(fine_phase_grid, gausses[:, p], 'y')
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
            R = gen.sig_to_R(s_eval, new_x)
            ax2_2.plot(new_x, R, color='k')
            for m, i in enumerate(fit_phis):
                ax2_2.plot(i, gen.sig_to_R(fit_sigs[m], i), '.', markersize=10, label=f'Order {spectro.orders[::-1][m]}')
            ax2.set_title(
                r'$E(\varphi)=$'f'{fit_e_coef[1]:.2e}P_2+{fit_e_coef[0]:.2f}P_1+{fit_e0:.2f}P_0\n'
                r'$\sigma(E)=$'f'{fit_s_coef[2]:.2e}P_2+{fit_s_coef[1]:.2e}P_1+{fit_s_coef[0]:.2e}P_0'
            )

            ax1.set_xticks([])  # removes axis labels
            ax2.set_xticks([])
            res1.set_xticks([])

            plt.show()
            pass

    # create off-blaze virtual pixels using average sigma to boundary:
    n_sigs_left = np.abs(all_fit_phi[1:, full_pix] - order_edges[1:-1, full_pix]) / all_fit_sig[1:, full_pix]
    n_sigs_right = np.abs(all_fit_phi[:-1, full_pix] - order_edges[1:-1, full_pix]) / all_fit_sig[:-1, full_pix]
    nsig_avg = np.average(np.append(n_sigs_left, n_sigs_right))

    for p in detector.pixel_indices:
        if inval_idx[p] is not None:
            # create new virtual pixel edges:
            for i in inval_idx[p]:
                if i != 0:
                    order_edges[i, p] = all_fit_phi[i - 1, p] + nsig_avg * all_fit_sig[i - 1, p]
                if i != nord - 1:
                    order_edges[i + 1, p] = all_fit_phi[i + 1, p] - nsig_avg * all_fit_sig[i + 1, p]

            # rerehistogram the photon table using the new virtual pixel edges:
            ord_counts[:, p], _ = np.histogram(photons_pixel[p], bins=order_edges[:, p])

            # redo order-bleeding covar:
            covariance[:, :, p] = cov_from_params(params=opt_param_all[p].params, model=gausses[:, p], nord=nord,
                                                  order_edges=order_edges[:, p], valid_idx=range(nord),
                                                  x_phases=fine_phase_grid, orders=spectro.orders,
                                                  legendre_e=leg_e, legendre_s=leg_s, to_sum=True)

            # redo errors:
            p_err[:, p] = np.array([int(np.sum(covariance[:, i, p] * ord_counts[i, p]) -
                                        covariance[i, i, p] * ord_counts[i, p]) for i in range(nord)])
            m_err[:, p] = np.array([int(np.sum(covariance[i, :, p] * ord_counts[:, p]) -
                                        covariance[i, i, p] * ord_counts[i, p]) for i in range(nord)])

    idxs = np.where(red_chi2 > redchi_val)
    logging.info(f'Number of pixels with red-chi2 less than {redchi_val}: {sim.npix - len(idxs[0])}')

    if plot:
        gausses[:, [idxs]] = 0  # suppresses any clearly poor fits
        gausses[gausses < 0.01] = 0.01  # fills in the white spaces of the graph nicely, doesn't mean anything
        plt.imshow(gausses[::-1], extent=[1, sim.npix, bin_centers[0], bin_centers[-1]], aspect='auto', norm=LogNorm())
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Photon Count')
        plt.title("Fit MSF Model")
        plt.xlabel("Pixel Index")
        plt.ylabel(r"Phase ($\times \pi /2$)")
        plt.tight_layout()
        plt.show()

        # plot the chi2 of all the pixels
        plt.grid()
        plt.plot(detector.pixel_indices[red_chi2 < 1e6], red_chi2[red_chi2 < 1e6], '.', markersize=3)
        plt.axhline(redchi_val, color='red', label='Cut-off')
        plt.semilogy()
        plt.title(r'Weighted Reduced $\chi^2$ for all pixels')
        plt.ylabel(r'Weighted Red. $\chi^2$')
        plt.xlabel('Pixel Index')
        plt.legend()
        plt.show()

        # plot the spectrum with error band, blaze left intact:
        fig, ax = plt.subplots(int(np.ceil(nord/2)), 2, figsize=(30, int(10*nord)), sharex=True)
        axes = ax.ravel()
        for i in range(nord):
            spec_w_merr = (ord_counts[i] - m_err[i])
            spec_w_perr = (ord_counts[i] + p_err[i])
            spec_w_merr[spec_w_merr < 0] = 0
            axes[i].grid()
            axes[i].fill_between(pixels, spec_w_merr, spec_w_perr, edgecolor='r', facecolor='r', linewidth=0.5)
            axes[i].plot(pixels, ord_counts[i])
            axes[i].set_title(f'Order {spectro.orders[::-1][i]}')
        axes[-1].set_xlabel("Pixel Index")
        axes[-2].set_xlabel("Pixel Index")
        axes[0].set_ylabel('Photon Count')
        axes[2].set_ylabel('Photon Count')
        plt.suptitle('Extracted Calibration Spectrum')
        plt.tight_layout()
        plt.show()

    logging.info(f"Finished fitting all pixels across all orders.")

    # assign bin edges, covariance matrices, virtual pix centers, and simulation settings to MSF class and save:
    covariance = np.nan_to_num(covariance)
    msf = MKIDSpreadFunction(bin_edges=order_edges, cov_matrix=covariance, waves=all_fit_phi, sim_settings=sim)
    msf_file = f'{outdir}/msf.npz'
    msf.save(msf_file)
    logging.info(f'\nSaved MSF bin edges and covariance matrix to {msf_file}.')
    return msf_file


if __name__ == '__main__':
    tic = time.perf_counter()  # recording start time for script

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
    parser.add_argument('outdir',
                        metavar='OUTPUT_DIRECTORY',
                        help='Directory for the output files (str).')
    parser.add_argument('caltable',
                        metavar='CALIBRATION_PHOTON_TABLE',
                        help='Directory/name of the calibration photon table, must be from simulation.')

    # optional MSF args:
    parser.add_argument('--bin_range', default=(-1, 0), type=tuple,
                        help='Start and stop of range for histogram.')
    parser.add_argument('--plot', action='store_true', default=False, type=bool, help='If passed, plots will be shown.')

    # set arguments as variables
    args = parser.parse_args()

    # ==================================================================================================================
    # CHECK AND/OR CREATE DIRECTORIES
    # ==================================================================================================================
    os.makedirs(name=f'{args.output_dir}', exist_ok=True)

    # ==================================================================================================================
    # START LOGGING TO FILE
    # ==================================================================================================================
    now = dt.now()
    logging.basicConfig(filename=f'{args.outdir}/{now.strftime("%Y%m%d_%H%M%S")}_msf.log',
                        format='%(levelname)s:%(message)s', level=logging.INFO)
    logging.info(msg="The process of modeling the MKID Spread Function (MSF) is recorded."
                     f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    cal_table = Photontable(file_name=args.caltable)

    # ==================================================================================================================
    # MSF EXTRACTION STARTS
    # ==================================================================================================================

    fitmsf(
        msf_table=cal_table,
        sim=cal_table.query_header('sim_settings'),
        outdir=args.outdir,
        bin_range=args.bin_range,
        plot=args.plot
    )

    logging.info(f'\nTotal script runtime: {((time.perf_counter() - tic) / 60):.2f} min.')
    # ==================================================================================================================
    # MSF EXTRACTION ENDS
    # ==================================================================================================================

    # ==================================================================================================================
    # DEBUGGING PLOTS
    # ==================================================================================================================
    # retrieving the theoretical blazed calibration spectrum shape:
    # spectra = get_spectrum(sim.type_spectra)
    # spectra = apply_bandpass(spectra, bandpass=[FineGrid(sim.minwave, sim.maxwave)])
    # spectra = clip_spectrum(spectra, clip_range=(sim.minwave, sim.maxwave))

    # blazed_spectrum, _, _ = eng.blaze(spectra.waveset, spectra)
    # blazed_spectrum = np.nan_to_num(blazed_spectrum)
    # pix_leftedge = spectro.pixel_wavelengths(edge='left')
    # blaze_shape = np.array([eng.lambda_to_pixel_space(spectra.waveset, blazed_spectrum[i],
    #                                                   pix_leftedge[i]) for i in range(nord)])[::-1]
    # blaze_shape = np.nan_to_num(blaze_shape)
    # blaze_shape /= np.max(blaze_shape)  # normalize max to 1
    # blaze_shape[blaze_shape < 1e-4] = 1  # prevent divide by 0 or close to 0

