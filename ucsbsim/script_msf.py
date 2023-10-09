import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
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
    parser.add_argument('-nb', '--n_bins',
                        metavar='NUM_OF_BINS',
                        default=180,
                        help='Number of bins to use in histogram of any pixel.')
    parser.add_argument('-br', '--bin_range',
                        metavar='BIN_RANGE',
                        default=(-2, 0),
                        help='Tuple containing start and stop range for histogram of any pixel.')
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

    resid_map = np.arange(sim.npix, dtype=int) * 10 + 100  # TODO replace once known
    phases = pt.query(column='wavelength')
    resID = pt.query(column='resID')
    idx = [np.where(resID == resid_map[j]) for j in range(sim.npix)]
    photons_pixel = [phases[idx[j]].tolist() for j in range(sim.npix)]

    # ==================================================================================================================
    # INSTANTIATE SPECTROGRAPH & DETECTOR
    # ==================================================================================================================
    if not os.path.exists(sim.R0s_file):
        IOError('File does not exist, check path and file name.')
    else:
        R0s = np.loadtxt(sim.R0s_file, delimiter=',')
        logging.info(f'\nThe individual R0s were imported from {sim.R0s_file}.')

    detector = MKIDDetector(sim.npix, sim.pixelsize, sim.designR0, sim.l0, R0s, None, resid_map)
    grating = GratingSetup(sim.alpha, sim.delta, sim.groove_length)
    spectro = SpectrographSetup(sim.m0, sim.m_max, sim.l0, sim.pixels_per_res_elem, sim.focallength, grating, detector)
    eng = engine.Engine(spectro)
    nord = spectro.nord
    sim_phase = wave_to_phase(spectro.pixel_wavelengths().to(u.nm)[::-1], minwave=sim.minwave, maxwave=sim.maxwave)
    # flip order axis, ascending phase

    # ==================================================================================================================
    # MSF EXTRACTION STARTS
    # ==================================================================================================================
    wave = np.linspace(sim.minwave.to(u.nm).value - 100, sim.maxwave.to(u.nm).value + 100, 10000) * u.nm

    # retrieving the blazed calibration spectrum shape and converting to counts in pixel-space phases:
    spectra = get_spectrum(sim.type_spectra)
    blazed_spectrum, _, _ = eng.blaze(wave, spectra)
    pix_leftedge = spectro.pixel_wavelengths(edge='left').to(u.nm).value
    blaze_shape = [eng.lambda_to_pixel_space(wave, blazed_spectrum[i], pix_leftedge[i]) for i in range(nord)][::-1]
    blaze_shape /= np.max(blaze_shape)  # normalize max to 1

    # initializing empty arrays for msf products:
    covariance = np.zeros([nord, nord, sim.npix])
    photon_bins = np.zeros([nord + 1, sim.npix])
    photon_bins[0, :] = -2

    # initializing empty arrays for recovered Gaussian fits:
    phase_centers = np.full([nord, sim.npix], fill_value=np.nan)
    phase_sigmas = np.empty([nord, sim.npix])
    phase_counts = np.empty([nord, sim.npix])
    redchi = np.empty(sim.npix)

    # initializing empty arrays to assist with cross-correlation and order-chooser:
    res_sq = np.zeros([nord, sim.npix])
    bin_edges = np.linspace(args.bin_range[0], args.bin_range[1], args.n_bins + 1, endpoint=True)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    bin_counts = np.zeros([args.n_bins, sim.npix])
    template = np.zeros([args.n_bins, sim.npix])
    peaks_in_pixel = np.full(sim.npix, fill_value=nord)

    # do least-squares fit to get out Gaussian mu, sig, A:
    for j in detector.pixel_indices:
        (mu, sig, A), bin_counts[:, j], redchi[j] = eng.fit_gaussians(j,
                                                                      photons_pixel[j],
                                                                      nord,
                                                                      bin_edges=bin_edges,
                                                                      plot=args.plotresults)

        phase_centers[:len(mu), j], phase_sigmas[:len(mu), j], phase_counts[:len(mu), j] = mu, sig, A

        if len(mu) != nord:
            peaks_in_pixel[j] = len(mu)

    # fit the points to a trendline:
    line = []
    for n, i in enumerate(spectro.orders[::-1]):
        coef = np.polyfit(detector.pixel_indices[~np.isnan(phase_centers[n])],
                          phase_centers[n][~np.isnan(phase_centers[n])],
                          1)
        line.append(np.poly1d(coef))
        res_sq[n][~np.isnan(phase_centers[n])] = (phase_centers[n][~np.isnan(phase_centers[n])] -
                                                  line[n](detector.pixel_indices)[~np.isnan(phase_centers[n])]) ** 2

    # compute the RMS for each pixel:
    pixel_rms = np.sqrt(np.mean(res_sq, axis=0))

    middle = int(args.n_bins / 2)
    # conduct cross-correlation to reduce phase offset:
    for j in detector.pixel_indices:
        # generate a cross-correlation template using the theoretical blaze:
        template[:, j] = np.sum(engine.gauss(bin_centers,
                                             sim_phase[:, j][:, None],
                                             np.array([3 * (bin_centers[1] - bin_centers[0])] * nord)[:, None],
                                             blaze_shape[:, j][:, None]),
                                axis=1)

        # plot the pre-cross correlation fits to see it working
        if args.plotresults:
            fig, ax = plt.subplots(1, 1)
            quick_plot(ax, [bin_centers], [bin_counts[:, j]], labels=['Data'], first=True, color='k')
            ax.fill_between(bin_centers, 0, bin_counts[:, j], color='k')
            x = [np.linspace(args.bin_range[0], args.bin_range[-1], 1000) for i in range(nord)]
            quick_plot(ax, x, engine.gauss(x,
                                           phase_centers[:, j][:, None],
                                           phase_sigmas[:, j][:, None],
                                           phase_counts[:, j][:, None]).T,
                       labels=[f'Order {i}' for i in spectro.orders[::-1]],
                       title=f'Pixel {j}, Pre-Cross Correlation', xlabel='Phase', ylabel='Photon Count')
            quick_plot(ax,
                       [bin_centers],
                       [template[:, j] * np.max(phase_counts[:, j])],
                       linestyle='--',
                       color='pink',
                       labels=['X-Correlation'])
            plt.legend()
            plt.show()

        # do the cross-correlation and pick out the best offset:
        corr = scipy.signal.correlate(bin_counts[:, j], template[:, j], mode='same')
        best = np.argmax(corr)
        diff = int(middle - best)

        if diff != 0:
            # shift the optimized mu and data by the proper phase shift:
            phase_shift = diff * (bin_centers[1] - bin_centers[0])
            phase_centers[:, j] += phase_shift
            if diff < 0:
                bin_counts[:, j] = np.array(list(bin_counts[np.abs(diff):, j]) + [0] * np.abs(diff))
            elif diff > 0:
                bin_counts[:, j] = np.array([0] * np.abs(diff) + list(bin_counts[:-diff, j]))

        # recompute the residual for the shift:
        line_says = np.array([line[n](j) for n, i in enumerate(spectro.orders[::-1])])
        pixel_rms[j] = np.sqrt(np.mean(
            (phase_centers[:, j][~np.isnan(phase_centers[:, j])] - line_says[~np.isnan(phase_centers[:, j])]) ** 2))

    freq_domain = np.linspace(-2, 0, 10000)
    # try assigning peaks to other orders by minimizing residual, when a pixel has less than n_ord peaks:
    for j in detector.pixel_indices:
        if peaks_in_pixel[j] < nord:
            # grab the current phases and RMS and copy them
            best_center = phase_centers[:, j]
            best_count = phase_counts[:, j]
            best_sigma = phase_sigmas[:, j]
            best_rms = pixel_rms[j]

            # iterate through each possible combination of orders:
            for n, comb in enumerate(itertools.combinations(range(nord), peaks_in_pixel[j])):
                new_center = np.full(nord, fill_value=np.nan)
                new_count = np.full(nord, fill_value=np.nan)
                new_sigma = np.full(nord, fill_value=np.nan)
                comb = list(comb)
                new_center[comb] = phase_centers[:peaks_in_pixel[j], j]
                new_count[comb] = phase_counts[:peaks_in_pixel[j], j]
                new_sigma[comb] = phase_sigmas[:peaks_in_pixel[j], j]

                # calculate the residual:
                line_says = np.array([line[n](j) for n, i in enumerate(spectro.orders[::-1])])
                rms_pixel = np.sqrt(np.sum((new_center[~np.isnan(new_center)] - line_says[~np.isnan(new_center)]) ** 2))

                # only if the found residual is better, replace the copy above:
                if rms_pixel < best_rms:
                    best_center = new_center
                    best_count = new_count
                    best_sigma = new_sigma
                    best_rms = rms_pixel

            # only if the best residual copy is better than what it is currently, replace in array:
            if best_rms < pixel_rms[j]:
                phase_centers[:, j] = best_center
                phase_counts[:, j] = best_count
                phase_sigmas[:, j] = best_sigma
                pixel_rms[j] = best_rms

                res_sq = np.zeros([nord, sim.npix])
                # recalculate the trendlines for the entire spectrum:
                for n, i in enumerate(spectro.orders[::-1]):
                    coef = np.polyfit(detector.pixel_indices[~np.isnan(phase_centers[n])],
                                      phase_centers[n][~np.isnan(phase_centers[n])], 1)
                    line[n] = np.poly1d(coef)
                    res_sq[n][~np.isnan(phase_centers[n])] = (phase_centers[n][~np.isnan(phase_centers[n])] -
                                                              line[n](detector.pixel_indices)[
                                                                  ~np.isnan(phase_centers[n])]) ** 2
                pixel_rms = np.sqrt(np.sum(res_sq, axis=0))

            # get the order edges, with special treatment for less than n_ord peaks:
            # find for which orders the peaks exist:
            exist_idx = np.argwhere(~np.isnan(phase_centers[:, j]))
            for n, i in enumerate(exist_idx[:-1]):
                # if there exists adjacent order peaks, just get the overlap point for order edge:
                if i+1 == exist_idx[n+1]:
                    photon_bins[i+1, j] = engine.gauss_intersect(phase_centers[[i, i+1], j],
                                                                 phase_sigmas[[i, i+1], j],
                                                                 phase_counts[[i, i+1], j])
                # if there is an order gap between 2 peaks, divide the difference between the 2 peaks into 4
                # and use the 1st and 3rd quadrant as edges--->   ,_n_,__,_n+2_,    :
                elif i+2 == exist_idx[n+1]:
                    two_peak_width = phase_centers[i+2, j] - phase_centers[i, j]
                    quarter = two_peak_width/4
                    photon_bins[i+1, j] = phase_centers[i, j] + quarter
                    photon_bins[i + 2, j] = phase_centers[i, j] + 3*quarter
            # if the 1st order is missing:
            if exist_idx[0] != 0:
                photon_bins[1, j] = photon_bins[2, j] - (photon_bins[3, j] - photon_bins[2, j])
            # if the last order is missing:
            if exist_idx[-1] != nord - 1:
                photon_bins[-2, j] = photon_bins[-3, j] + (photon_bins[-3, j] - photon_bins[-4, j])
        else:
            photon_bins[1:-1, j] = engine.gauss_intersect(phase_centers[:, j], phase_sigmas[:, j], phase_counts[:, j])

        # calculate the covariance between orders:
        gauss_sum = np.sum(engine.gauss(freq_domain, phase_centers[:, j][:, None],
                                        phase_sigmas[:, j][:, None],
                                        phase_counts[:, j][:, None]),
                           axis=0) * (freq_domain[1] - freq_domain[0])
        gauss_int = [interp.InterpolatedUnivariateSpline(freq_domain,
                                                         engine.gauss(freq_domain,
                                                                      phase_centers[i, j],
                                                                      phase_sigmas[i, j],
                                                                      phase_counts[i, j]),
                                                         k=1,
                                                         ext=1) for i in range(nord)]
    
        covariance[:, :, j] = [[gauss_int[i].integral(photon_bins[k, j],
                                                      photon_bins[k + 1, j]) / gauss_sum[i]
                                for k in range(nord)]
                               for i in range(nord)]

    # plot each order with trendline and assigned peak phases:
    for n, i in enumerate(spectro.orders[::-1]):
        plt.grid()
        plt.title(f"Phases and Fit, Order {i}")
        plt.plot(detector.pixel_indices, phase_centers[n], '.', markersize=1, label='Data')
        plt.plot(detector.pixel_indices, line[n](detector.pixel_indices), '--', label='Fit')
        plt.xlabel("Pixel Index")
        plt.ylabel(r"Phase ($\times \pi /2$)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # plot the entire histogram with overlaid peaks and trendlines:
    plt.figure(figsize=(10, 8))
    hist_array = np.empty([detector.n_pixels, 90])
    for j in detector.pixel_indices:
        if photons_pixel[j]:
            counts, edges = np.histogram(photons_pixel[j], bins=int(args.n_bins / 2), range=(-1, 0))
            hist_array[j, :] = np.array([float(x) for x in counts])
    plt.imshow(hist_array[:, ::-1].T,
               extent=[0, detector.n_pixels - 1, -1, 0],
               aspect='auto',
               cmap='gray',
               norm=colors.LogNorm())
    for n, i in enumerate(spectro.orders[::-1]):
        plt.plot(detector.pixel_indices[~np.isnan(phase_centers[n])],
                 phase_centers[n][~np.isnan(phase_centers[n])],
                 '.',
                 label=f'O{i}')
        plt.plot(detector.pixel_indices,
                 line[n](detector.pixel_indices),
                 label=f'O{i} Fit')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Photon Count')
    plt.title("Photons Binned in Phase & Pixel with Order-Assigned Peaks and Trendlines")
    plt.xlabel("Pixel Index")
    plt.ylabel(r"Phase ($\times \pi /2$)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plot the final fits, phase shift, and template for each pixel:
    if args.plotresults:
        for j in detector.pixel_indices:
            fig, ax = plt.subplots(1, 1)
            quick_plot(ax, [bin_centers], [bin_counts[:, j]], labels=['Data'], first=True, color='k')
            ax.fill_between(bin_centers, 0, bin_counts[:, j], color='k')
            x = [np.linspace(args.bin_range[0], args.bin_range[-1], 1000) for i in range(nord)]
            quick_plot(ax, x, engine.gauss(x,
                                           phase_centers[:, j][:, None],
                                           phase_sigmas[:, j][:, None],
                                           phase_counts[:, j][:, None]).T,
                       labels=[f'Order {i}' for i in spectro.orders[::-1]],
                       title=f'Least-Squares Fit for Pixel {j}', xlabel='Phase', ylabel='Photon Count')
            quick_plot(ax,
                       [bin_centers],
                       [template[:, j] * np.max(phase_counts[:, j])],
                       linestyle='--',
                       color='pink',
                       labels=['X-Correlation'])
            for b in photon_bins[:, j]:
                ax.axvline(b, linestyle='--', color='black')
            plt.legend()
            plt.show()

    logging.info(f"Finished fitting all {sim.npix} pixels.")

    # assign bin edges, covariance matrices, bin centers, and simulation settings to MSF class and save:
    covariance = np.nan_to_num(covariance)
    msf = MKIDSpreadFunction(bin_edges=photon_bins, cov_matrix=covariance, waves=phase_centers,
                             sim_settings=sim)
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

    # plot reduced chi-sq (should not contain significant outliers):
    fig1, ax1 = plt.subplots(1, 1)
    quick_plot(ax1, [range(sim.npix)], [redchi], title='Reduced Chi-Sq', xlabel='Pixel', first=True)
    plt.show()

    # plot unblazed, unshaped calibration spectrum (should be mostly flat line):
    fig2, ax2 = plt.subplots(1, 1)
    quick_plot(ax2, phase_centers, phase_counts / blaze_shape, labels=[f'O{i}' for i in spectro.orders[::-1]],
               first=True, title='Calibration Spectrum (Blaze Divided Out)', xlabel='Phase',
               ylabel='Photon Count', linestyle='None', marker='.')
    ax2.set_yscale('log')
    plt.show()
    
    plt.grid()
    for i in range(1, nord):
        plt.plot(photon_bins[i])
    plt.title("Bin edges between orders")
    plt.ylabel("Phase")
    plt.xlabel("Pixel Index")
    plt.show()
