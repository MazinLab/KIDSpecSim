import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy
import scipy.interpolate as interp
from scipy.stats import norm
import astropy.units as u
import time
from datetime import datetime as dt
import logging
import argparse
from specutils import Spectrum1D
from synphot import SpectralElement

from mkidpipeline.photontable import Photontable
from ucsbsim.mkidspec.spectra import get_spectrum, apply_bandpass, AtmosphericTransmission, FilterTransmission, \
    TelescopeTransmission, FineGrid, clip_spectrum
from ucsbsim.mkidspec.spectrograph import GratingSetup, SpectrographSetup
from ucsbsim.mkidspec.detector import MKIDDetector, wave_to_phase
import ucsbsim.mkidspec.engine as engine
from ucsbsim.mkidspec.plotting import quick_plot
from ucsbsim.mkidspec.simsettings import SpecSimSettings

# TODO this will need work as the pipeline will probably default to MEC HDF headers
from mkidpipeline.steps.buildhdf import buildfromarray

"""
Simulation of an MKID spectrograph observation run. The steps are:
-A source spectrum is loaded.
-Atmosphere, telescope, and/or filter bandpasses may be applied depending on whether source originates from sky or lab.
-It is multiplied by the blaze efficiency of the grating, which determines how much of the flux that got through the
 filter(s) will be incident on each pixel of the detector.
-It is broadened according to the optical LSF, the limit to the resolution of the optics used.
-It is convolved with the MKID resolution width as a function of wavelength to simulate MKID wavelength discrimination.
 This puts the spectrum into pixel space (flux) as each pixel has a different dlambda, whereas before it was in
  wavelength space (flux density).
-The photons are randomly assigned phase and timestamp according to Poisson statistics and MKID-specific properties
 such as dead time and minimum trigger energy.
-The photon table is saved to an h5 file.
"""

if __name__ == '__main__':
    tic = time.perf_counter()  # recording start time for script

    # ==================================================================================================================
    # PARSE COMMAND LINE ARGUMENTS
    # ==================================================================================================================
    arg_desc = '''
               Simulate an MKID Spectrograph observation run.
               ---------------------------------------------------------
               This program loads, filters, blazes, convolves, and converts a synphot spectrum to a photon table.
               '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)

    # required simulation args:
    parser.add_argument('output_dir',
                        metavar='OUTPUT_DIRECTORY',
                        help='Directory for the output files. (str)')
    parser.add_argument('R0s_file',
                        metavar='R0S_FILE_NAME',
                        help="Directory/filename of the R0s file, will be created if it doesn't exist. (str)")
    parser.add_argument('phaseoffset_file',
                        metavar='PHASE_OFFSET_FILE_NAME',
                        help="Directory/filename of the phase offset file, will be created if it doesn't exist. (str)")
    parser.add_argument('type_spectra',
                        metavar='TYPE_SPECTRA',
                        help='The type of spectra: can be "blackbody", "phoenix", "flat", or "emission". (str)')

    # optional simulation args:
    parser.add_argument('-st', '--spectro_type',
                        metavar='SPECTROGRAPH_TYPE',
                        default='default',
                        help='Type of spectrograph settings, in general. E.g.: "default", "FWHMlimit", etc.')
    parser.add_argument('-ef', '--emission_file',
                        metavar='EMISSION_FILE_NAME',
                        default=None,
                        help='Directory/filename of NIST emission line spectrum, REQUIRED if spectra is "emission".')
    parser.add_argument('-dist', '--distance',
                        metavar='DISTANCE_PS',
                        default=2.64,  # default is based on Sirius A, brightest star in the night sky
                        help='Distance to target star in parsecs, REQUIRED if spectra is "phoenix". (float)')
    parser.add_argument('-rad', '--radius',
                        metavar='RADIUS_RSUN',
                        default=1.711,  # default is based on Sirius A, brightest star in the night sky
                        help='Radius of target star in R_sun, REQUIRED if spectra is "phoenix". (float)')
    parser.add_argument('-T', '--temp',
                        metavar='TEMP_K',
                        default=9940,  # default is based on Sirius A, brightest star in the night sky
                        help='Temperature of target in K, REQUIRED if spectra is "blackbody" or "phoenix". (float)')
    parser.add_argument('-se', '--sky_emission',
                        metavar='NIGHT_SKY_EMISSION_LINES',
                        default=False,
                        help='Whether to add night sky emission lines to the spectrum.'
                             'Always "False" for laboratory simulations. (bool)')
    parser.add_argument('-et', '--exptime',
                        metavar='EXPOSURE_TIME_S',
                        default=50,
                        help='The total exposure time of the observation in seconds. (float)')
    parser.add_argument('-ta', '--telearea',
                        metavar='TELESCOPE_AREA_CM2',
                        default=np.pi * 4 ** 2,
                        help='The telescope area in cm2. (float)')
    parser.add_argument('-sc', '--simpconvol',
                        action='store_true',
                        default=False,
                        help='If passed, indicates that a faster, simplified convolution should be conducted.')
    parser.add_argument('-sky', '--on_sky',
                        action='store_true',
                        default=False,
                        help='If passed, the observation is conducted "on-sky" instead of in the laboratory and'
                             'indicates the spectrum will be multiplied with atmospheric/telescopic attenuation.')
    parser.add_argument('-ref', '--reflect',
                        metavar='TELESCOPE_REFLECTIVITY',
                        default=0.9,
                        help='Factor to attenuate spectrum due to telescope reflectivity, between 0 and 1.'
                             'REQUIRED if "on_sky" is True. (float)')
    parser.add_argument('-minw', '--minwave',
                        metavar='MINIMUM_WAVELENGTH_NM',
                        default=400,
                        help='The minimum wavelength of the spectrograph in nm. (float)')
    parser.add_argument('-maxw', '--maxwave',
                        metavar='MAXIMUM_WAVELENGTH_NM',
                        default=800,
                        help='The maximum wavelength of the spectrograph in nm. (float)')

    # optional spectrograph args:
    parser.add_argument('-np', '--npix',
                        metavar='NUMBER_OF_PIXELS',
                        default=2048,
                        help='The number of pixels in the MKID detector. (int)')
    parser.add_argument('-ps', '--pixelsize',
                        metavar='PIXEL_SIZE_UM',
                        default=20,
                        help='The length of the MKID pixel in the dispersion direction in um. (float)')
    parser.add_argument('-R0', '--designR0',
                        metavar='DESIGN_R0',
                        default=15,
                        help='The spectral resolution at the maximum wavelength. (float)')
    parser.add_argument('-l0', '--l0',
                        metavar='LAMBDA_0_NM',
                        default=800,
                        help="The longest wavelength in nm. Can be float or 'same' to be equal to 'maxwave' arg.")
    parser.add_argument('-os', '--osamp',
                        metavar='OVERSAMPLING',
                        default=10,
                        help='The number of samples to use for the smallest pixel dlambda during convolution. (int)')
    parser.add_argument('-ns', '--nsig',
                        metavar='NUMBER_OF_SIGMA',
                        default=3,
                        help='The number of sigma to use for Gaussian during convolution. (int)')
    parser.add_argument('-a', '--alpha',
                        metavar='INCIDENCE_ANGLE',
                        default=28.3,
                        help='Alpha, the angle of incidence on the grating in degrees. (float)')
    parser.add_argument('-b', '--beta',
                        metavar='REFLECTANCE_ANGLE',
                        default=28.3,
                        help='Beta, the reflectance angle at the central pixel in degrees. (float)')
    parser.add_argument('-del', '--delta',
                        metavar='BLAZE_ANGLE',
                        default=63,
                        help='Delta, the grating blaze angle in degrees. (float)')
    parser.add_argument('-d', '--groove_length',
                        metavar='GROOVE_LENGTH',
                        default=((1 / 316) * u.mm).to(u.nm).value,
                        help='The groove length d, or distance between slits, of the grating in nm. (float)')
    parser.add_argument('-m0', '--m0',
                        metavar='INITIAL_ORDER',
                        default=3,
                        help='The initial order, at the longer wavelength end. (int)')
    parser.add_argument('-mm', '--m_max',
                        metavar='FINAL_ORDER',
                        default=7,
                        help='The final order, at the shorter wavelength end. (int)')
    parser.add_argument('-ppre', '--pixels_per_res_elem',
                        metavar='PIXELS_PER_RESOLUTION_ELEMENT',
                        default=2.5,
                        help='Number of pixels per spectral resolution element for the spectrograph. (float)')
    parser.add_argument('-fl', '--focallength',
                        metavar='FOCAL_LENGTH_MM',
                        default=300,
                        help='The focal length of the detector in mm. (float)')

    # get arguments & simulation settings
    args = parser.parse_args()
    sim = SpecSimSettings(
        minwave_nm=args.minwave,
        maxwave_nm=args.maxwave,
        npix=args.npix,
        pixelsize_um=args.pixelsize,
        designR0=args.designR0,
        l0_nm=args.l0,
        alpha_deg=args.alpha,
        delta_deg=args.delta,
        beta_deg=args.beta,
        groove_length_nm=args.groove_length,
        m0=args.m0,
        m_max=args.m_max,
        pixels_per_res_elem=args.pixels_per_res_elem,
        focallength_mm=args.focallength,
        R0s_file=args.R0s_file,
        phaseoffset_file=args.phaseoffset_file,
        type_spectra=args.type_spectra,
        emission_file=args.emission_file,
        exptime_s=args.exptime,
        telearea_cm2=args.telearea,
        distance_ps=args.distance,
        radius_Rsun=args.radius,
        temp_K=args.temp,
        sky_emission_lines=args.sky_emission,
        simpconvol=args.simpconvol
    )
    sigma_frac = norm.cdf(args.nsig)  # convert number of sigma to a fraction

    sys.path.insert(1, '/home/kimc/pycharm/KIDSpecSim/ucsbsim')

    # ==================================================================================================================
    # CHECK FOR OR CREATE DIRECTORIES
    # ==================================================================================================================
    now = dt.now()
    os.makedirs(name=f'{args.output_dir}/{now.strftime("%y%m%d")}_{args.spectro_type}/logging/', exist_ok=True)
    os.makedirs(name=os.path.dirname(sim.R0s_file), exist_ok=True)
    os.makedirs(name=os.path.dirname(sim.phaseoffset_file), exist_ok=True)

    # ==================================================================================================================
    # START LOGGING TO FILE
    # ==================================================================================================================
    logging.basicConfig(filename=f'{args.output_dir}/{now.strftime("%y%m%d")}_{args.spectro_type}'
                                 f'/logging/simulate_{sim.type_spectra}.log',
                        format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info(msg=f"The simulation of a(n) {sim.type_spectra} spectrum observation is recorded."
                     f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    # ==================================================================================================================
    # INSTANTIATE SPECTROGRAPH & DETECTOR
    # ==================================================================================================================
    try:  # check for the spectral resolution file, create if not exist
        R0s = np.loadtxt(fname=sim.R0s_file, delimiter=',')
        logging.info(f'\nThe individual R0s were imported from {sim.R0s_file}.')
    except IOError as e:
        logging.info(msg=e)
        R0s = np.random.uniform(low=.85, high=1.15, size=sim.npix) * sim.designR0
        np.savetxt(fname=sim.R0s_file, X=R0s, delimiter=',')
        logging.info(msg=f'The individual R0s were generated randomly from the design R0 and saved to {sim.R0s_file}.')

    try:  # check for the phase offset file, create if not exist
        phase_offsets = np.loadtxt(fname=sim.phaseoffset_file, delimiter=',')
        logging.info(msg=f'\nThe pixel center phase offsets were imported from {sim.phaseoffset_file}.')
    except IOError as e:
        logging.info(msg=e)
        phase_offsets = np.random.uniform(low=.8, high=1.2, size=sim.npix)
        np.savetxt(fname=sim.phaseoffset_file, X=phase_offsets, delimiter=',')
        logging.info(msg=f'The pixel phase offsets were generated randomly and saved to {sim.phaseoffset_file}.')

    # TODO: replace resonator IDs once known
    resid_map = np.arange(sim.npix, dtype=int) * 10 + 100

    detector = MKIDDetector(n_pix=sim.npix, pixel_size=sim.pixelsize, design_R0=sim.designR0, l0=sim.l0, R0s=R0s,
                            phase_offsets=phase_offsets, resid_map=resid_map)
    grating = GratingSetup(alpha=sim.alpha, delta=sim.delta, beta_center=sim.beta, groove_length=sim.groove_length)
    spectro = SpectrographSetup(order_range=sim.order_range, final_wave=sim.l0,
                                pixels_per_res_elem=sim.pixels_per_res_elem,
                                focal_length=sim.focallength, grating=grating, detector=detector)
    eng = engine.Engine(spectrograph=spectro)

    # shorten commonly used properties:
    nord = spectro.nord
    lambda_pixel = spectro.pixel_wavelengths().to(u.nm)
    lambda_left = spectro.pixel_wavelengths(edge='left')

    # ==================================================================================================================
    # SIMULATION STARTS
    # ==================================================================================================================

    # obtaining spectra:
    spectrum = get_spectrum(spectrum_type=sim.type_spectra, distance=sim.distance, radius=sim.radius, teff=sim.temp,
                            emission_file=sim.emission_file, minwave=sim.minwave, maxwave=sim.maxwave)

    # populate bandpasses based on on-sky or lab:
    # always have finer grid spacing and min/max filter:
    bandpasses = [FineGrid(min=sim.minwave, max=sim.maxwave)]
    if args.on_sky:
        bandpasses.append(AtmosphericTransmission())
        if sim.sky_emission_lines:
            spectrum = apply_bandpass(spectra=spectrum,
                                      bandpass=bandpasses)  # apply bandpasses prior to night sky emission
            # TODO add the night sky emission line option
            pass
        bandpasses.append(TelescopeTransmission(reflectivity=args.reflect))

    # apply bandpasses and obtain associated noise:
    bandpasses.append(FilterTransmission(min=sim.minwave, max=sim.maxwave))
    bandpass_spectrum = apply_bandpass(spectra=spectrum, bandpass=bandpasses)

    # clip spectrum in order to blaze within limits
    clipped_spectrum = clip_spectrum(x=bandpass_spectrum, clip_range=(sim.minwave, sim.maxwave))

    # blaze spectrum and directly integrate into pixel space to begin noise calculation:
    blazed_spectrum, masked_waves, masked_blaze = eng.blaze(wave=clipped_spectrum.waveset, spectra=clipped_spectrum)
    blazed_int_spec = np.array([
        eng.lambda_to_pixel_space(
            array_wave=clipped_spectrum.waveset,
            array=blazed_spectrum[i],
            leftedge=lambda_left[i]
        ) for i in range(nord)
    ])

    # optically-broaden spectrum and obtain associated noise:
    broadened_spectrum = eng.optically_broaden(wave=clipped_spectrum.waveset, flux=blazed_spectrum)
    broad_int_spec = np.array([
        eng.lambda_to_pixel_space(
            array_wave=clipped_spectrum.waveset,
            array=broadened_spectrum[i],
            leftedge=lambda_left[i]
        ) for i in range(nord)
    ])
    broad_noise = np.nan_to_num(x=np.abs(blazed_int_spec - broad_int_spec))

    # conducting the convolution with MKID resolution widths:
    convol_wave, convol_result = eng.convolve_mkid_response(wave=clipped_spectrum.waveset,
                                                            spectral_fluxden=broadened_spectrum,
                                                            oversampling=args.osamp,
                                                            n_sigma_mkid=args.nsig, simp=sim.simpconvol)

    # building the kernel to divide out:
    mkid_kernel = eng.build_mkid_kernel(n_sigma=args.nsig, sampling=spectro.sampling(args.osamp))
    x = eng.mkid_kernel_waves(n_points=len(mkid_kernel), n_sigma=args.nsig, oversampling=args.osamp)

    # integrating the kernel with each grid spacing:
    norms = np.sum(a=mkid_kernel) * (x[:, :, 1] - x[:, :, 0]) / sigma_frac  # since n sigma is not exactly 1

    # calculating the spacing for every pixel-order:
    dx = (convol_wave[1, :, :] - convol_wave[0, :, :]).to(u.nm)

    # returning the convolution spacing back in line with everything else:
    convol_normed = convol_result.to(u.ph / u.cm ** 2 / u.s) / norms[None, ...] * dx[None, ...].value

    # integrating the convolution to go to pixel-order array size:
    convol_int = np.sum(convol_normed, axis=0)

    # getting the relative noise as a result of convolution:
    convol_noise = np.nan_to_num(x=np.abs(blazed_int_spec - convol_int.value)) - broad_noise

    # putting convolved spectrum through MKID observation sequence:
    photons, observed, reduce_factor = detector.observe(convol_wave=convol_wave, convol_result=convol_normed,
                                                        minwave=sim.minwave, maxwave=sim.maxwave,
                                                        exptime=sim.exptime, area=sim.telearea)

    # separate photons by resid (pixel) and realign (no offset):
    idx = [np.where(photons[:observed].resID == resid_map[j]) for j in range(sim.npix)]
    photons_realign = [(photons[:observed].wavelength[idx[j]] / phase_offsets[j]).tolist() for j in range(sim.npix)]

    # use FSR to bin and order sort:
    fsr = spectro.fsr(order=spectro.orders).to(u.nm)
    hist_bins = np.empty((nord + 1, sim.npix))  # choosing rough histogram bins by using FSR of each pixel/wave
    hist_bins[0, :] = (lambda_pixel[-1, :] - fsr[-1] / 2).value
    hist_bins[1:, :] = [(lambda_pixel[i, :] + fsr[i] / 2).value for i in range(nord)[::-1]]
    hist_bins = wave_to_phase(waves=hist_bins, minwave=sim.minwave, maxwave=sim.maxwave)

    photons_binned = np.empty((nord, sim.npix))
    for j in range(sim.npix):
        photons_binned[:, j], _ = np.histogram(a=photons_realign[j], bins=hist_bins[:, j], density=False)

    # normalize to level of convolution since that's where it came from and calculate noise:
    photons_binned = (
            photons_binned * u.ph * reduce_factor[None, :] / (sim.exptime.to(u.s) * sim.telearea.to(u.cm ** 2))).to(
        u.ph / u.cm ** 2 / u.s).value
    observe_noise = np.nan_to_num(np.abs(blazed_int_spec - photons_binned[::-1])) - broad_noise - convol_noise

    # save noise to file:
    # np.savez(
    #     file=f'{args.output_dir}/{args.output_h5file}_noise.npz',
    #     original=blazed_int_spec,
    #     broad_noise=broad_noise,
    #     convol_noise=convol_noise,
    #     observe_noise=observe_noise
    # )
    # logging.info(f'\nSaved noise profiles to {args.output_dir}/{args.output_h5file}_noise.npz')

    # saving final photon list to h5 file, store linear phase conversion in header:
    h5_file = f'{args.output_dir}/{now.strftime("%y%m%d")}_{args.spectro_type}/{sim.type_spectra}.h5'
    buildfromarray(array=photons[:observed], user_h5file=h5_file)
    pt = Photontable(file_name=h5_file, mode='write')
    pt.update_header(key='sim_settings', value=sim)
    pt.update_header(key='phase_expression', value='0.6 * (freq_allwave - freq_minw) / (freq_maxw - freq_minw) - 0.8')
    pt.disablewrite()  # allows other scripts to open the table

    # at this point we are simulating the pipeline and have gone past the "wavecal" part. Next is >to spectrum.
    logging.info(msg=f'\nSaved photon table to {h5_file}.')
    logging.info(msg=f'\nTotal simulation time: {((time.perf_counter() - tic) / 60):.2f} min.')
    # ==================================================================================================================
    # SIMULATION ENDS
    # ==================================================================================================================

    # ==================================================================================================================
    # DEBUGGING PLOTS:
    # ==================================================================================================================
    warnings.filterwarnings(action="ignore")  # ignore tight_layout warnings

    # phase/pixel space plot to verify that phases are within proper values and orders are more-or-less visible
    bin_edges = np.linspace(-1, -0.1, 100)
    centers = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_array = np.zeros([sim.npix, len(bin_edges) - 1])
    for j in detector.pixel_indices:
        if photons_realign[j]:
            counts, edges = np.histogram(a=photons_realign[j], bins=bin_edges)
            hist_array[j, :] = np.array([float(x) for x in counts])
    plt.imshow(hist_array[:, ::-1].T, extent=[1, sim.npix, -1, -0.1], aspect='auto', norm=LogNorm())
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Photon Count')
    plt.title("Phase & Pixel Binning of Observed Photons")
    plt.xlabel("Pixel Index")
    plt.ylabel(r"Phase ($\times \pi /2$)")
    plt.tight_layout()
    plt.show()

    """Calculations for intermediate plotting"""
    masked_broad = [eng.optically_broaden(wave=masked_waves[i], flux=masked_blaze[i], axis=0) for i in range(nord)]

    """Plotting stuff only below"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    axes = axes.ravel()
    plt.suptitle(f"Intermediate plots for MKIDSpec {sim.type_spectra} simulation", fontweight='bold')

    # plotting bandpassed and blazed/broadened spectrum:
    quick_plot(ax=axes[0], x=[bandpass_spectrum.waveset.to(u.nm)],
               y=[bandpass_spectrum(bandpass_spectrum.waveset) / np.max(bandpass_spectrum(bandpass_spectrum.waveset))],
               labels=['Incident, R~50,000'], color='r', first=True, xlim=[400, 800])
    broad_max = np.max([np.max(masked_broad[i].value) for i in range(nord)])
    quick_plot(ax=axes[0], x=masked_waves,
               y=[masked_broad[i] / broad_max for i in range(nord)], color='g',
               labels=['Blazed+Broadened, R~3500'] + ['_nolegend_' for o in spectro.orders[:-1]],
               title="Comparison of Input (Instrument-incident) and Blazed/Opt.-Broadened Spectra (not relative)",
               ylabel=r"Normalized Flux Density")

    # plotting comparison between flux-integrated spectrum and integrated/convolved spectrum, must be same,
    # also plotting final counts FSR-binned
    quick_plot(ax=axes[1], x=lambda_pixel, y=photons_binned[::-1], color='k', linewidth=1, linestyle='--',
               labels=['Obs. Count to Flux'] + ['_nolegend_' for o in spectro.orders[:-1]], first=True, xlim=[400, 800])
    quick_plot(ax=axes[1], x=lambda_pixel, y=convol_int, color='red', linewidth=1.5, alpha=0.4,
               labels=["Post-Convol"] + ['_nolegend_' for o in range(nord - 1)], xlim=[400, 800])
    quick_plot(ax=axes[1], x=lambda_pixel, y=blazed_int_spec, color='b', ylabel=r"Flux (phot $cm^{-2} s^{-1})$",
               labels=["Input-integrated"] + ['_nolegend_' for o in range(nord - 1)], alpha=0.4, linewidth=1.5,
               xlabel='Wavelength (nm)', title=r"Comparison of Input (integrated to pixel space), "
                                               r"Post-Convolution (integrated), and Observed Photon Count (FSR-binned)")
    fig.tight_layout()
    plot_file = f'{args.output_dir}/{now.strftime("%y%m%d")}_{args.spectro_type}/{sim.type_spectra}.pdf'
    fig.savefig(plot_file)
    logging.info(msg=f'\nSaved intermediate plots to {plot_file}.')
    plt.show()

    # plot noise from various sources only
    fig2, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.suptitle(f"Noise plot for MKIDSpec {sim.type_spectra} simulation", fontweight='bold')

    quick_plot(ax=ax, x=lambda_pixel, y=broad_noise + convol_noise + observe_noise, color='b', first=True,
               labels=["Opt.-Broadening Noise"] + ['_nolegend_' for o in spectro.orders[:-1]])
    for i in range(nord):
        ax.fill_between(lambda_pixel[i].value, 0, (broad_noise + convol_noise + observe_noise)[i], color='b')
    quick_plot(ax=ax, x=lambda_pixel, y=convol_noise, color='g',
               labels=["Convol. Noise"] + ['_nolegend_' for o in spectro.orders[:-1]])
    for i in range(nord):
        ax.fill_between(lambda_pixel[i].value, 0, (convol_noise + observe_noise)[i], color='g')
    quick_plot(ax=ax, x=lambda_pixel, y=observe_noise, color='pink',
               labels=["Observing Noise"] + ['_nolegend_' for o in spectro.orders[:-1]])
    for i in range(nord):
        ax.fill_between(lambda_pixel[i].value, 0, observe_noise[i], color='pink')
    quick_plot(ax=ax, x=lambda_pixel, y=blazed_int_spec, color='k', linestyle='--', linewidth=0.5,
               labels=["Original, Blazed"] + ['_nolegend_' for o in spectro.orders[:-1]],
               xlabel='Wavelength (nm)', ylabel=r"Flux (phot $cm^{-2} s^{-1})$")
    fig2.tight_layout()
    plot_file = f'{args.output_dir}/{now.strftime("%y%m%d")}_{args.spectro_type}/{sim.type_spectra}_noiseplots.pdf'
    fig2.savefig(plot_file)
    logging.info(msg=f'\nSaved noise plots to {plot_file}.')
    plt.show()
