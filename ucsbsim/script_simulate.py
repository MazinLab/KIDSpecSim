import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate as interp
import astropy.units as u
import time
from datetime import datetime as dt
import logging
import argparse
from specutils import Spectrum1D
from synphot import SpectralElement

from spectra import get_spectrum, apply_bandpass, AtmosphericTransmission, FilterTransmission, TelescopeTransmission
from spectrograph import GratingSetup, SpectrographSetup
from detector import MKIDDetector
from engine import Engine
from plotting import quick_plot, simulation_plot

# TODO this will need work as the pipeline will probably default to MEC HDF headers
from mkidpipeline.steps.buildhdf import buildfromarray

"""
Simulation of the MKID spectrometer. The steps are:
-A source spectrum is obtained.
-It is filtered through atmospheric, telescopic, and filter bandpasses to emulate the spectrum's journey from the
 star to the spectrograph.
-It is multiplied by the blaze efficiency of the grating, which determines how much of the flux will be incident on
 each pixel of the detector.
-It is broadened according to the optical LSF, the limit to the resolution of the optics used.
-It is convolved with the MKID resolution width to simulate the wavelength detection of the MKIDs as a function of
 wavelength. This puts the spectrum into pixel space (flux), whereas before it was in wavelength space (flux density).
-The photons are drawn and subsequently observed by the MKIDs according to Poisson statistics, random draws, and MKID-
 specific properties such as dead time and minimum trigger energy.
-The photon table is saved to an h5 file.

Notes:
-When simulating a calibration spectrum, there will be no multiplication with atmospheric or telescopic bandpasses
 to set the source in the laboratory setting, i.e. not coming from on-sky.
-The intermediate plots shown will therefore be more interesting when simulating an on-sky source (i.e. Phoenix model).
"""

if __name__ == '__main__':
    tic = time.time()  # recording start time for script
    u.photlam = u.photon / u.s / u.cm ** 2 / u.AA  # new unit name, photon flux per wavelength

    # ==================================================================================================================
    # CONSTANTS
    # ==================================================================================================================
    N_SIGMA_MKID = 3
    THREESIG = 0.9973
    OSAMP = 10

    # ==================================================================================================================
    # PARSE ARGUMENTS
    # ==================================================================================================================
    arg_desc = '''
               Simulate an observed spectrum from the MKID Spectrometer.
               ---------------------------------------------------------
               This program loads a synphot spectrum and filters, blazes, convolves, and converts to photon table.
               '''
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=arg_desc)

    # required simulation args:
    parser.add_argument('output_file',
                        metavar='OUTPUT_FILE_NAME',
                        help='Directory/name of the output photon table.')
    parser.add_argument('R0s_file',
                        metavar='R0S_FILE_NAME',
                        help="Directory/name of the R0s file, will be created if it doesn't exist.")
    parser.add_argument('type_of_spectra',
                        metavar='TYPE_OF_SPECTRA',
                        help='The type of spectrum to be simulated: can be "blackbody", "phoenix", or "delta".')

    # optional simulation args:
    parser.add_argument('--pixel_lim',
                        metavar='PIXEL_LIMIT',
                        default=5000,
                        help='Limit to the # of photons per pixel.')
    parser.add_argument('--exptime',
                        metavar='EXPOSURE_TIME',
                        default=500*u.s,
                        help='The exposure time of the observation.')
    parser.add_argument('--telescope_area',
                        metavar='TELESCOPE_AREA',
                        default=np.pi * (4 * u.cm) ** 2,
                        help='The telescope area of the observation.')
    parser.add_argument('--temp',
                        metavar='TEMP',
                        default=4300,
                        help='The temperature of the spectrum in K if "blackbody" or "phoenix".')
    parser.add_argument('--full_convol',
                        action='store_true',
                        default=True,
                        help='Indicates that a full convolution should be conducted.')
    parser.add_argument('--atmo_bandpass',
                        action='store_true',
                        default=False,
                        help='Indicates that the spectrum should be multiplied with the atmospheric attenuation.')
    parser.add_argument('--filter_bandpass',
                        action='store_true',
                        default=False,
                        help='Indicates that the spectrum should be multiplied with the filter bandpass.')
    parser.add_argument('--minwave',
                        metavar='MINIMUM_WAVELENGTH',
                        default=400*u.nm,
                        help='The minimum wavelength of the spectrometer.')
    parser.add_argument('--maxwave',
                        metavar='MAXIMUM_WAVELENGTH',
                        default=800*u.nm,
                        help='The maximum wavelength of the spectrometer.')
    parser.add_argument('--tele_bandpass',
                        action='store_true',
                        default=False,
                        help='Indicates that the spectrum should be multiplied with the telescopic attenuation.')
    parser.add_argument('--reflectivity',
                        metavar='TELESCOPE_REFLECTIVITY',
                        default=0.9,
                        help='Factor by which to attenuate spectrum due to telescope reflectivity.')

    # optional detector args:
    parser.add_argument('--npix',
                        metavar='NUMBER_OF_PIXELS',
                        default=2048,
                        help='The number of pixels in the MKID detector.')
    parser.add_argument('--pixel_size',
                        metavar='PIXEL_SIZE',
                        default=20*u.micron,
                        help='The length of the MKID pixel in the dispersion direction.')
    parser.add_argument('--design_R0',
                        metavar='DESIGN_R0',
                        default=15,
                        help='The spectral resolution at the longest wavelength of all orders.')
    parser.add_argument('--l0',
                        metavar='LAMBDA_0',
                        default=800*u.nm,
                        help='The longest wavelength of any order.')

    # optional grating args:
    parser.add_argument('--m0',
                        metavar='INITIAL_ORDER',
                        default=5,
                        help='The initial order, at the longer wavelength end.')
    parser.add_argument('--m_max',
                        metavar='FINAL_ORDER',
                        default=9,
                        help='The final order, at the shorter wavelength end.')
    parser.add_argument('--focal_length',
                        metavar='FOCAL_LENGTH',
                        default=350*u.mm,
                        help='The focal length of the grating.')
    parser.add_argument('--littrow',
                        action='store_true',
                        default=False,
                        help='Indicates whether to configure grating on Littrow (incident=reflected angle).')

    # optional spectrograph args:
    parser.add_argument('--pixels_per_res_elem',
                        metavar='PIXELS_PER_RESOLUTION_ELEMENT',
                        default=2.5,
                        help='Number of pixels per spectral resolution element for the spectrograph.')

    # set arguments as variables
    args = parser.parse_args()
    h5_file = args.output_file
    R0s_file = args.R0s_file
    type_of_spectra = args.type_of_spectra
    pixel_lim = args.pixel_lim
    exptime = args.exptime
    area = args.telescope_area
    temp = args.temp
    full_convol = args.full_convol
    atmo_bandpass = args.atmo_bandpass
    filter_bandpass = args.filter_bandpass
    minwave = args.minwave
    maxwave = args.maxwave
    tele_bandpass = args.tele_bandpass
    reflectivity = args.reflectivity
    npix = args.npix
    pixel_size = args.pixel_size
    R0 = args.design_R0
    l0 = args.l0
    m0 = args.m0
    m_max = args.m_max
    focal_length = args.focal_length
    littrow = args.littrow
    pixels_per_res_elem = args.pixels_per_res_elem

    now = dt.now()
    logging.basicConfig(filename=f'output_files/{type_of_spectra}/simulate_{now.strftime("%Y%m%d_%H%M%S")}.log',
                        format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info(f"The simulation of a {type_of_spectra} spectrum's journey through the spectrometer is recorded."
                 f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    # ==================================================================================================================
    # INSTANTIATE SPECTROGRAPH & DETECTOR
    # ==================================================================================================================
    try:
        R0s = np.loadtxt(R0s_file, delimiter=',')
        logging.info(f'\nThe individual R0s were imported from {R0s_file}.')
    except IOError as e:
        logging.info(e)
        R0s = np.random.uniform(.85, 1.15, size=npix) * R0
        np.savetxt(R0s_file, R0s, delimiter=',')
        logging.info(f'\nThe individual R0s were generated randomly from the design R0 and saved to {R0s_file}.')

    bandpasses = []
    if atmo_bandpass:
        bandpasses.append(AtmosphericTransmission())
    if filter_bandpass:
        bandpasses.append(FilterTransmission(minwave, maxwave))
    if tele_bandpass:
        bandpasses.append(TelescopeTransmission(reflectivity=reflectivity))
    if not bandpasses:  # for no bandpasses, just convert to finer grid spacing
        w = np.linspace(300, 1000, 10000) * u.nm
        t = np.ones(10000) * u.dimensionless_unscaled
        ones = Spectrum1D(spectral_axis=w, flux=t)
        bandpasses.append(SpectralElement.from_spectrum1d(ones))

    resid_map = np.arange(npix, dtype=int) * 10 + 100  # TODO replace once known

    detector = MKIDDetector(npix, pixel_size, R0, l0, R0s, resid_map)
    grating = GratingSetup(l0, m0, m_max, pixel_size, npix, focal_length, littrow)
    spectro = SpectrographSetup(grating, detector, pixels_per_res_elem)
    eng = Engine(spectro)

    # ==================================================================================================================
    # SIMULATION STARTS:
    # ==================================================================================================================
    # setting up engine and spectrograph, obtaining various needed properties:
    nord = spectro.nord
    lambda_pixel = spectro.pixel_center_wavelengths().to(u.nm)

    # obtaining spectra and passing through several transformations:
    spectrum = get_spectrum(type_of_spectra, teff=temp, min=minwave, max=maxwave)
    bandpass_spectrum = apply_bandpass(spectrum, bandpass=bandpasses)
    blazed_spectrum, masked_waves, masked_blaze = eng.blaze(bandpass_spectrum.waveset, bandpass_spectrum)
    broadened_spectrum = eng.optically_broaden(bandpass_spectrum.waveset, blazed_spectrum)
    masked_broad = [eng.optically_broaden(masked_waves[i], masked_blaze[i], axis=0) for i in range(nord)]

    # conducting the convolution with MKID resolution widths:
    convol_wave, convol_result = eng.convolve_mkid_response(bandpass_spectrum.waveset, broadened_spectrum, OSAMP,
                                                            N_SIGMA_MKID, full=full_convol)

    # putting convolved spectrum through MKID observation sequence:
    photons, observed = detector.observe(convol_wave, convol_result, limit_to=pixel_lim, exptime=exptime, area=area)

    # saving final photon list to h5 file:
    h5_file = f'{args.output_dir}/R0{args.design_R0}_{args.pixel_lim}.h5'
    buildfromarray(photons[:observed], user_h5file=h5_file)
    pt = Photontable(h5_file, mode='write')
    pt.update_header('sim_parameters', vars(args))
    pt.disablewrite()  # allows other scripts to open the table
    # at this point we are simulating the pipeline and have gone past the "wavecal" part. Next is >to spectrum.
    logging.info(f'\nSaved photon table of {type_of_spectra} spectrum to {h5_file}')
    logging.info(f'\nTotal simulation time: {((time.time() - tic) / 60):.2f} min.')
    # ==================================================================================================================
    # SIMULATION ENDS
    # ==================================================================================================================

    # ==================================================================================================================
    # DEBUGGING PLOTS:
    # ==================================================================================================================
    warnings.filterwarnings("ignore")  # ignore tight_layout warnings

    # integrating the flux density spectrum to go to pixel space
    mkid_kernel = eng.build_mkid_kernel(N_SIGMA_MKID, eng.spectrograph.sampling(OSAMP))
    pix_leftedge = spectro.pixel_center_wavelengths(edge='left').to(u.nm).value
    direct_flux_calc = [eng.lambda_to_pixel_space(masked_waves[i], masked_broad[i],
                                                  pix_leftedge[i]) for i in range(nord)]

    # dividing convolution by kernel normalization and integrating
    x = eng.mkid_kernel_waves(len(mkid_kernel), n_sigma=N_SIGMA_MKID, oversampling=OSAMP)
    norms = np.sum(mkid_kernel) * (x[:, :, 1] - x[:, :, 0]) / THREESIG  # since 3 sigma is not exactly 1
    convol_for_plot = (convol_result / (norms[None, ...] * u.nm)).to(u.photlam)
    dx = (convol_wave[1, :, :] - convol_wave[0, :, :]).to(u.nm)
    convol_summed = (np.sum(convol_for_plot.value, axis=0) * u.photlam * dx)

    # use FSR to bin
    fsr = spectro.fsr(eng.spectrograph.orders).to(u.nm)
    hist_bins = np.empty((nord + 1, npix))  # choosing rough histogram bins by using FSR of each pixel/wave
    hist_bins[0, :] = (lambda_pixel[-1, :] - fsr[-1] / 2).value
    hist_bins[1:, :] = [(lambda_pixel[i, :] + fsr[i] / 2).value for i in range(nord)[::-1]]

    photons_binned = np.empty((nord, npix))
    for j in range(npix):  # sorting photons by resID (i.e. pixel)
        ph = photons[:observed].wavelength[np.where(photons.resID == resid_map[j])].tolist()
        photons_binned[:, j], _ = np.histogram(ph, bins=hist_bins[:, j], density=False)

    # TODO have the full plotting be a function called makeplot or some shit
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.5, 11), dpi=300)
    axes = axes.ravel()
    plt.suptitle(f"Intermediate plots for MKID spectrometer simulation ({pixel_lim} Photon Limit)",
                 fontweight='bold')

    # plotting input, bandpassed, blazed, and broadened spectrum:
    quick_plot(axes[0], [spectrum.waveset.to(u.nm), bandpass_spectrum.waveset.to(u.nm)],
               [spectrum(spectrum.waveset), bandpass_spectrum(bandpass_spectrum.waveset)],
               labels=['Original', 'Filtered'], first=True)
    quick_plot(axes[0], masked_waves, masked_blaze, labels=[f'Blazed O{o}' for o in spectro.orders])
    quick_plot(axes[0], masked_waves, masked_broad, color='k',
               labels=['Broadened'] + ['_nolegend_' for o in spectro.orders[:-1]],
               title="Original through Optically-Broadened",
               ylabel=r"Flux Density (phot $cm^{-2} s^{-1} \AA^{-1})$")
    axes[0].set_xlim([minwave.value - 25, l0.value + 25])

    # plotting comparison between lambda-to-pixel spectrum and integrated convolution spectrum, must be same:
    quick_plot(axes[1], lambda_pixel, convol_summed, color='red',
               labels=["Conv+Int"] + ['_nolegend_' for o in range(nord - 1)], first=True)
    quick_plot(axes[1], lambda_pixel, direct_flux_calc, color='k',
               labels=[r"$\lambda$2Pix Int."] + ['_nolegend_' for o in range(nord - 1)],
               title=r"Convolved+Integrated & $\lambda$-to-Pixel Integrated",
               ylabel=r"Flux (phot $cm^{-2} s^{-1})$")

    # plotting comparison between final counts and convolved-integrated spectrum:
    quick_plot(axes[2], lambda_pixel, photons_binned[::-1], ylabel="Photon Count", color='k',
               labels=['Observed+FSR-Binned'] + ['_nolegend_' for o in spectro.orders[:-1]], first=True)
    twin = axes[2].twinx()
    quick_plot(twin, lambda_pixel, convol_summed, ylabel=r"Flux (phot $cm^{-2} s^{-1})$", color='red',
               labels=['Conv.+Int.'] + ['_nolegend_' for o in spectro.orders[:-1]],
               title="Convolved+Integrated vs. Observed Photons", xlabel="Wavelength (nm)", twin='red', alpha=0.5)
    fig.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.6, left=0.1)
    plot_file = f'output_files/{type_of_spectra}/intplots_R0{R0}.pdf'
    fig.savefig(plot_file)
    logging.info(f'\nSaved intermediate plots to {plot_file}.')
    plt.show()
