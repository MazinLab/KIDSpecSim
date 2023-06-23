import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
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

from spectra import get_spectrum, apply_bandpass, AtmosphericTransmission, FilterTransmission, TelescopeTransmission,\
    clip_spectrum
from spectrograph import GratingSetup, SpectrographSetup
from detector import MKIDDetector
from engine import Engine
from plotting import quick_plot
from mkidpipeline.photontable import Photontable
from simsettings import SpecSimSettings

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
    SIGMA_FRAC = norm.cdf(N_SIGMA_MKID)
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
    parser.add_argument('output_dir',
                        metavar='OUTPUT_DIRECTORY',
                        help='Directory for the output files. (str)')
    parser.add_argument('R0s_file',
                        metavar='R0S_FILE_NAME',
                        help="Directory/filename of the R0s file, will be created if it doesn't exist. (str)")
    parser.add_argument('type_spectra',
                        metavar='TYPE_SPECTRA',
                        help='The type of spectrum to be simulated: can be "blackbody", "phoenix", or "delta". (str)')

    # optional simulation args:
    parser.add_argument('-pl', '--pixellim',
                        metavar='PIXEL_LIMIT',
                        default=5000,
                        help='Limit to the # of photons per pixel. (int)')
    parser.add_argument('-et', '--exptime',
                        metavar='EXPOSURE_TIME_S',
                        default=500,
                        help='The exposure time of the observation in seconds. (float)')
    parser.add_argument('-ta', '--telearea',
                        metavar='TELESCOPE_AREA_CM2',
                        default=np.pi * 4 ** 2,
                        help='The telescope area of the observation in cm2. (float)')
    parser.add_argument('-T', '--temp',
                        metavar='TEMP_K',
                        default=4300,
                        help='Temperature of the spectrum in K if "type_spectra" is "blackbody" or "phoenix". (float)')
    parser.add_argument('-sc', '--simpconvol',
                        action='store_true',
                        default=False,
                        help='If passed, indicates that a simplified convolution should be conducted.')
    parser.add_argument('-ab', '--atmobandpass',
                        action='store_true',
                        default=False,
                        help='If passed, indicates the spectrum will be multiplied with atmospheric attenuation.')
    parser.add_argument('-fb', '--filterbandpass',
                        action='store_true',
                        default=False,
                        help='If passed, indicates the spectrum will be multiplied with the filter bandpass.')
    parser.add_argument('-minw', '--minwave',
                        metavar='MINIMUM_WAVELENGTH_NM',
                        default=400,
                        help='The minimum wavelength of the spectrometer in nm. (float)')
    parser.add_argument('-maxw', '--maxwave',
                        metavar='MAXIMUM_WAVELENGTH_NM',
                        default=800,
                        help='The maximum wavelength of the spectrometer in nm. (float)')
    parser.add_argument('-tb', '--telebandpass',
                        action='store_true',
                        default=False,
                        help='If passed, indicates the spectrum will be multiplied with telescopic attenuation.')
    parser.add_argument('-ref', '--reflect',
                        metavar='TELESCOPE_REFLECTIVITY',
                        default=0.9,
                        help='Factor to attenuate spectrum due to telescope reflectivity, between 0 and 1. (float)')

    # optional detector args:
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
                        help='The spectral resolution at the longest wavelength of any order in use. (float)')
    parser.add_argument('-l0', '--l0',
                        metavar='LAMBDA_0_NM',
                        default=800,
                        help='The longest wavelength of any order in use in nm. (float)')

    # optional grating args:
    parser.add_argument('-m0', '--m0',
                        metavar='INITIAL_ORDER',
                        default=5,
                        help='The initial order, at the longer wavelength end. (int)')
    parser.add_argument('-mm', '--m_max',
                        metavar='FINAL_ORDER',
                        default=9,
                        help='The final order, at the shorter wavelength end. (int)')
    parser.add_argument('-fl', '--focallength',
                        metavar='FOCAL_LENGTH_MM',
                        default=350,
                        help='The focal length of the detector in mm. (float)')
    parser.add_argument('-nl', '--nolittrow',
                        action='store_true',
                        default=False,
                        help='If passed, indicates NOT to configure grating on Littrow (incident=reflected angle).')

    # optional spectrograph args:
    parser.add_argument('-ppre', '--pixels_per_res_elem',
                        metavar='PIXELS_PER_RESOLUTION_ELEMENT',
                        default=2.5,
                        help='Number of pixels per spectral resolution element for the spectrograph. (float)')

    # get arguments & simulation settings
    args = parser.parse_args()
    sim = SpecSimSettings(args.R0s_file, args.designR0, args.simpconvol, args.nolittrow, args.l0, args.m0, args.m_max,
                          args.minwave, args.maxwave, args.npix, args.pixellim, args.exptime, args.telearea,
                          args.pixelsize, args.focallength, args.pixels_per_res_elem, args.temp, args.type_spectra)

    # ==================================================================================================================
    # START LOGGING TO FILE
    # ==================================================================================================================
    now = dt.now()
    logging.basicConfig(filename=f'{args.output_dir}/simulate_{now.strftime("%Y%m%d_%H%M%S")}.log',
                        format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info(f"The simulation of a {sim.type_spectra} spectrum's journey through the spectrometer is recorded."
                 f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    # ==================================================================================================================
    # INSTANTIATE SPECTROGRAPH & DETECTOR
    # ==================================================================================================================
    try:
        R0s = np.loadtxt(sim.R0s_file, delimiter=',')
        logging.info(f'\nThe individual R0s were imported from {sim.R0s_file}.')
    except IOError as e:
        logging.info(e)
        R0s = np.random.uniform(.85, 1.15, size=sim.npix) * sim.designR0
        np.savetxt(sim.R0s_file, R0s, delimiter=',')
        logging.info(f'\nThe individual R0s were generated randomly from the design R0 and saved to {sim.R0s_file}.')

    bandpasses = []
    if args.atmobandpass:
        bandpasses.append(AtmosphericTransmission())
    if args.filterbandpass:
        bandpasses.append(FilterTransmission(sim.minwave, sim.maxwave))
    if args.telebandpass:
        bandpasses.append(TelescopeTransmission(reflectivity=args.reflect))
    if not bandpasses:  # for no bandpasses, just convert to finer grid spacing
        w = np.linspace(300, 1000, 10000) * u.nm
        t = np.ones(10000) * u.dimensionless_unscaled
        ones = Spectrum1D(spectral_axis=w, flux=t)
        bandpasses.append(SpectralElement.from_spectrum1d(ones))

    resid_map = np.arange(sim.npix, dtype=int) * 10 + 100  # TODO replace once known

    detector = MKIDDetector(sim.npix, sim.pixelsize, sim.designR0, sim.l0, R0s, resid_map)
    grating = GratingSetup(sim.l0, sim.m0, sim.m_max, sim.pixelsize, sim.npix, sim.focallength, sim.nolittrow)
    spectro = SpectrographSetup(grating, detector, sim.pixels_per_res_elem)
    eng = Engine(spectro)

    # ==================================================================================================================
    # SIMULATION STARTS:
    # ==================================================================================================================
    # setting up engine and spectrograph, obtaining various needed properties:
    nord = spectro.nord
    lambda_pixel = spectro.pixel_wavelengths().to(u.nm)

    # obtaining spectra and passing through several transformations:
    spectrum = get_spectrum(sim.type_spectra, teff=sim.temp, min=sim.minwave, max=sim.maxwave)
    bandpass_spectrum = apply_bandpass(spectrum, bandpass=bandpasses)
    clipped_spectrum = clip_spectrum(bandpass_spectrum, clip_range=(sim.minwave, sim.maxwave))
    blazed_spectrum, masked_waves, masked_blaze = eng.blaze(clipped_spectrum.waveset, clipped_spectrum)
    broadened_spectrum = eng.optically_broaden(clipped_spectrum.waveset, blazed_spectrum)
    masked_broad = [eng.optically_broaden(masked_waves[i], masked_blaze[i], axis=0) for i in range(nord)]

    # conducting the convolution with MKID resolution widths:
    convol_wave, convol_result = eng.convolve_mkid_response(clipped_spectrum.waveset, broadened_spectrum, OSAMP,
                                                            N_SIGMA_MKID, simp=sim.simpconvol)

    # putting convolved spectrum through MKID observation sequence:
    photons, observed = detector.observe(convol_wave, convol_result, limit_to=sim.pixellim, exptime=sim.exptime,
                                         area=sim.telearea)

    # saving final photon list to h5 file:
    h5_file = f'{args.output_dir}/R0{sim.designR0}_{sim.pixellim}.h5'
    buildfromarray(photons[:observed], user_h5file=h5_file)
    pt = Photontable(h5_file, mode='write')
    pt.update_header('sim_settings', sim)
    pt.disablewrite()  # allows other scripts to open the table

    # at this point we are simulating the pipeline and have gone past the "wavecal" part. Next is >to spectrum.
    logging.info(f'\nSaved photon table of {sim.type_spectra} spectrum to {h5_file}')
    logging.info(f'\nTotal simulation time: {((time.time() - tic) / 60):.2f} min.')
    # ==================================================================================================================
    # SIMULATION ENDS
    # ==================================================================================================================


    # ==================================================================================================================
    # DEBUGGING PLOTS:
    # ==================================================================================================================
    warnings.filterwarnings("ignore")  # ignore tight_layout warnings

    """Calculations for plotting"""
    # integrating the flux density spectrum to go to pixel space
    mkid_kernel = eng.build_mkid_kernel(N_SIGMA_MKID, spectro.sampling(OSAMP))
    pix_leftedge = spectro.pixel_wavelengths(edge='left').to(u.nm).value
    direct_flux_calc = [eng.lambda_to_pixel_space(masked_waves[i], masked_broad[i],
                                                  pix_leftedge[i]) for i in range(nord)]

    # dividing convolution by kernel normalization and integrating
    x = eng.mkid_kernel_waves(len(mkid_kernel), n_sigma=N_SIGMA_MKID, oversampling=OSAMP)
    norms = np.sum(mkid_kernel) * (x[:, :, 1] - x[:, :, 0]) / SIGMA_FRAC  # since 3 sigma is not exactly 1
    convol_for_plot = (convol_result / (norms[None, ...] * u.nm)).to(u.photlam)
    dx = (convol_wave[1, :, :] - convol_wave[0, :, :]).to(u.nm)
    convol_summed = (np.sum(convol_for_plot.value, axis=0) * u.photlam * dx)

    # use FSR to bin
    fsr = spectro.fsr(spectro.orders).to(u.nm)
    hist_bins = np.empty((nord + 1, sim.npix))  # choosing rough histogram bins by using FSR of each pixel/wave
    hist_bins[0, :] = (lambda_pixel[-1, :] - fsr[-1] / 2).value
    hist_bins[1:, :] = [(lambda_pixel[i, :] + fsr[i] / 2).value for i in range(nord)[::-1]]

    photons_binned = np.empty((nord, sim.npix))
    for j in range(sim.npix):  # sorting photons by resID (i.e. pixel)
        ph = photons[:observed].wavelength[np.where(photons.resID == resid_map[j])].tolist()
        photons_binned[:, j], _ = np.histogram(ph, bins=hist_bins[:, j], density=False)

    """Plotting stuff only below"""
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.5, 11), dpi=300)
    axes = axes.ravel()
    plt.suptitle(f"Intermediate plots for MKID spectrometer simulation ({sim.pixellim} Photon Limit)",
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
    axes[0].set_xlim([sim.minwave.value - 25, sim.l0.value + 25])

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
    plot_file = f'{args.output_dir}/intplots_R0{sim.designR0}.pdf'
    fig.savefig(plot_file)
    logging.info(f'\nSaved intermediate plots to {plot_file}.')
    plt.show()
