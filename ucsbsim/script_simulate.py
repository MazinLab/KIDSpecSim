import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate as interp
import astropy.units as u
import time
from datetime import datetime as dt
import logging

from spectra import get_spectrum, apply_bandpass
from spectrograph import GratingSetup, SpectrographSetup
from detector import MKIDDetector
from engine import Engine, quick_plot

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

    # ========================================================================
    # Constants which may or may not be needed in the future:
    # ========================================================================
    N_SIGMA_MKID = 3
    THREESIG = 0.9973
    OSAMP = 10
    MINWAVE = 400 * u.nm

    # ========================================================================
    # Assign spectrograph properties below if different from defaults. Check the module spectrograph for syntax.
    # ========================================================================
    # R0 = 8, etc.

    # ========================================================================
    # Simulation properties:
    # ========================================================================
    calibration = False
    full_convol = True
    type_of_spectra, temp = 'phoenix', 4300  # in K
    pixel_lim = 1000

    # ========================================================================
    # Simulation begins:
    # ========================================================================
    now = dt.now()
    logging.basicConfig(filename=f'output_files/{type_of_spectra}/logging/simulate_{now.strftime("%Y%m%d_%H%M%S")}.log',
                        format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.info(f"The simulation of a {type_of_spectra} spectrum's journey through the spectrometer is recorded."
                 f"\nThe date and time are: {now.strftime('%Y-%m-%d %H:%M:%S')}.")

    # setting up engine and spectrograph, obtaining various needed properties:
    eng = Engine(SpectrographSetup(GratingSetup(), MKIDDetector()))
    nord = eng.spectrograph.nord
    npix = eng.spectrograph.detector.n_pixels
    resid_map = np.arange(npix, dtype=int) * 10 + 100  # TODO replace once known
    lambda_pixel = eng.spectrograph.pixel_center_wavelengths().to(u.nm)

    # obtaining spectra and passing through several transformations:
    u.photlam = u.photon / u.s / u.cm ** 2 / u.AA  # photon flux per wavelength
    spectrum = get_spectrum(type_of_spectra, teff=temp)  # TODO change kwargs to min/max if delta
    bandpass_spectrum = apply_bandpass(spectrum, cal=calibration, min=MINWAVE, max=eng.spectrograph.l0)
    blazed_spectrum, masked_waves, masked_blaze = eng.blaze(bandpass_spectrum.waveset, bandpass_spectrum)
    broadened_spectrum = eng.optically_broaden(bandpass_spectrum.waveset, blazed_spectrum)
    masked_broad = [eng.optically_broaden(masked_waves[i], masked_blaze[i], axis=0) for i in range(nord)]

    # conducting the convolution with MKID resolution widths:
    convol_wave, convol = eng.convolve_mkid_response(bandpass_spectrum.waveset, broadened_spectrum, OSAMP,
                                                     N_SIGMA_MKID, full=full_convol)

    # putting convolved spectrum through MKID observation sequence:
    photons, observed = eng.spectrograph.detector.observe(convol_wave, convol, limit_to=pixel_lim)

    # saving final photon list to h5 file:
    # TODO this will need work as the pipeline will probably default to MEC HDF headers
    from mkidpipeline.steps.buildhdf import buildfromarray

    h5_file = f'output_files/{type_of_spectra}/table_R0{eng.spectrograph.detector.design_R0}.h5'
    buildfromarray(photons[:observed], user_h5file=h5_file)
    # at this point we are simulating the pipeline and have gone past the "wavecal" part. Next is >to spectrum.
    logging.info(f'\nSaved photon table of {type_of_spectra} spectrum to {h5_file}')
    logging.info(f'\nTotal simulation time: {((time.time() - tic) / 60):.2f} min.')
    # ========================================================================
    # End of simulation
    # ========================================================================

    # ========================================================================
    # Plots for debugging and reasoning through:
    # ========================================================================
    import warnings

    warnings.filterwarnings("ignore")
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.5, 11), dpi=300)
    axes = axes.ravel()
    plt.suptitle(f"Intermediate plots for MKID spectrometer simulation ({pixel_lim} Photon Limit)",
                 fontweight='bold')

    # plotting input, bandpassed, blazed, and broadened spectrum:
    quick_plot(axes[0], [spectrum.waveset.to(u.nm), bandpass_spectrum.waveset.to(u.nm)],
               [spectrum(spectrum.waveset), bandpass_spectrum(bandpass_spectrum.waveset)],
               labels=['Original', 'Filtered'], first=True)
    quick_plot(axes[0], masked_waves, masked_blaze, labels=[f'Blazed O{o}' for o in eng.spectrograph.orders])
    quick_plot(axes[0], masked_waves, masked_broad, color='k',
               labels=['Broadened'] + ['_nolegend_' for o in eng.spectrograph.orders[:-1]],
               title="Original through Optically-Broadened",
               ylabel=r"Flux Density (phot $cm^{-2} s^{-1} \AA^{-1})$")
    axes[0].set_xlim([MINWAVE.value - 25, eng.spectrograph.l0.value + 25])

    # plotting comparison between lambda-to-pixel spectrum and integrated convolution spectrum, must be same:
    # integrating the flux density spectrum to go to pixel space
    mkid_kernel = eng.build_mkid_kernel(N_SIGMA_MKID, eng.spectrograph.sampling(OSAMP))
    pix_leftedge = eng.spectrograph.pixel_center_wavelengths(edge='left').to(u.nm).value
    direct_flux_calc = [eng.lambda_to_pixel_space(masked_waves[i], masked_broad[i],
                                                  pix_leftedge[i]) for i in range(nord)]

    # dividing convolution by kernel normalization and integrating
    x = eng.mkid_kernel_waves(len(mkid_kernel), n_sigma=N_SIGMA_MKID, oversampling=OSAMP)
    norms = np.sum(mkid_kernel) * (x[:, :, 1] - x[:, :, 0]) / THREESIG  # since 3 sigma is not exactly 1
    convol_for_plot = (convol / (norms[None, ...] * u.nm)).to(u.photlam)
    dx = (convol_wave[1, :, :] - convol_wave[0, :, :]).to(u.nm)
    convol_summed = (np.sum(convol_for_plot.value, axis=0) * u.photlam * dx)

    quick_plot(axes[1], lambda_pixel, convol_summed, color='red',
               labels=["Conv+Int"] + ['_nolegend_' for o in range(nord - 1)], first=True)
    quick_plot(axes[1], lambda_pixel, direct_flux_calc, color='k',
               labels=[r"$\lambda$2Pix Int."] + ['_nolegend_' for o in range(nord - 1)],
               title=r"Convolved+Integrated & $\lambda$-to-Pixel Integrated",
               ylabel=r"Flux (phot $cm^{-2} s^{-1})$")

    # plotting comparison between final counts and convolved-integrated spectrum:
    fsr = eng.spectrograph.fsr(eng.spectrograph.orders).to(u.nm)
    hist_bins = np.empty((nord + 1, npix))  # choosing rough histogram bins by using FSR of each pixel/wave
    hist_bins[0, :] = (lambda_pixel[-1, :] - fsr[-1] / 2).value
    hist_bins[1:, :] = [(lambda_pixel[i, :] + fsr[i] / 2).value for i in range(nord)[::-1]]

    photons_binned = np.empty((nord, npix))
    for j in range(npix):  # sorting photons by resID (i.e. pixel)
        ph = photons[:observed].wavelength[np.where(photons.resID == resid_map[j])].tolist()
        photons_binned[:, j], _ = np.histogram(ph, bins=hist_bins[:, j], density=False)

    quick_plot(axes[2], lambda_pixel, photons_binned[::-1], ylabel="Photon Count", color='k',
               labels=['Observed+FSR-Binned'] + ['_nolegend_' for o in eng.spectrograph.orders[:-1]], first=True)
    twin = axes[2].twinx()
    quick_plot(twin, lambda_pixel, convol_summed, ylabel=r"Flux (phot $cm^{-2} s^{-1})$", color='red',
               labels=['Conv.+Int.'] + ['_nolegend_' for o in eng.spectrograph.orders[:-1]],
               title="Convolved+Integrated vs. Observed Photons", xlabel="Wavelength (nm)", twin='red')
    if type_of_spectra == 'phoenix':
        twin.set_ylim(top=4.7e16)
    elif type_of_spectra == 'blackbody':
        twin.set_ylim(top=0.000235)
    fig.tight_layout()
    plt.subplots_adjust(top=0.92, right=0.6, left=0.1)
    plot_file = f'output_files/{type_of_spectra}/intplots_R0{eng.spectrograph.detector.design_R0}.pdf'
    fig.savefig(plot_file)
    logging.info(f'\nSaved intermediate plots to {plot_file}.')
    plt.show()
