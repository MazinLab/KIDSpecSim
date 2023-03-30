import numpy as np
import scipy.interpolate as interp
import scipy
import scipy.signal as sig
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import time
import astropy.units as u
from spectra import PhoenixModel, AtmosphericTransmission, \
    FilterTransmission, TelescopeTransmission, clip_spectrum
from spectrograph import GratingSetup, SpectrographSetup
from detector import MKIDDetector
from engine import Engine
from synphot import SourceSpectrum
from specutils import Spectrum1D

"""
Generates fake calibration data from flat fluxden spectrum.

***BEFORE RUNNING THIS ORDERSORT_CALIBRATION.PY SCRIPT***

*MAKE SURE R0 ASSIGNMENT IN DETECTOR.PY IS FIXED (I.E. NOT RANDOM). Then change back to random.
*CHOOSE R0 AT 800 NM (15 IS DEFAULT).
*CHOOSE IF YOU WANT PLOTS (DEFAULT) OR NOT.
"""
R0 = 15
plot = False

u.photlam = u.photon / u.s / u.cm ** 2 / u.AA  # photon flux per wavelength
exptime = 1 * u.s
# c_beta = .1  # cos(beta)/cos(beta_center) at the end of m0
npix = 2048
l0 = 800 * u.nm
m0 = 5
m_max = 9
l0_center = l0 / (1 + 1 / (2 * m0))
# l0_fsr = l0_center/m0
R0_min = .8 * R0
n_sigma_mkid = 3
osamp = 10
minwave = 400 * u.nm
maxwave = 800 * u.nm
pixels_per_res_elem = 2.5
pixel_size = 20 * u.micron
focal_length = 350 * u.mm

angular_dispersion = 2 * np.arctan(pixel_size * npix / 2 / focal_length) / (l0_center / m0)
incident_angle = np.arctan((l0_center / 2 * angular_dispersion).value) * u.rad
groove_length = m0 * l0_center / 2 / np.sin(incident_angle)
blaze_angle = incident_angle  # +10*u.deg  # a good bit off blaze
beta_central_pixel = incident_angle  # in Littrow alpha = beta

detector = MKIDDetector(npix, pixel_size, R0, l0)
grating = GratingSetup(incident_angle, blaze_angle, groove_length)
spectrograph = SpectrographSetup(m0, m_max, l0, pixels_per_res_elem, focal_length, beta_central_pixel,
                                 grating, detector)

bandpasses = [FilterTransmission(minwave, maxwave)]

w = np.linspace(400, 800, 4000) * u.nm
f = np.full(4000, 1) * u.photlam
spectra = [SourceSpectrum.from_spectrum1d(Spectrum1D(spectral_axis=w, flux=f))]
spectra_plot = SourceSpectrum.from_spectrum1d(Spectrum1D(spectral_axis=w, flux=f))

if plot:
    plt.plot(spectra_plot.waveset.to('nm'), spectra_plot(spectra_plot.waveset))
    plt.ylabel("Flux Density (phot $cm^{-2} s^{-1} \AA^{-1})$")
    plt.xlabel("Wavelength (nm)")
    plt.title("Input Spectrum")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    plt.grid()
    plt.show()

engine = Engine(spectrograph)

for i, s in enumerate(spectra):
    for b in bandpasses:
        s *= b
    spectra[i] = s

inbound = clip_spectrum(spectra[0], minwave, maxwave)

blaze_efficiencies = spectrograph.blaze(inbound.waveset)
order_mask = spectrograph.order_mask(inbound.waveset.to(u.nm), fsr_edge=False)
blazed_spectrum = blaze_efficiencies * inbound(inbound.waveset)
broadened_spectrum = engine.optically_broaden(inbound.waveset, blazed_spectrum)

sampling_data = engine.determine_mkid_convolution_sampling(oversampling=osamp)
result_wave, result, mkid_kernel = engine.convolve_mkid_response(
    inbound.waveset, broadened_spectrum, *sampling_data, n_sigma_mkid=n_sigma_mkid, plot_int=False)

blaze_wave, blaze_result, mkid_kernel = engine.convolve_mkid_response(
    inbound.waveset, broadened_spectrum, *sampling_data, n_sigma_mkid=n_sigma_mkid, plot_int=False)

new_wave = np.linspace(100, 1000, 10000)  # interpolate result so relative fractions can be calculated
result_interp = np.zeros([10000, 5, 2048])
for i in range(5):
    for j in range(2048):
        result_interp[:, i, j] = interp.interp1d(result_wave[:, i, j], result[:, i, j],
                                                 fill_value=0, bounds_error=False, copy=False)(new_wave)

wave_frac = np.zeros([10000, 5, 2048])
for j in range(2048):
    sumd = np.sum(result_interp[:,:,j], axis=1)
    wave_frac[:,:, j] = result_interp[:,:,j]/sumd[:,None]
np.nan_to_num(wave_frac, copy=False, nan=0)

np.savetxt(f'fraccal_wave.csv', new_wave, delimiter=',')  # save files to be imported elsewhere
for i in range(5):
    np.savetxt(f'fraccal{i}_R0of{R0}.csv', wave_frac[:,i,:], delimiter=',')

pixel_samples_frac, pixel_max_npoints, pixel_rescale, dl_pixel, lambda_pixel, dl_mkid_max, sampling = sampling_data
pix_leftedge = spectrograph.pixel_to_wavelength(detector.pixel_indices, spectrograph.orders[:, None])

if plot:
    norms = np.empty([5, 2048])  # obtaining and dividing by the normalization constants
    norms_blaze = np.empty([5, 2048])
    for i in range(5):
        for j in range(2048):
            x = np.linspace(
                -3 * pixel_rescale[i, j].to(u.nm).value * dl_mkid_max.to(u.nm).si.value / sampling.to(
                    u.nm).si.value / 2.355,
                3 * pixel_rescale[i, j].to(u.nm).value * dl_mkid_max.to(u.nm).si.value / sampling.to(
                    u.nm).si.value / 2.355,
                len(mkid_kernel))
            dx = x[1] - x[0]
            norms[i, j] = np.sum(mkid_kernel) * dx / 0.9973
    result_plot = (result.to(u.ph / u.cm ** 2 / u.s) / (norms * u.nm)).to(u.photlam)
    blaze_plot = blaze_result / norms

    # interpolate the result so they can be added
    inted = np.empty([5, 2048])
    blaze_inted = np.empty([5, 2048])
    for i in range(5):
        for j in range(2048):
            dx = result_wave[1, i, j].to(u.nm).value - result_wave[0, i, j].to(u.nm).value
            inted[i, j] = np.sum(result_plot[:, i, j].to(u.photlam).value) * dx
            blaze_inted[i, j] = np.sum(blaze_plot[:, i, j].value) * dx

    new_blaze = blaze_inted / blaze_inted.max()
    np.savetxt('convolved_blaze.csv', new_blaze, delimiter=',')

    for i in range(5):
        plt.plot(lambda_pixel[i, :], inted[i, :] / new_blaze[i, :], label=f"Order {i + 5}")
    plt.grid()
    plt.ylabel("Flux Density (phot $cm^{-2} s^{-1} \AA^{-1})$")
    plt.xlabel("Wavelength (nm)")
    plt.title("Spectrum Convolved with MKID Response (Integrated)")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.legend()
    plt.show()
    for i in range(5):
        plt.plot(lambda_pixel[i, :], new_blaze[i, :], label=f"Order {i + 5}")
    plt.grid()
    plt.ylabel("Relative Transmission")
    plt.xlabel("Wavelength (nm)")
    plt.title("Blaze Convolved with MKID Response (Integrated)")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.legend()
    plt.show()
    print("Shown.")
