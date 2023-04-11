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

"""
***BEFORE RUNNING THIS MAIN.PY SCRIPT***

*CHOOSE WHETHER TO GENERATE FAKE CALIBRATION DATA OR NOT (DEFAULT).

*CHOOSE THE TYPE OF SPECTRA TO SIMULATE:
'phoenix' -     Phoenix model spectrum of a 4300 K star with 0 (default).
'blackbody' -   Blackbody model spectrum of a 4300 K star with R_sun at 1kpc (choose for calibration data).
'delta' -       Narrow-width delta-like spectrum at the central wavelength 600 nm.

*CHOOSE TO DISPLAY INTERMEDIATE PLOTS (SLOWER) OR NOT (DEFAULT).
*CHOOSE TO CONDUCT A FULL MKID CONVOLUTION (SLOWER, DEFAULT) OR NOT.
*CHOOSE MAX # OF PHOTONS PER PIXEL (LARGER IS SLOWER, 1000 TAKES ~MINS).
*CHOOSE EXPOSURE TIME (LONGER RESULTS IN FEWER MERGED PHOTONS).
"""
gen_cal = False
type_of_spectra = 'blackbody'
plot_int = False
full_convolution = True
pixel_lim = 50000
exptime = 200 * u.s

tic = time.time()
print("***Beginning the MKID Spectrometer spectrum simulation.***")

# TODO move some of this into grating
u.photlam = u.photon / u.s / u.cm ** 2 / u.AA  # photon flux per wavelength
# c_beta = .1  # cos(beta)/cos(beta_center) at the end of m0
npix = 2048
R0 = 15
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

# The groove length and angles need to be ~self-consistent.
# We want to map l0 to the detector so:
# angular_dispersion = dbeta/dlambda
# dbeta = 2*np.arctan(pixel_size*npix/2/focal_length)
# dlambda = (l0_center/m0)

angular_dispersion = 2 * np.arctan(pixel_size * npix / 2 / focal_length) / (l0_center / m0)

# From the grating equation with Littrow:
# m0*l0_center=2*groove_length*np.sin(incident_angle) along with
# angular_dispersion = m0/groove_length/np.cos(incident_angle)
# solve for incident_angle, then groove_length:
incident_angle = np.arctan((l0_center / 2 * angular_dispersion).value) * u.rad
groove_length = m0 * l0_center / 2 / np.sin(incident_angle)
blaze_angle = incident_angle  # +10*u.deg  # a good bit off blaze
beta_central_pixel = incident_angle  # in Littrow alpha = beta

print(f"\nConfigured spectrometer and observation specifications."
      f"\n\tExposure time: {exptime}"
      f"\n\tSpectral resolution: {R0} at {l0}"
      f"\n\tOrders: {m0} to {m_max}"
      f"\n\tAngular dispersion: {angular_dispersion:.3e}")
if beta_central_pixel == incident_angle:
    print(f"\tIncident angle/angle of central pixel: {incident_angle:.3e} (Littrow)")
else:
    print(f"\tIncident angle: {incident_angle:.3e}"
          f"\n\tAngle of central pixel: {beta_central_pixel:.3e}")

detector = MKIDDetector(npix, pixel_size, R0, l0, generate_R0=gen_cal)
grating = GratingSetup(incident_angle, blaze_angle, groove_length)
spectrograph = SpectrographSetup(m0, m_max, l0, pixels_per_res_elem, focal_length, beta_central_pixel,
                                 grating, detector)

bandpasses = [AtmosphericTransmission(), TelescopeTransmission(reflectivity=.9), FilterTransmission(minwave, maxwave)]

if type_of_spectra == 'phoenix':
    spectra = [PhoenixModel(4300, 0, 4.8)]
    # non-list form in order to plot properly
    spectra_plot = PhoenixModel(4300, 0, 4.8)
    title = 'Input Spectrum: 4300 K Phoenix Stellar Model'
    print(f"\nObtained Phoenix model spectrum of star with T_eff of 4300 K.")
elif type_of_spectra == 'blackbody':
    # Optional comparison with a blackbody model:
    from synphot import SourceSpectrum
    from synphot.models import BlackBodyNorm1D

    spectra = [SourceSpectrum(BlackBodyNorm1D, temperature=4300)]  # flux for star of 1 R_sun at distance of 1 kpc
    spectra_plot = SourceSpectrum(BlackBodyNorm1D, temperature=4300)
    title = "Input Spectrum: 4300 K Blackbody Model ($R_\odot$, 1kpc)"
    print(f"\nObtained blackbody model spectrum of 4300 K star.")
elif type_of_spectra == 'delta':
    # Optional sanity check: the below produces a single block spectrum at 600 nm that is 40 nm wide.
    from synphot import SourceSpectrum
    from specutils import Spectrum1D

    w = np.linspace(400, 800, 4000) * u.nm
    tens = np.full(1950, 0)
    sp = np.full(100, 0.01)
    f = np.append(tens, sp)
    f = np.append(f, tens) * u.photlam
    spectra = [SourceSpectrum.from_spectrum1d(Spectrum1D(spectral_axis=w, flux=f)) / 1e20]
    spectra_plot = SourceSpectrum.from_spectrum1d(Spectrum1D(spectral_axis=w, flux=f)) / 1e20
    title = "Input Spectrum: 10 nm wide with 1 photlam flux at 600 nm"
    print(f"\nObtained 0.01 photlam narrow width spectrum at 600 nm "
          f"that is 10 nm wide with 0 photlam elsewhere.")
else:
    raise ValueError("Spectra must be 'phoenix,' 'blackbody,' or 'delta.'")

if plot_int:
    print("\nPlotting original spectrum...")
    # In our bandpass of interest:
    plt.plot(spectra_plot.waveset.to('nm'), spectra_plot(spectra_plot.waveset))
    plt.xlim([400, 800])
    plt.ylabel("Flux Density (phot $cm^{-2} s^{-1} \AA^{-1})$")
    plt.xlabel("Wavelength (nm)")
    plt.title(title)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    plt.grid()
    plt.show()
    # The full input incl. outside of bandpass
    plt.grid()
    plt.plot(spectra_plot.waveset.to('nm'), spectra_plot(spectra_plot.waveset))
    plt.ylabel("Flux Density (phot $cm^{-2} s^{-1} \AA^{-1})$")
    plt.xlabel("Wavelength (nm)")
    plt.title(title)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    plt.show()
    print("Shown.")

engine = Engine(spectrograph)

# Pre grating throughput effects, operates on wavelength grid of inbound flux
if not gen_cal:  # don't apply bandpasses to calibration spectra
    for i, s in enumerate(spectra):
        for b in bandpasses:
            s *= b
        spectra[i] = s
    print("\nApplied atmospheric, telescopic, and filter bandpasses to spectrum.")
else:
    from specutils import Spectrum1D
    from synphot import SpectralElement
    # finer grid spacing
    w = np.linspace(300, 1000, 1400000) * u.nm
    t = np.ones(1400000) * u.dimensionless_unscaled
    ones = Spectrum1D(spectral_axis=w, flux=t)
    spectra[0] *= SpectralElement.from_spectrum1d(ones)

if gen_cal:  # ensure blaze interpolation isnt cut off
    inbound = clip_spectrum(spectra[0], 340 * u.nm, 900*u.nm)
else:
    inbound = clip_spectrum(spectra[0], minwave, maxwave)

if plot_int:
    print("\nPlotting post-atmospheric spectrum...")
    plt.grid()
    plt.plot(spectra_plot.waveset.to('nm'), spectra_plot(spectra_plot.waveset), label="Input")
    plt.plot(inbound.waveset.to('nm'), inbound(inbound.waveset), label="Post-Atmo")
    plt.xlim([400, 800])
    plt.ylabel("Flux Density (phot $cm^{-2} s^{-1} \AA^{-1})$")
    plt.xlabel("Wavelength (nm)")
    plt.title("Post-Atmospheric Effects")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("Shown.")

# Grating throughput impact, is a function of wavelength and grating angles, handled for each order
blaze_efficiencies = spectrograph.blaze(inbound.waveset)
order_mask = spectrograph.order_mask(inbound.waveset.to(u.nm), fsr_edge=False)
# spectrograph.blaze returns 2D array of blaze efficiencies [wave.size, norders]
blazed_spectrum = blaze_efficiencies * inbound(inbound.waveset)
print(f"Multiplied blaze efficiencies with spectrum.")

checker = []  # holds order-masked spectrum in list for variable size
check_wave = []
for i in range(5):
    checker.append(blazed_spectrum[i, order_mask[i]])
    check_wave.append(inbound.waveset[order_mask[i]].to(u.nm))

if plot_int:
    print("\nPlotting transmission rate for each order...")
    plt.grid()
    for o, m, b in zip(spectrograph.orders, order_mask, blaze_efficiencies):
        plt.plot(inbound.waveset[m], b[m], label=f'Order {o}')
    plt.title("Blaze Efficiencies")
    plt.ylabel("Normalized Transmission")
    plt.xlabel("Wavelength (nm)")
    plt.legend()
    plt.show()
    print("Shown.")
if plot_int:
    print("\nPlotting blazed spectrum...")
    plt.grid()
    plt.plot(spectra_plot.waveset.to('nm'), spectra_plot(spectra_plot.waveset), label="Input")
    plt.plot(inbound.waveset.to('nm'), inbound(inbound.waveset), label="Post-Atmo")
    plt.xlim([400, 800])
    for i in range(5):
        plt.plot(check_wave[i], checker[i], label=f'Order {i + 5}')
    plt.ylabel("Flux Density (phot $cm^{-2} s^{-1} \AA^{-1})$")
    plt.xlabel("Wavelength (nm)")
    plt.title("Blazed Spectrum")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    print("Shown.")

broadened_spectrum = engine.optically_broaden(inbound.waveset, blazed_spectrum)
print("\nOptically broadened the spectrum.")

broad_checker = []
for i in range(5):  # broadening previous list of variable-length arrays
    broad_checker.append(engine.optically_broaden(check_wave[i], checker[i], axis=0))

if plot_int:
    print("\nPlotting optically-broadened spectrum...")
    plt.grid()
    for i in range(4):
        plt.plot(check_wave[i], checker[i], 'k')
    plt.plot(check_wave[4], checker[4], 'k', label="Input (Blazed)")
    for i in range(5):
        plt.plot(check_wave[i], broad_checker[i], label=f"Order {i + 5}")
    plt.title("Optically-broadened Spectrum")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux Density (phot $cm^{-2} s^{-1} \AA^{-1})$")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.legend()
    plt.tight_layout()
    plt.show()

if full_convolution:
    print("\nConducting full convolution with MKID response.")
    sampling_data = engine.determine_mkid_convolution_sampling(oversampling=osamp)
    result_wave, result, mkid_kernel = \
        engine.convolve_mkid_response(inbound.waveset, broadened_spectrum, *sampling_data, n_sigma_mkid=n_sigma_mkid,
                                      plot_int=plot_int)
else:
    print("\nConducting simplied 'convolution' with MKID response.")
    result_wave, result, mkid_kernel = engine.multiply_mkid_response(
        inbound.waveset, broadened_spectrum, oversampling=osamp, n_sigma_mkid=n_sigma_mkid, plot_int=plot_int)

pixel_samples_frac, pixel_max_npoints, pixel_rescale, dl_pixel, lambda_pixel, dl_mkid_max, sampling = sampling_data
pix_leftedge = spectrograph.pixel_to_wavelength(detector.pixel_indices, spectrograph.orders[:, None])

if gen_cal:  # must save calibration spectrum to file before running through detector sim
    norms = np.empty([5, 2048])  # obtaining and dividing by the normalization constants
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
    blaze_result = result / norms

    blaze_sumd = np.empty([5, 2048])
    for i in range(5):
        dx = result_wave[1, i, :].to(u.nm).value - result_wave[0, i, :].to(u.nm).value
        blaze_sumd[i, :] = np.sum(blaze_result[:, i, :].value, axis=0) * dx
        blaze_sumd[i, :] /= np.max(blaze_sumd)
    np.savetxt('blaze_sumd.csv', blaze_sumd, delimiter=',')
    np.savetxt('lambda_pixel.csv', lambda_pixel.to(u.nm).value, delimiter=',')

if plot_int:
    print("\nPlotting post-convolved spectrum by rough integration...")
    split_indices = np.empty([5, 2048])  # splitting pixel flux by the left edges of the expected pixel wavelength
    for i in range(2048):
        for j in range(5):
            if not gen_cal:
                if pix_leftedge[j, i].to(u.nm).value > 400:
                    split_indices[j, i] = np.where(np.abs(
                        check_wave[j].to(u.nm).value - pix_leftedge[j, i].to(u.nm).value) < 5e-4)[0][0]
            else:
                split_indices[j, i] = np.where(np.abs(
                    check_wave[j].to(u.nm).value - pix_leftedge[j, i].to(u.nm).value) < 5e-4)[0][0]

    fluxden = np.empty([5, 2048])  # regaining flux density by summing flux by previous indices and multiplying by dx
    for j in range(5):
        for i in range(2047):
            idx_left = int(split_indices[j, i])
            idx_right = int(split_indices[j, i + 1])
            fluxden[j, i] = scipy.integrate.trapz(
                broad_checker[j][idx_left:idx_right].to(u.photlam).value,
                x=check_wave[j][idx_left:idx_right].to(u.nm).value)
            fluxden[j, 2047] = fluxden[j, 2046]

    norms = np.empty([5, 2048])  # obtaining and dividing by the normalization constants
    for i in range(5):
        for j in range(2048):
            x = np.linspace(
                -3 * pixel_rescale[i, j].to(u.nm).value * dl_mkid_max.to(u.nm).si.value / sampling.to(
                    u.nm).si.value/2.355,
                3 * pixel_rescale[i, j].to(u.nm).value * dl_mkid_max.to(u.nm).si.value / sampling.to(
                    u.nm).si.value/2.355,
                len(mkid_kernel))
            dx = x[1] - x[0]
            norms[i, j] = np.sum(mkid_kernel) * dx / 0.9973
    result_plot = (result / (norms * u.nm)).to(u.photlam)

    # interpolate the result so they can be added
    inted = np.empty([5, 2048])
    for i in range(5):
        for j in range(2048):
            dx = result_wave[1, i, j].to(u.nm).value - result_wave[0, i, j].to(u.nm).value
            inted[i, j] = np.sum(result_plot[:, i, j].to(u.photlam).value) * dx

    # flux in = flux out comparison of convolution
    for i in range(5):
        flux_prec = scipy.integrate.trapz(fluxden[i, :], x=lambda_pixel[i, :].to(u.nm).value)
        flux_conv = scipy.integrate.trapz(inted[i, :], x=lambda_pixel[i, :].to(u.nm).value)
        # ensuring flux before and after are roughly equal
        print(f'Order {i + 5}: Flux In {flux_prec:.3e}, Flux Out {flux_conv:.3e}')
    for i in range(4):
        plt.plot(lambda_pixel[i, :].to(u.nm).value, fluxden[i, :], 'k', markersize=1)
    plt.plot(lambda_pixel[4, :].to(u.nm).value, fluxden[4, :], 'k', markersize=1, label='Input')
    for i in range(5):
        plt.plot(lambda_pixel[i, :], inted[i, :], label=f"Order {i + 5}")
    plt.grid()
    plt.ylabel("Flux Density (phot $cm^{-2} s^{-1} \AA^{-1})$")
    plt.xlabel("Wavelength (nm)")
    plt.title("Spectrum Convolved with MKID Response (Integrated)")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.legend()
    plt.show()
    print("Shown.")

t_photons, l_photons = engine.draw_photons(result_wave, result, exptime=exptime, limit_to=pixel_lim, plot_int=plot_int)

if plot_int:
    print("\nPlotting drawn photon list by rough histogram...")
    lambda_pixel = lambda_pixel[::-1, :]  # flipping order of arrays since photon draw is list of wavelengths
    result = result[:, ::-1, :]
    result_wave = result_wave[:, ::-1, :]
    fsr = spectrograph.fsr(spectrograph.orders)[::-1]
    pixel_rescale = pixel_rescale[::-1, :]

    hist_bins = np.empty((6, 2048))  # choosing histogram bins by using FSR of each pixel/wave
    hist_bins[0, :] = (lambda_pixel[0, :] - fsr[0] / 2).to(u.nm).value
    for i in range(4):
        hist_bins[i + 1, :] = (lambda_pixel[i, :] + fsr[i] / 2).to(u.nm).value
    hist_bins[5, :] = np.full(2048, 900)

    hist = np.empty((5, 2048))  # photons binned at this step
    for j in range(2048):
        hist[:, j], bin_edges = np.histogram(l_photons[j].to(u.nm).value, bins=hist_bins[:, j], density=False)

    result_r = np.empty((5, 2048))  # something to compare to: the post-convolved, but undrawn spectrum
    for i in range(5):
        for j in range(2048):
            result_r[i, j] = np.sum(result[:, i, j].decompose().to(u.ph / u.cm ** 2 / u.s).value)
    result_raw = result_r * hist.flatten().max() / result_r.flatten().max()  # normalizedish to compare better

    plt.grid()
    for i in range(5):
        plt.plot(lambda_pixel[i, :], hist[i, :], label=f'Order {9 - i}')
    for i in range(4):
        plt.plot(lambda_pixel[i, :], result_raw[i, :], 'k')
    plt.plot(lambda_pixel[4, :], result_raw[4, :], 'k', label='Input')
    plt.ylabel("Total Photons")
    plt.xlabel("Wavelength (nm)")
    plt.title("Drawn Photons vs. MKID Convolved Step")
    plt.legend()
    plt.show()
    print("Shown.")

# Merge and filter
photons, observed = detector.observe(t_photons, l_photons)

# checking original vs. final spectrum
if plot_int:
    print("\nPlotting output spectrum...")
    final_list = []
    resid_map = np.arange(2048, dtype=int) * 10 + 100
    for i in range(2048):  # sorting photons by resID (i.e. pixel)
        idx = np.where(photons.resID == resid_map[i])
        final_list.append(photons[:observed].wavelength[idx].tolist())

    final = np.empty((5, 2048))
    for j in range(2048):  # sorting by histogram bins as before
        final[:, j], bin_edges = np.histogram(final_list[j], bins=hist_bins[:, j], density=False)
    plt.grid()
    for i in range(4):
        plt.plot(lambda_pixel[i, :], result_raw[i, :], 'blue')
    plt.plot(lambda_pixel[4, :], result_raw[4, :], 'blue', label='Input')
    for i in range(5):
        plt.plot(lambda_pixel[i, :], final[i, :], '.', label=f'Order {9-i}')
    plt.title("Input and Observed Spectra")
    plt.ylabel("Total Photons")
    plt.xlabel("Wavelength (nm)")
    plt.legend()
    plt.show()

    final_interp = np.empty([5, 10000])  # interpolating both to sum to complete spectrum
    result_interp = np.empty([5, 10000])
    wave = np.linspace(350, 800, 10000)
    for i in range(5):
        final_interp[i, :] = interp.interp1d(lambda_pixel[i, :], final[i, :],
                                             fill_value=0, bounds_error=False, copy=False)(wave)
        result_interp[i, :] = interp.interp1d(lambda_pixel[i, :], result_raw[i, :],
                                              fill_value=0, bounds_error=False, copy=False)(wave)
    final_sumd = np.sum(final_interp, axis=0)
    result_sumd = np.sum(result_interp, axis=0)

    with open('convolved_blaze.csv') as f:
        conv_blaze = np.loadtxt(f, delimiter=",")[::-1,:]
    blaze_interp = np.empty([5, 10000])  # interpolating blaze efficiencies to divide out of spectrum
    for i in range(5):
        blaze_interp[i] = interp.interp1d(lambda_pixel[i,:], conv_blaze[i,:],
                                          fill_value=0, bounds_error=False, copy=False)(wave)
    blaze_sumd = np.sum(blaze_interp, axis=0)
    final_sumd /= blaze_sumd
    result_sumd /= blaze_sumd

    plt.grid()
    plt.plot(wave, result_sumd, 'k', label='Input')
    plt.plot(wave, final_sumd, label='Output')
    plt.title("Input and Observed Spectra (Divided Out Blaze)")
    plt.ylabel("Total Photons")
    plt.xlabel("Wavelength (nm)")
    plt.legend()
    plt.show()
    print("Shown.")

# Dump to HDF5
# TODO this will need work as the pipeline will probably default to MEC HDF headers
from mkidpipeline.steps import buildhdf

buildhdf.buildfromarray(photons[:observed], user_h5file=f'./spec_{type_of_spectra}_{pixel_lim}ppp.h5')
# at this point we are simulating the pipeline and have gone past the "wavecal" part. Next is >to spectrum.
print("\nCompiled data to h5 file.")
toc = time.time()
print(f"\n***Completed MKID Spectrometer spectrum simulation in {round((toc - tic) / 60, 2)} minutes.***")

#print(f"\n***Beginning spectral order sorting via Gaussian Mixture Models.***")
