import numpy as np
import scipy.interpolate as interp
import scipy
import scipy.signal as sig
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
import time

import astropy.units as u
from ucsbsim.spectra import PhoenixModel, AtmosphericTransmission, FilterTransmission, TelescopeTransmission
from ucsbsim.spectrograph import GratingSetup, SpectrographSetup
from ucsbsim.detector import MKIDDetector
from ucsbsim.engine import Engine
from ucsbsim.spectra import clip_spectrum


tic=time.time()

#TODO move some of this into grating
u.photlam=u.photon/u.s/u.cm**2/u.AA
exptime = 1*u.s
# c_beta = .1  # cos(beta)/cos(beta_center) at the end of m0
npix = 2048
R0=15
l0=800 * u.nm
m0=5
m_max = 9
l0_center=l0/(1+1/(2*m0))
# l0_fsr = l0_center/m0
R0_min=.8*R0
n_sigma_mkid=3
osamp = 10
minwave=400 * u.nm
maxwave=800 * u.nm
pixels_per_res_elem = 2.5
pixel_size = 20 * u.micron  #20 um in nm
focal_length = 350 * u.mm

# The groove length and angles need to be ~self-consistent
# From the grating equation with Littrow
# m0*l0_center=2*groove_length*np.sin(incident_angle)
# We want to map l0 to the detector so
# A=dbeta/dlambda = m0/groove_length/np.cos(incident_angle)
# DELTA beta = 2*np.arctan(pixel_size*npix/2/focal_length)
# DELTA lambda = (l0_center/m0)
# two equations two unknowns:

angular_dispersion=2*np.arctan(pixel_size*npix/2/focal_length)/(l0_center/m0)
incident_angle = np.arctan((l0_center/2 * angular_dispersion).value)*u.rad
groove_length = m0*l0_center/2/np.sin(incident_angle)
blaze_angle = incident_angle#+10*u.deg  # a good bit off blaze

beta_central_pixel = incident_angle

detector = MKIDDetector(npix, pixel_size, R0, l0)
grating = GratingSetup(incident_angle, blaze_angle, groove_length)
spectrograph = SpectrographSetup(m0, m_max, l0, pixels_per_res_elem, focal_length, beta_central_pixel,
                                 grating, detector)

bandpasses = [AtmosphericTransmission(),
              TelescopeTransmission(reflectivity=.9),
              FilterTransmission(minwave, maxwave)]

spectra = [PhoenixModel(4300, 0, 4.8)]


engine = Engine(spectrograph)


# Pre grating throughput effects, operates on wavelength grid of inbound flux
for i,s in enumerate(spectra):
    for b in bandpasses:
        s *= b
    spectra[i] = s

inbound = clip_spectrum(spectra[0], minwave, maxwave)

# Grating throughput impact, is a function of wavelength and grating angles, needs to be handled for each order
blaze_efficiencies = spectrograph.blaze(inbound.waveset)

# order_mask = spectrograph.order_mask(inbound.waveset, fsr_edge=False)
# for o,m, b in zip(spectrograph.orders, order_mask, blaze_efficiencies):
#     plt.plot(inbound.waveset[m], b[m], label=f'Order {o}')
# plt.legend()
# plt.show()

blazed_spectrum = blaze_efficiencies*inbound(inbound.waveset) # .blaze returns 2D array of blaze efficiencies [wave.size, norders]

broadened_spectrum = engine.opticaly_broaden(inbound.waveset, blazed_spectrum)

full_convolution=True
if full_convolution:
    sampling_data = engine.determine_mkid_convolution_sampling(oversampling=osamp)

    result_wave, result = engine.convolve_mkid_response(inbound.waveset, broadened_spectrum, *sampling_data,
                                                        n_sigma_mkid=n_sigma_mkid)
else:
    result_wave, result = engine.multiply_mkid_response(inbound.waveset, broadened_spectrum,
                                                        oversampling=osamp, n_sigma_mkid=n_sigma_mkid)


# # Compare the convolved result with the "crude" approximation where a gaussian is scaled to the pixels average flux
# flux = u.photlam*interp.interp1d(inbound.waveset.to('nm').value, broadened_spectrum[0],
#                        fill_value=0, bounds_error=False, copy=False)(lambda_pixel[0,0])*dl_pixel[0,0]*mkid_kernel
# trueflux = pixel_rescale[0,0]*result[:,0,0]


t_photons, l_photons = engine.draw_photons(result_wave, result, limit_to=1000)

# Merge and filter
photons, observed = detector.observe(t_photons, l_photons)

# Dump to HDF5
# TODO this will need work as the pipeline will probably default to MEC HDF headers
from mkidpipeline.steps import buildhdf
buildhdf.buildfromarray(photons[:observed], user_h5file=f'./spec.h5')
toc=time.time()
print(f"Done in {toc-tic} s")
