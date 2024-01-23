# """Purpose of file is to determine required R for optimal efficiency of stock gratings and to determine
# order efficiency for MKIDs of fiducial R"""


"""Purpose of file is to determine required R for optimal efficiency of stock gratings and to determine
order efficiency for MKIDs of fiducial R"""

import numpy as np
import scipy.interpolate as interp
import scipy
import scipy.signal as sig
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
import time

import astropy.units as u

u.photlam = u.photon / u.s / u.cm ** 2 / u.AA
from ucsbsim.spectrograph import GratingSetup, SpectrographSetup, NEWPORT_GRATINGS
from ucsbsim.detector import MKIDDetector

m = 4
R0 = 15
R0_l = 800
# c_beta = .1  # cos(beta)/cos(beta_center) at the end of m0
npix = 2048
R0 = R0
R0_l = R0_l * u.nm
l0 = 800 * u.nm
m0 = m
minwave = 400 * u.nm
maxwave = 800 * u.nm
pixels_per_res_elem = 2.5
pixel_size = 20 * u.micron  # 20 um in nm
focal_length = 300 * u.mm
MKID_FWHM_MARGIN = 1.33


def grating_explore(grating, name, amin=20, n_a=5, plot=False, r800_lim=7, plot_blaze=False,
                    verbose=False, min_coverage=.2, m0=2, m_max=30):
    detector = MKIDDetector(npix, pixel_size, R0, R0_l, randomize_r0=None)
    spectrograph = SpectrographSetup(m0, m_max, l0, pixels_per_res_elem, focal_length, grating.delta, grating, detector)
    for i in np.linspace(amin, grating.delta.value, num=n_a):
        spectrograph.m0 = m0
        spectrograph.m_max = m_max
        spectrograph.set_beta_center(i)
        edges = spectrograph.edge_wave(fsr=True)
        use_orders = (edges[:, 0] <= maxwave) & (edges[:, 1] >= minwave)
        if use_orders.sum() >=2:
            spectrograph.m0, spectrograph.m_max = spectrograph.orders[use_orders][[0, -1]]
            if plot or plot_blaze:
                spectrograph.plot_echellogram(center_orders=True, title=name, blaze=plot_blaze)
            edges = spectrograph.edge_wave(fsr=True)
            detector_edges = spectrograph.edge_wave(fsr=False)

            gaps = np.array([np.select([detector_edges > edges, detector_edges <= edges],
                                       [detector_edges - edges, np.zeros_like(edges)])[:, 0],
                             np.select([detector_edges < edges, detector_edges > edges],
                                       [edges - detector_edges, np.zeros_like(edges)])[:, 1]]).T * u.nm

            # observable_waves = np.array([np.select([detector_edges > edges, detector_edges <= edges],
            #                            [detector_edges, edges])[:, 0],
            #                  np.select([detector_edges < edges, detector_edges > edges],
            #                            [detector_edges, edges])[:, 1]]).T * u.nm

            needed_mkid_r = detector_edges[:-1]/np.abs(np.diff(detector_edges,axis=0))*MKID_FWHM_MARGIN

        if use_orders.sum() < 2:
            if verbose:
                print(f"  a={i:.1f} deg < 2 orders")
            continue

        r800 = (needed_mkid_r[0, 1] * detector_edges[0, 1]) / maxwave

        if r800>r800_lim:
            if verbose:
                print(f"  R800 > {r800_lim}")
            continue

        if (1 - gaps.sum() / (maxwave - minwave)) < min_coverage:
            if verbose:
                print(f"  Coverage < {min_coverage*100:.0f:}%")
            continue

        print(f"  a={i:.1f} deg m={spectrograph.m0}-{spectrograph.m_max}")
        print(f"     % coverage: {1 - gaps.sum() / (maxwave - minwave):.2f}")
        print(f"     R800 {r800:.1f}")
        print(f"     System R: {spectrograph.average_res[0]:.0f}")
        o=f'{np.log10(spectrograph.mean_blaze_eff_est(n=200).mean()):.1f}'
        print(f"     Blaze %: {spectrograph.mean_blaze_eff_est(n=200).mean():.2f} O(%)={o}")


def grating_report(name, alpha=None, empiric_factor=1.0, m0=9, m_max=19):
    detector = MKIDDetector(npix, pixel_size, R0, R0_l, randomize_r0=None)
    grating = NEWPORT_GRATINGS[name]  # or maybe 452E
    grating.empiric_blaze_factor = empiric_factor
    spectrograph = SpectrographSetup(m0, m_max, l0, pixels_per_res_elem, focal_length, grating.delta, grating, detector)
    spectrograph.set_beta_center(grating.delta if alpha is None else alpha)
    spectrograph.plot_echellogram(center_orders=True, blaze=True)
    edges = spectrograph.edge_wave(fsr=True)
    detector_edges = spectrograph.edge_wave(fsr=False)

    gaps = np.array([np.select([detector_edges > edges, detector_edges <= edges],
                               [detector_edges - edges, np.zeros_like(edges)])[:, 0],
                     np.select([detector_edges < edges, detector_edges > edges],
                               [edges - detector_edges, np.zeros_like(edges)])[:, 1]]).T * u.nm

    needed_mkid_r = detector_edges[:-1] / np.abs(np.diff(detector_edges, axis=0)) * MKID_FWHM_MARGIN
    r800 = (needed_mkid_r[0, 1] * detector_edges[0, 1]) / maxwave
    print(f"  a={spectrograph.beta_central_pixel:.1f} deg m={spectrograph.m0}-{spectrograph.m_max}")
    print(f"     % coverage: {1 - gaps.sum() / (maxwave - minwave):.2f}")
    print(f"     R800 {r800:.1f}")
    print(f"     System R: {spectrograph.average_res[0]:.0f}")
    o = f'{np.log10(spectrograph.mean_blaze_eff_est(n=200).mean()):.1f}'
    print(f"     Blaze %: {spectrograph.mean_blaze_eff_est(n=200).mean():.2f} O(%)={o}")

    return spectrograph



# The groove length and angles need to be ~self-consistent
# From the grating equation with Littrow
# m0*l0_center=2*groove_length*np.sin(incident_angle=reflected_angle)
# We want to map l0 to the detector so
# A=dbeta/dlambda = m0/groove_length/np.cos(incident_angle)
# DELTA beta = 2*np.arctan(pixel_size*npix/2/focal_length)
# DELTA lambda = (l0_center/m0)
# two equations two unknowns:

# angular_dispersion = 2 * np.arctan(pixel_size * npix / 2 / focal_length) / (l0_center / m0)


# Investigate grating
NEWPORT_GRATINGS['JB1']=GratingSetup(None, 34.38 * u.deg, 1e6*u.nm/147.84)
NEWPORT_GRATINGS['JB2']=GratingSetup(None, 22.24 * u.deg, 1e6*u.nm/170.87)
NEWPORT_GRATINGS['JB3']=GratingSetup(None, 15.25 * u.deg, 1e6*u.nm/184.95)

# for name, grating in NEWPORT_GRATINGS.items():
#     print(f'Grating {name}')
#     grating_explore(grating, name, amin=grating.delta.value)

# for name in ('451E', '452E'):
#     grating = NEWPORT_GRATINGS[name]
#     print(f'Grating {name}')
#     grating_explore(grating, name, amin=grating.delta.value/20, n_a=20, r800_lim=8,
#                     plot_blaze=False, verbose=False, min_coverage=.2)

# grating_report('149E', empiric_factor=0.82, m0=9, m_max=19)


spec=grating_report('451E', alpha=28.3, empiric_factor=1, m0=4, m_max=8)
spec.pixel_wavelengths()[:, np.linspace(0, 2047, num=5, dtype=int)].to('um').value
print(spec.grating.alpha-spec.grating.delta,spec.grating.delta,spec.grating.d/1000)

def determine_empric_blaze_factor(spec, o, w, newport):
    eff = spec.grating.blaze(spec.grating.beta(w, o), o)
    return newport/eff
