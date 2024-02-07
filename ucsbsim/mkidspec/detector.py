import numpy as np
import astropy.units as u
import logging
from scipy.constants import c

from ucsbsim.filterphot import mask_deadtime
from ucsbsim.mkidspec.engine import draw_photons


def wave_to_phase(waves, minwave, maxwave):
    """
    :param waves: wavelengths in nm
    :param minwave: minimum wavelength of spectrograph
    :param maxwave: maximum wavelength of spectrograph
    :return: phase values corresponding to wavelength
    """
    shape = np.shape(waves)
    waves = waves.flatten()
    if isinstance(waves, u.Quantity):
        waves = waves.to(u.nm).value
    # range is -1(pi/2) to 1(pi/2)
    # smaller wavelengths wrap beginning at -1 and larger wavelengths wrap beginning at 1
    # line from (freq_minw, -0.8) to (freq_maxw, -0.2)
    # if -1.1, negative: -1.1+2*max_phase, positive: if 1.1, 1.1+2*min_phase, repeating until between -1 to 1
    # linear equation: y = (y2-y1)/(x2-x1)*(x-x1) + y1 = 0.6/(freq_maxw-freq_minw)*(x-freq_minw) - 0.8
    freq_minw = (c * u.m / u.s / minwave).decompose()  # this will be mapped to -0.8
    freq_maxw = (c * u.m / u.s / maxwave).decompose()  # this will be mapped to -0.2
    freqs = (c * u.m / u.s / (waves * u.nm)).decompose()  # converted wavelength to frequency (Hz)
    phases = np.nan_to_num((0.6 / (freq_maxw - freq_minw) * (freqs - freq_minw)).decompose().value - 0.8,
                           posinf=0, neginf=-1)
    phases = np.reshape(phases, shape)
    return phases


class MKIDDetector:
    def __init__(
            self,
            n_pix: int,
            pixel_size: u.Quantity,
            design_R0: float,
            l0: u.Quantity,
            R0s: np.ndarray = None,
            phase_offsets: np.ndarray = None,
            resid_map: np.ndarray = None
    ):
        """
        Simulation of an MKID detector array.
        :param int n_pix: number of pixels in linear array
        :param u.Quantity pixel_size: physical size of each pixel in astropy units
        :param float R0: spectral resolution of the longest wavelength in spectrometer range
        :param u.Quantity l0: longest wavelength in spectrometer range in astropy units
        :param np.ndarray R0s: array of R0s that deviate slightly from design as expected, None means no deviation
        :param np.ndarray phase_offsets: the phase center offset factor for each pixel
        :param np.ndarray resid_map: the IDs for each resonator (pixel).
        """
        self.n_pixels = n_pix
        self.pixel_size = pixel_size
        self.length = self.n_pixels * pixel_size
        self.waveR0 = l0
        self.design_R0 = design_R0
        self.pixel_indices = np.arange(self.n_pixels, dtype=int)
        if R0s is None:
            self.R0s = np.ones(self.n_pixels) * self.design_R0
        else:
            self.R0s = R0s
        if phase_offsets is None:
            self.pixel_phase_offsets = np.ones(self.n_pixels)
        else:
            self.pixel_phase_offsets = phase_offsets
        self.resid_map = resid_map

    def R0(self, pixel: int):
        """
        :param pixel: the pixel index or indices
        :return: spectral resolution for given pixel
        """
        if pixel not in self.pixel_indices:
            raise ValueError(f"Pixel {pixel + 1} not in instantiated detector, max of {self.n_pixels}.")
        if len(self.R0s) != self.n_pixels:
            raise ValueError('The number of R0s does not match number of pixels.')
        elif np.abs(np.average(self.R0s)-self.design_R0) > 1:
            raise ValueError('The user-supplied array of R0s and design R0 do not match.')
        return self.R0s[pixel.astype(int)]

    def mkid_constant(self, pixel):
        """
        :param pixel: the pixel index or indices
        :return: MKID constant for given pixel, R0*l0
        """
        return self.R0(pixel) * self.waveR0

    def mkid_resolution_width(self, wave, pixel):
        """
        :param wave: wavelength(s) as u.Quantity
        :param pixel: the pixel index or indices
        :return: FWHM of the MKID at given wavelength and pixel
        """
        rc = self.mkid_constant(pixel)
        try:
            if wave.shape != rc.shape:
                if wave.ndim == rc.ndim:
                    raise ValueError('Arrays of the same dimensions much have matching shapes')
                if wave.shape[-1] != rc.shape[-1]:
                    raise ValueError('Arrays of differing dimension must match along the final dimension')
                rc = rc[None, :]
        except AttributeError:  # allow non-array args
            pass
        return wave ** 2 / rc

    def observe(self, convol_wave, convol_result, phase: bool = True, minwave=None, maxwave=None, **kwargs):
        """
        :param convol_wave: wavelength array that matches convol_result
        :param convol_result: convolution array
        :param bool phase: True if resulting recarray to be in phase values not wavelength
        :param minwave: pass value of spectrograph minwave for phase=True
        :param maxwave: pass value of spectrograph maxwave for phase=True
        :param kwargs: additional keyword args to pass to draw_photons (exptime, area, etc.)
        :return: recarray of observed photons, total number observed
        """
        arrival_times, arrival_wavelengths, reduce_factor = draw_photons(convol_wave, convol_result, **kwargs)

        from mkidcore.binfile.mkidbin import PhotonNumpyType
        pixel_count = np.array([x.size for x in arrival_times])
        total_photons = pixel_count.sum()

        merge_time_window_s = 1e-6 * u.s
        MIN_TRIGGER_ENERGY = 1 / (1.5 * u.um)
        SATURATION_WAVELENGTH_NM = 350 * u.nm
        DEADTIME = 10 * u.us

        logging.info("\nBeginning MKID detector observation sequence with:"
                     f"\n\tMinimum trigger energy: {MIN_TRIGGER_ENERGY:.3e}"
                     f"\n\tPhoton merge time: {merge_time_window_s:.0e}"
                     f"\n\tSaturation wavelength: {SATURATION_WAVELENGTH_NM}"
                     f"\n\tDeadtime: {DEADTIME}")
        logging.warning(f'Simulated dataset may take up to {total_photons * 16 / 1024 ** 3:.2} GB of RAM.')

        if self.resid_map is None:  # TODO check if shape of resid matches npix
            self.resid_map = np.arange(pixel_count.size, dtype=int) * 10 + 100  # something arbitrary

        photons = np.recarray(total_photons, dtype=PhotonNumpyType)
        photons[:] = 0
        photons.weight[:] = 1.0
        observed = 0
        total_merged = 0
        total_missed = []
        # Compute photon arrival times and wavelengths for each photon
        for pixel, n in enumerate(pixel_count):
            if not n:
                continue

            # get photons and arrival times for pixel
            a_times = arrival_times[pixel]
            arrival_order = a_times.argsort()
            a_times = a_times[arrival_order]
            energies = 1 / arrival_wavelengths[pixel].to(u.um)[arrival_order]

            # merge photon energies within 1us
            to_merge = (np.diff(a_times) < merge_time_window_s).nonzero()[0]
            if to_merge.size:
                cluster_starts = to_merge[np.concatenate(([0], (np.diff(to_merge) > 1).nonzero()[0] + 1))]
                cluser_last = to_merge[(np.diff(to_merge) > 1).nonzero()[0]] + 1
                cluser_last = np.append(cluser_last, to_merge[-1] + 1)  # inclusive
                for start, stop in zip(cluster_starts, cluser_last):
                    merge = slice(start + 1, stop + 1)
                    energies[start] += energies[merge].sum()
                    energies[merge] = np.nan
                    total_merged += energies[merge].size

            # TODO for LANL we determined the energies via the R AFTER coincidence
            #  binning. That isn't possible with his approach (as far as I can tell)
            measured_energies = energies

            # Filter those that wouldn't trigger
            will_trigger = measured_energies > MIN_TRIGGER_ENERGY
            if not will_trigger.any():
                continue

            # Drop photons that arrive within the deadtime
            detected = mask_deadtime(a_times[will_trigger], DEADTIME.to(u.s))

            missed = will_trigger.sum() - detected.sum()
            total_missed.append(missed)

            a_times = a_times[will_trigger][detected]
            measured_wavelengths = 1000 / measured_energies[will_trigger][detected]
            measured_wavelengths.clip(SATURATION_WAVELENGTH_NM, out=measured_wavelengths)

            # Add photons to the pot
            sl = slice(observed, observed + a_times.size)
            photons.wavelength[sl] = measured_wavelengths
            photons.time[sl] = a_times * 1e6  # in microseconds
            photons.resID[sl] = self.resid_map[pixel]
            observed += a_times.size

        if phase:
            # replace wavelength information with phases:
            # range is -1(pi/2) to 1(pi/2)
            # smaller wavelengths wrap beginning at -1 and larger wavelengths wrap beginning at 1
            # line from (freq_minw, -0.8) to (freq_maxw, -0.2)
            # if -1.1, negative: -1.1+2*max_phase, positive: if 1.1, 1.1+2*min_phase, repeating until between -1 to 1
            # linear equation: y = (y2-y1)/(x2-x1)*(x-x1) + y1 = 0.6/(freq_maxw-freq_minw)*(x-freq_minw) - 0.8
            photons.wavelength = wave_to_phase(photons.wavelength, minwave, maxwave)
            for j in self.pixel_indices:  # sorting photons by resID (i.e. pixel) and multiplying phase center offsets
                try:
                    photons.wavelength[np.where(photons.resID == self.resid_map[j])] *= self.pixel_phase_offsets[j]
                except TypeError:
                    print("Phases were not offset because offset values were not provided.")
            if photons.wavelength.size:
                for n, j in enumerate(photons.wavelength):
                    while photons.wavelength[n] < -1:
                        photons.wavelength[n] += 2
                    while photons.wavelength[n] > 1:
                        photons.wavelength[n] -= 2

        logging.info(f'Completed detector observation sequence. '
                     f'{total_merged} photons had their energies merged, '
                     f'{np.sum(total_missed)} photons were missed due to deadtime,'
                     f'and {observed} photons were observed.')
        return photons, observed, reduce_factor
