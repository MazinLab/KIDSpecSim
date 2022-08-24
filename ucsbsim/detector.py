import numpy as np
import astropy.units as u
from mkidsim.filterphot import mask_deadtime

class MKIDDetector:
    def __init__(self, n, pixel_size, R0, wave_R0):
        self.n_pixels = n
        self.pixel_size = pixel_size
        self.length = self.n_pixels * pixel_size
        self.waveR0 = wave_R0
        self.design_R0 = R0
        self.pixel_indices = np.arange(self.n_pixels, dtype=int)
        self._R0s = None

    def R0(self, pixel):
        if self._R0s is None:
            self._R0s = np.random.uniform(.85, 1.15, size=self.n_pixels) * self.design_R0
        return self._R0s[pixel.astype(int)]

    def mkid_constant(self, pixel):
        """ R0*l0 divide by wave to get effective R and """
        return self.R0(pixel) * self.waveR0

    def mkid_resolution_width(self, wave, pixel):
        """ Returns the wavelength width of the mkid at wave, pixel
        last axis should be npixel
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
        return wave ** 2 / rc  # (nord, npixel) actual mkid sigma at each pixel/order center

    def observe(self, arrival_times, arrival_wavelengths, resid_map=None):

        from mkidcore.binfile.mkidbin import PhotonNumpyType

        pixel_count = np.array([x.size for x in arrival_times])
        total_photons=pixel_count.sum()

        print(f'Simulated dataset may take up to {total_photons * 16 / 1024 ** 3:.2} GB of RAM')

        merge_time_window_s = 1e-6*u.s
        MIN_TRIGGER_ENERGY = 1 / (1.5 * u.um)
        SATURATION_WAVELENGTH_NM = 350*u.nm
        DEADTIME = 10 * u.us

        if resid_map is None:
            resid_map = np.arange(pixel_count.size, dtype=int)*10 + 100  #something arbitrary

        photons = np.recarray(total_photons, dtype=PhotonNumpyType)
        photons[:] = 0
        photons.weight[:] = 1.0
        observed = 0
        total_merged = 0
        total_missed = []
        # Compute photon arrival times and wavelengths for each photon
        for pixel, n in np.ndenumerate(pixel_count):
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
            will_trigger = measured_energies.value > MIN_TRIGGER_ENERGY
            if not will_trigger.any():
                continue

            # Drop photons that arrive within the deadtime
            detected = mask_deadtime(a_times[will_trigger], DEADTIME.to(u.s).value)

            missed = will_trigger.sum() - detected.sum()
            total_missed.append(missed)

            a_times = a_times[will_trigger][detected]
            measured_wavelengths = 1000 / measured_energies[will_trigger][detected]
            measured_wavelengths.clip(SATURATION_WAVELENGTH_NM, out=measured_wavelengths)

            # Add photons to the pot
            sl = slice(observed, observed + a_times.size)
            photons.wavelength[sl] = measured_wavelengths
            photons.time[sl] = (a_times * 1e6)  # in microseconds
            photons.resID[sl] = resid_map[pixel[::-1]]
            observed += a_times.size
        print(f'A total of {total_merged} photons had their energies '
              f'merged and {np.sum(total_missed)} were missed due to deadtime, {observed} observed.')
        return photons, observed
