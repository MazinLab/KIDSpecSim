import numpy as np
import astropy.units as u
from filterphot import mask_deadtime


class MKIDDetector:
    def __init__(
            self,
            n: int = 2048,
            pixel_size: u.Quantity = 20 * u.micron,
            R0: float = 15,
            l0: u.Quantity = 800 * u.nm,
            R0_type: str = 'from_file'
    ):
        """
        Simulation of an MKID detector array.

        :param n: number of pixels in linear array
        :param pixel_size: physical size of each pixel in astropy units
        :param R0: spectral resolution of the longest wavelength in spectrometer range
        :param l0: longest wavelength in spectrometer range in astropy units
        :param R0_type: where to obtain array of R0s for each pixel, can be:
            'from_file' - load a file containing random R0s named 'generated_R0s.csv'
            'fixed' - keep all R0s for all pixels fixed to the supplied R0
            'generate' - generate a new list of randomized R0s about the given R0
        :return: MKIDDetector class object
        """
        self.n_pixels = n
        self.pixel_size = pixel_size
        self.length = self.n_pixels * pixel_size
        self.waveR0 = l0
        self.design_R0 = R0
        self.pixel_indices = np.arange(self.n_pixels, dtype=int)
        self.R0_type = R0_type
        self.R0s = None

    def R0(self, pixel: int):
        """
        :param pixel: the pixel index or indices
        :return: spectral resolution for given pixel
        """
        if pixel not in self.pixel_indices:
            raise ValueError(f"Pixel {pixel+1} not in instantiated detector, max of {self.n_pixels}.")
        if self.R0_type == 'from_file':
            with open('generated_R0s.csv') as f:
                self.R0s = np.loadtxt(f, delimiter=",")
        elif self.R0_type == 'fixed':
            self.R0s = np.full(self.n_pixels, self.design_R0)
        elif self.R0_type == 'generate':
            self.R0s = np.random.uniform(.85, 1.15, size=self.n_pixels) * self.design_R0
            np.savetxt('generated_R0s.csv', self.R0s, delimiter=',')
        return self.R0s[pixel.astype(int)]

    def mkid_constant(self, pixel: int):
        """
        :param pixel: the pixel index or indices
        :return: MKID constant for given pixel, R0*l0
        """
        return self.R0(pixel) * self.waveR0

    def mkid_resolution_width(self, wave, pixel: int):
        """
        :param wave: wavelength(s) as float, int, or u.Quantity
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

    def observe(self,
                arrival_times: u.Quantity,
                arrival_wavelengths: u.Quantity,
                resid_map=None
                ):
        """
        :param arrival_times: timestamps of photons in astropy units
        :param arrival_wavelengths: wavelengths of photons in astropy units
        :param resid_map: IDs for each pixel (resonator)
        :return: recarray of observed photons, total number observed
        """
        from mkidcore.binfile.mkidbin import PhotonNumpyType
        print("\nBeginning detector observation sequence.")
        pixel_count = np.array([x.size for x in arrival_times])
        total_photons = pixel_count.sum()

        print(f'\tWARNING: Simulated dataset may take up to {total_photons * 16 / 1024 ** 3:.2} GB of RAM.')

        merge_time_window_s = 1e-6 * u.s
        MIN_TRIGGER_ENERGY = 1 / (1.5 * u.um)
        SATURATION_WAVELENGTH_NM = 350 * u.nm
        DEADTIME = 10 * u.us

        if resid_map is None:
            resid_map = np.arange(pixel_count.size, dtype=int) * 10 + 100  # something arbitrary

        photons = np.recarray(total_photons, dtype=PhotonNumpyType)
        photons[:] = 0
        photons.weight[:] = 1.0
        observed = 0
        total_merged = 0
        total_missed = []

        print(f"\tComputing detected arrival times and wavelengths for individual photons."
              f"\n\tMinimum trigger energy: {MIN_TRIGGER_ENERGY:.3e}"
              f"\n\tPhoton merge time: {merge_time_window_s:.0e}"
              f"\n\tSaturation wavelength: {SATURATION_WAVELENGTH_NM}"
              f"\n\tDeadtime: {DEADTIME}")
        for pixel, n in enumerate(pixel_count):
            if not n:
                continue

            # get photons and arrival times for pixel
            a_times = arrival_times[pixel]
            arrival_order = a_times.argsort()
            a_times = a_times[arrival_order]
            energies = 1 / arrival_wavelengths[pixel].to(u.um)[arrival_order]

            if self.R0_type != 'generate':
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
            photons.resID[sl] = resid_map[pixel]
            observed += a_times.size
        print(f'Completed detector observation sequence. {total_merged} photons had their energies merged, '
              f'{np.sum(total_missed)} photons were missed due to deadtime, and {observed} photons were observed.')
        return photons, observed
