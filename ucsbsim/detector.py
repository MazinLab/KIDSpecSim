import numpy as np


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
