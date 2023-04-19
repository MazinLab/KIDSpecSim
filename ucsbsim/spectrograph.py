from detector import MKIDDetector
import numpy as np
import astropy.units as u


class GratingSetup:
    def __init__(self, l0=800 * u.nm, m0=5, pixel_size=20 * u.micron, npix=2048, focal_length=350 * u.mm, littrow=True):
        """
        alpha = incidence angle
        d = length/groove in units of wavelength (same as schroeder sigma)
        delta = blaze angle
        theta = off blaze angle
        """
        self.l0_center = l0 / (1 + 1 / (2 * m0))
        self.angular_disp = 2 * np.arctan(pixel_size * npix / 2 / focal_length) / (self.l0_center / m0)
        self.alpha = np.arctan((self.l0_center / 2 * self.angular_disp).value) * u.rad
        self.d = m0 * self.l0_center / 2 / np.sin(self.alpha)
        if littrow:
            self.delta = self.alpha
            self.beta_central_pixel = self.alpha
        else:
            self.delta = None
            self.beta_central_pixel = None
        print(f"\nConfigured the grating.\n\tBlaze angle: {self.delta:.3e}"
              f"\n\tGroove length: {self.d:.3}")

    def blaze(self, beta, m):
        """
        Follows Casini & Nelson 2014 J Opt Soc Am A eq 18 with notation modified to match Schroder
        beta = reflectance
        m = order
        """
        k = np.cos(beta) * np.cos(self.alpha - self.delta) / (np.cos(self.alpha) * np.cos(beta - self.delta))
        k[k > 1] = 1  # k must be the minimum value between k and 1
        q4 = np.cos(self.delta) - np.sin(self.delta) / np.tan((self.alpha + beta) / 2)
        if self.alpha < self.delta:
            rho = np.cos(self.delta)  # 2 different rho depending on whether alpha or delta is larger
        else:
            rho = np.cos(self.alpha) / np.cos(self.alpha - self.delta)
        print(f"\nCalculated relative transmission (blaze efficiencies).")
        return k * np.sinc((m * rho * q4).value) ** 2  # omit np.pi as np.sinc includes it

    def beta(self, wave, m):
        """
        wave = wavelengths
        m = order
        """
        return np.arcsin(m * wave / self.d - np.sin(self.alpha))  # rad

    def wave(self, beta, m):
        """
        beta = diffracted angle
        m = order
        :return: the wavelength of beta in that order
        """
        return self.d * (np.sin(beta) + np.sin(self.alpha)) / m

    def resolution(self, entrance_beam_size, order):
        """The limited resolution of the grating config
        entrance_beam_size must be in units of wavelength
        """
        return order * entrance_beam_size / (self.d * np.cos(self.alpha))

    def resolution_eff(self, entrance_beam_size, order, wave, phi, tele_d):
        """The limited resolution of the grating config
        entrance_beam_size must be in units of wavelength

        phi - angular slit width (small angle approx: width/tele_f_len
        tele_d - tele diameter
        """
        return self.resolution(entrance_beam_size, order) * wave / (phi * tele_d)

    def angular_dispersion(self, m, beta):
        """Schroder A dbeta/dlambda"""
        return m / (self.d * np.cos(beta)) * u.rad  # rad per wavelength unit


class SpectrographSetup:
    def __init__(self, grating: GratingSetup, detector: MKIDDetector, min_order=5, max_order=9,
                 final_wave=800*u.nm, pixels_per_res_elem=2.5, focal_length=350 * u.mm, littrow=True):
        """focal length must be in units of wavelength"""
        self.m0 = min_order
        self.l0 = final_wave
        self.m_max = max_order
        self.grating = grating
        self.detector = detector
        self.focal_length = focal_length
        self.pixel_scale = np.arctan(self.detector.pixel_size / self.focal_length)  # angle in rad
        if littrow:
            self.beta_central_pixel = grating.beta_central_pixel
        else:
            self.beta_central_pixel = None
        self.orders = np.arange(min_order, max_order + 1, dtype=int)
        self.nominal_pixels_per_res_elem = pixels_per_res_elem
        self.minimum_wave = self.central_wave(max_order) - self.fsr(max_order) / 2

        # We assume that the system is designed to place pixels_per_res_elem pixels across the width of the dispersed
        # slit image at some fiducial wavelength and that the resolution element has some known width there
        # (and that the slit image is a gaussian)
        self.nondimensional_lsf_width = 1 / self.design_res
        print(f"\nConfigured the spectrograph.\n\tNo. of pixels per resolution element: "
              f"{self.nominal_pixels_per_res_elem}\n\tFocal length: {self.focal_length}")

    def pixel_for_beta(self, beta):
        """
        :param beta: radians
        :return: pixel at beta
        """
        delta_angle = np.tan(beta - self.beta_central_pixel)
        return self.focal_length * delta_angle / self.detector.pixel_size + self.detector.n_pixels / 2

    def beta_for_pixel(self, pixel):
        """
        :param pixel:
        :return: beta (radians)
        """
        center_offset = self.detector.pixel_size * (pixel - self.detector.n_pixels / 2)
        return self.beta_central_pixel + np.arctan(center_offset / self.focal_length)

    def blaze(self, wave):
        return self.grating.blaze(self.grating.beta(wave, self.orders[:, None]), self.orders[:, None])

    def order_mask(self, wave, fsr_edge=False):
        """
        Return a boolean array [norders, wave.size] where true corresponds to the wavelengths in the order
        If fsr_edge is true mask at the FSR limit otherwise mask at the detector edge
        """
        if fsr_edge:
            o = self.orders[:, None]
            return np.abs(wave - self.central_wave(o)) < self.fsr(o)
        else:
            x = self.wavelength_to_pixel(wave, self.orders[:, None])
            mask = (x >= 0) & (x < self.detector.n_pixels)
            return mask

    def central_wave(self, order):
        l0_center = self.l0 / (1 + 1 / (2 * self.m0))
        return l0_center * self.m0 / order

    def fsr(self, order):
        return self.central_wave(order) / order

    def wavelength_to_pixel(self, wave, m):
        return self.pixel_for_beta(self.grating.beta(wave, m))

    def pixel_to_wavelength(self, pixel, m):
        return self.grating.wave(self.beta_for_pixel(pixel), m)

    def pixel_center_wavelengths(self):
        """
        :return:  an (nord, npixel) array of pixel center (e.g. pixel 0.5, 1.5,...) wavelengths.

        note that wavelengths will be computed outside each order's FSR

        NB this is approximately equal to the simple approximation:
        (np.linspace(-.5, .5, num=detector.n_pixels) * self.fsr(m0) + self.central_wave(m0)) * (m0/self.orders)[:,None]


        """
        return self.pixel_to_wavelength(self.detector.pixel_indices + .5, self.orders[:, None])
        # x = self.detector.pixel_indices+.5
        # return np.array([self.pixel_to_wavelength(x, o) for o in self.orders])

    def angular_dispersion(self):
        """
        :return: The angular dispersion at the center of each pixel for each order (nord, npixel)
        """
        beta = self.beta_for_pixel(self.detector.pixel_indices + .5)
        return self.grating.angular_dispersion(self.orders[:, None], beta)

    @property
    def design_res(self):
        """ Assume that m0 FSR fills detector with some sampling """
        dlambda = self.fsr(self.m0) / self.detector.n_pixels * self.nominal_pixels_per_res_elem
        return self.l0 / dlambda
