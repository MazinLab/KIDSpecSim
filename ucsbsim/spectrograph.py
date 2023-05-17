from detector import MKIDDetector
import numpy as np
import astropy.units as u
import logging


class GratingSetup:
    def __init__(self,
                 l0: u.Quantity = 800 * u.nm,
                 m0: int = 5,
                 m_max: int = 9,
                 pixel_size: u.Quantity = 20 * u.micron,
                 npix: int = 2048,
                 focal_length: u.Quantity = 350 * u.mm,
                 littrow: bool = True
                 ):
        """
        Simulation of an echelle/echellete grating for spectrometer.

        :param l0: longest wavelength in bandpass in astropy units
        :param m0: starting (smallest) order
        :param m_max: ending (largest) order
        :param pixel_size: physical size of pixel in astropy units
        :param npix: number of pixels in MKID array
        :param focal_length: expected focal length of grating
        :param littrow: True if Littrow configuration where incident = reflected angle
        :return: GratingSetup class object
        """
        self.l0 = l0
        self.m0 = m0
        self.m_max = m_max
        self.focal_length = focal_length
        self.l0_center = l0 / (1 + 1 / (2 * m0))
        self.angular_disp = 2 * np.arctan(pixel_size * npix / 2 / focal_length) / (self.l0_center / m0)
        self.alpha = np.arctan((self.l0_center / 2 * self.angular_disp).value) * u.rad
        self.groove_length = m0 * self.l0_center / 2 / np.sin(self.alpha)
        self.littrow = littrow
        if littrow:
            self.delta = self.alpha
            self.beta_central_pixel = self.alpha
        else:
            self.delta = None  # need to add case of not littrow
            self.beta_central_pixel = None

    def blaze(self, beta: float, m: int):
        """
        Blaze throughput function follows Casini & Nelson 2014 J Opt Soc Am A eq 18 with notation modified to
        match Schroder.
        :param beta: reflectance angle
        :param m: order
        :return: grating throughput
        """
        k = np.cos(beta) * np.cos(self.alpha - self.delta) / (np.cos(self.alpha) * np.cos(beta - self.delta))
        k[k > 1] = 1  # k must be the minimum value between k and 1
        q4 = np.cos(self.delta) - np.sin(self.delta) / np.tan((self.alpha + beta) / 2)
        if self.alpha < self.delta:
            rho = np.cos(self.delta)  # 2 different rho depending on whether alpha or delta is larger
        else:
            rho = np.cos(self.alpha) / np.cos(self.alpha - self.delta)
        logging.info(f"\nCalculated relative transmission (grating efficiency).")
        return k * np.sinc((m * rho * q4).value) ** 2  # omit np.pi as np.sinc includes it

    def beta(self, wave, m: int):
        """
        :param wave: wavelength(s) as floats or u.Quantity
        :param m: order
        :return: reflectance angle
        """
        return np.arcsin(m * wave / self.groove_length - np.sin(self.alpha))  # rad

    def wave(self, beta: float, m: int):
        """
        :param beta: diffracted angle
        :param m: order
        :return: wavelength
        """
        return self.groove_length * (np.sin(beta) + np.sin(self.alpha)) / m

    def resolution(self, entrance_beam_size: u.Quantity, order: int):
        """
        :param entrance_beam_size: size of the incoming beam in astropy units of wavelength
        :param order: order
        :return: the limited resolution of the grating configuration
        """
        return order * entrance_beam_size / (self.groove_length * np.cos(self.alpha))

    def resolution_eff(self,
                       entrance_beam_size: u.Quantity,
                       order: int,
                       wave: u.Quantity,
                       phi: float,
                       tele_d: u.Quantity
                       ):
        """
        :param entrance_beam_size: size of the incoming beam in astropy units of wavelength
        :param order: order
        :param wave: wavelength in astropy units
        :param phi: angular slit width (small angle approx: width/tele_f_len)
        :param tele_d: telescope diameter in astropy units
        :return: the effective resolution of the grating configuration
        """
        return self.resolution(entrance_beam_size, order) * wave / (phi * tele_d)

    def angular_dispersion(self, m, beta):
        """
        :param m: order
        :param beta: reflectance angle
        :return: angular dispersion [rad/wavelength], Schroder A dbeta/dlambda
        """
        return m / (self.groove_length * np.cos(beta)) * u.rad


class SpectrographSetup:
    def __init__(self,
                 grating: GratingSetup,
                 detector: MKIDDetector,
                 pixels_per_res_elem: float = 2.5,
                 ):
        """
        Simulation of a spectrograph.

        :param grating: configured grating
        :param detector: configured detector
        :param pixels_per_res_elem: number of pixels per resolution element of spectrometer
        """
        self.m0 = grating.m0
        self.l0 = grating.l0
        self.m_max = grating.m_max
        self.grating = grating
        self.detector = detector
        self.focal_length = grating.focal_length
        self.pixel_scale = np.arctan(self.detector.pixel_size / grating.focal_length)  # angle in rad
        if grating.littrow:
            self.beta_central_pixel = grating.beta_central_pixel
        else:
            self.beta_central_pixel = None  # figure out nonLittrow cases
        self.orders = np.arange(self.m0, self.m_max + 1, dtype=int)
        self.nord = len(self.orders)
        self.nominal_pixels_per_res_elem = pixels_per_res_elem
        self.minimum_wave = self.central_wave(self.m_max) - self.fsr(self.m0) / 2

        # We assume that the system is designed to place pixels_per_res_elem pixels across the width of the dispersed
        # slit image at some fiducial wavelength and that the resolution element has some known width there
        # (and that the slit image is a gaussian)
        self.nondimensional_lsf_width = 1 / self.design_res
        logging.info(f'\nThe spectrograph has been setup with the following properties:'
                     f'\n\tl0: {self.l0}'
                     f'\n\tR0: {self.detector.design_R0}'
                     f'\n\tOrders: {self.orders}'
                     f'\n\tFocal length: {self.focal_length}'
                     f'\n\tIncident angle: {self.grating.alpha:.3f}'
                     f'\n\tGroove length: {self.grating.groove_length:.2f}'
                     f'\n\tLittrow config.: {self.grating.littrow}'
                     f'\n\t# of pixels: {self.detector.n_pixels}'
                     f'\n\tPixel size: {self.detector.pixel_size}'
                     f'\n\tPixels per res. element: {self.nominal_pixels_per_res_elem}')

    def pixel_for_beta(self, beta: float):
        """
        :param beta: reflectance angle in radians
        :return: pixel at beta
        """
        delta_angle = np.tan(beta - self.beta_central_pixel)
        return self.focal_length * delta_angle / self.detector.pixel_size + self.detector.n_pixels / 2

    def beta_for_pixel(self, pixel: int):
        """
        :param pixel: pixel
        :return: beta (radians) at pixel
        """
        center_offset = self.detector.pixel_size * (pixel - self.detector.n_pixels / 2)
        return self.beta_central_pixel + np.arctan(center_offset / self.focal_length)

    def max_beta_m0(self):
        """
        :return: largest reflectance angle (which is at the initial order)
        """
        return self.grating.beta(self.l0, self.m0)

    def min_beta_mmax(self):
        """
        :return: smallest reflectance angle (which is at the final order)
        """
        return self.grating.beta(self.minimum_wave, self.m_max)

    def blaze(self, wave):
        """
        :param wave: wavelength
        :return: blaze throughput
        """
        return self.grating.blaze(self.grating.beta(wave, self.orders[:, None]), self.orders[:, None])

    def order_mask(self, wave, fsr_edge: bool = False):
        """
        :param wave: wavelength
        :param fsr_edge: True to mask at the FSR, goes to detector edge if not
        :return: a boolean array [norders, wave.size] where true corresponds to the wavelengths in the order
        """
        if fsr_edge:
            o = self.orders[:, None]
            return np.abs(wave - self.central_wave(o)) < self.fsr(o)
        else:
            x = self.wavelength_to_pixel(wave, self.orders[:, None])
            mask = (x >= 0) & (x < self.detector.n_pixels)
            return mask

    def central_wave(self, order: int):
        """
        :param order: order
        :return: central wavelength for the entire order
        """
        return self.grating.l0_center * self.m0 / order

    def fsr(self, order: int):
        """
        :param order: order
        :return: free spectral range in units of wavelength
        """
        return self.central_wave(order) / order

    def wavelength_to_pixel(self, wave, m: int):
        """
        :param wave: wavelength
        :param m: order
        :return: pixel location of given wavelength
        """
        return self.pixel_for_beta(self.grating.beta(wave, m))

    def pixel_to_wavelength(self, pixel: int, m: int):
        """
        :param pixel: pixel
        :param m: order
        :return: wavelength for given pixel
        """
        return self.grating.wave(self.beta_for_pixel(pixel), m)

    def pixel_center_wavelengths(self, edge=None):
        """
        :param edge: left or right indicates the edges of the pixel instead of exactly at center
        :return:  array of pixel center (or left/right edge) wavelengths for spectrometer

        Note that wavelengths will be computed outside each order's FSR.
        NB this is approximately equal to the simple approximation:
        (np.linspace(-.5, .5, num=detector.n_pixels) * self.fsr(m0) + self.central_wave(m0)) * (m0/self.orders)[:,None]
        """
        if edge == 'left':
            return self.pixel_to_wavelength(self.detector.pixel_indices, self.orders[:, None])
        elif edge == 'right':
            return self.pixel_to_wavelength(self.detector.pixel_indices + 1, self.orders[:, None])
        else:
            return self.pixel_to_wavelength(self.detector.pixel_indices + .5, self.orders[:, None])

    def dl_pix_max_wave(self):
        """
        :return: maximum change in wavelength in any pixel
        """
        return self.pixel_scale / self.grating.angular_dispersion(self.m0, self.max_beta_m0())

    def dl_pix_min_wave(self):
        """
        :return: minimum change in wavelength in any pixel
        """
        return self.pixel_scale / self.grating.angular_dispersion(self.m_max, self.min_beta_mmax())

    def dl_mkid_max(self):
        """
        :return: largest MKID resolution width
        """
        return (self.l0 ** 2 / self.detector.mkid_constant(self.detector.pixel_indices)).max()

    def sampling(self, oversampling):
        """
        :param oversampling: factor by which to oversample smallest wavelength extent
        :return: size of sampling in wavelength units
        """
        return self.dl_pix_min_wave() / oversampling

    def dl_pixel(self):
        """
        :return: change in wavelength for every pixel
        """
        return self.pixel_scale / self.angular_dispersion()

    def dl_mkid_pixel(self):
        """
        :return: MKID resolution width for every pixel
        """
        return self.detector.mkid_resolution_width(self.pixel_center_wavelengths(), self.detector.pixel_indices)

    def pixel_rescale(self, oversampling):
        """
        :param oversampling: factor by which to oversample smallest wavelength extent
        :return: The sample size in wavelength units for every pixel. Every MKID resolution width divided by the
                 total # of samples that are in the largest width, which is the largest width divided by the
                 smallest sample size.
        """
        return self.dl_mkid_pixel() * self.sampling(oversampling) / self.dl_mkid_max()

    def pixel_samples_frac(self, oversampling):
        """
        :return: number of samples for every pixel, retrieved by dividing change in wavelength for a pixel
                 by the sample size for that pixel.
        """
        return (self.dl_pixel() / self.pixel_rescale(oversampling)).si.value

    def pixel_max_npoints(self, oversampling):
        """
        :return: the maximum number of points in any given pixel, as an integer value, ensuring there is atleast one
                 sample at pixel center
        """
        pixel_max_npoints = np.ceil(self.pixel_samples_frac(oversampling).max()).astype(int)
        if not pixel_max_npoints % 2:  # ensure there is a point at the pixel center
            pixel_max_npoints += 1
        return pixel_max_npoints

    def sigma_mkid_pixel(self):
        """
        :return: the standard deviation for each pixel resolution
        """
        return self.dl_mkid_pixel() / 2.355

    def angular_dispersion(self):
        """
        :return: angular dispersions of each pixel for each order
        """
        beta = self.beta_for_pixel(self.detector.pixel_indices + .5)
        return self.grating.angular_dispersion(self.orders[:, None], beta)

    @property
    def design_res(self):
        """
        :return: design resolution for spectrometer
        Assume that m0 FSR fills detector with some sampling
        """
        dlambda = self.fsr(self.m0) / self.detector.n_pixels * self.nominal_pixels_per_res_elem
        return self.l0 / dlambda
