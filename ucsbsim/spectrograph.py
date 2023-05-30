from ucsbsim.detector import MKIDDetector
import numpy as np
import astropy.units as u


class GratingSetup:
    def __init__(self, alpha, delta, groove_length):
        """
        alpha = incidence angle
        d = length/groove in units of wavelength (same as schroeder sigma)
        delta = blaze angle
        theta = off blaze angle
        """
        self.alpha = alpha
        self.delta = delta
        self.d = groove_length
        self.empiric_blaze_factor = 1.0

    def __str__(self):
        return (f"a={np.rad2deg(self.alpha):.2f}\n"
                f"d={np.rad2deg(self.delta):.2f}\n"
                f"l={self.d.to('mm'):.2f}/l ({1/self.d.to('mm'):.2f})")

    def blaze(self, beta, m):
        """
        Follows Casini & Nelson 2014 J Opt Soc Am A eq 25 (based on 18) with notation modified to match Schroder
        beta = reflectance
        m = order
        """
        k = np.cos(beta) * np.cos(self.alpha - self.delta) / (np.cos(self.alpha) * np.cos(beta - self.delta))
        k[k > 1] = 1
        # q1 = np.cos(alpha) / np.cos(alpha - delta)
        # q3=np.cos(delta)-np.sin(delta)*np.cot((alpha+beta)/2)
        # return k*np.sinc(m*q1*q3)**2
        q4 = np.cos(self.delta) - np.sin(self.delta) / np.tan((self.alpha + beta) / 2)
        rho = np.cos(self.delta) if self.alpha < self.delta else np.cos(self.alpha) / np.cos(self.alpha - self.delta)
        ret = k * np.sinc((m * rho * q4).value) ** 2  # omit np.pi as np.sinc includes it
        return self.empiric_blaze_factor*ret

    def beta(self, wave, m):
        """
        wave = wavelengths
        m = order
        """
        return np.arcsin(m * wave / self.d - np.sin(self.alpha))

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
        return m / (self.d * np.cos(beta)) * u.rad


class SpectrographSetup:
    def __init__(self, min_order, max_order, final_wave, pixels_per_res_elem,
                 focal_length, beta_central_pixel,
                 grating: GratingSetup, detector: MKIDDetector):
        """focal length must be in units of wavelength"""
        self.m0 = min_order
        self.l0 = final_wave
        self.m_max = max_order
        self.grating = grating
        self.detector = detector
        self.focal_length = focal_length
        self.pixel_scale = np.arctan(self.detector.pixel_size / self.focal_length)
        self.beta_central_pixel = beta_central_pixel
        self.nominal_pixels_per_res_elem = pixels_per_res_elem
        # We assume that the system is designed to place pixels_per_res_elem pixels across the width of the dispersed
        # slit image at some fiducial wavelength and that the resolution element has some known width there
        # (and that the slit image is a gaussian)
        self.nondimensional_lsf_width = 1 / self.design_res

    def set_beta_center(self, beta):
        if not isinstance(beta, u.Quantity):
            beta *= u.deg
        self.beta_central_pixel = beta
        self.grating.alpha = beta #Enforce littrow

    @property
    def orders(self):
        try:
            assert self._orders[0] == (self.m0, self.m_max)
        except (AttributeError, AssertionError):
            self._orders = (self.m0, self.m_max), np.arange(self.m0, self.m_max + 1, dtype=int)
        return self._orders[1]

    @property
    def minimum_wave(self):
        return self.central_wave(self.m_max) - self.fsr(self.m_max) / 2

    def info_str(self):
        gstr=str(self.grating)
        betactr = np.rad2deg(self.grating.beta(self.central_wave(self.m0), self.m0))
        gstr+=f'\nb={betactr:.2f}'
        ret = [f'    {x}' for x in gstr.split('\n')]
        ret.insert(0, 'Grating:')
        beta_ext = np.rad2deg(self.beta_for_pixel(self.detector.pixel_indices[[0, -1]]+.5))
        ret.append("    beta extent: {:.2f} - {:.2f}".format(*beta_ext))
        ret.append('Orders:')
        for o in self.orders[::-1]:
            w_c = self.central_wave(o)
            w_i = w_c - self.fsr(o)/2
            w_f = w_c + self.fsr(o)/2
            p_i = self.wavelength_to_pixel(w_i, o)
            p_f = self.wavelength_to_pixel(w_f, o)
            ret.append(f"    m{o:2} @ {w_c:.0f}: {w_i:.0f} - {w_f:.0f}, {p_i:.0f} - {p_f:.0f}")
        return ret

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

    def mean_blaze_eff_est(self, n=10):
        edges = self.edge_wave(fsr=True)
        detector_edges = self.edge_wave(fsr=False)

        ow = np.array([np.select([detector_edges > edges, detector_edges <= edges],
                                 [detector_edges, edges])[:, 0],
                       np.select([detector_edges < edges, detector_edges > edges],
                                 [detector_edges, edges])[:, 1]]).T * u.nm
        v = self.blaze(np.array(list(map(lambda x: np.linspace(*x, num=n), ow))) * u.nm).mean(1)
        return v.value

    def order_mask(self, wave, fsr_edge=False):
        """
        Return a boolean array [norders, wave.size] where true corresponds to the wavelengths in the order
        If fsr_edge is true mask at the FSR limit otherwise mask at the detector edge
        """
        if fsr_edge:
            o = self.orders[:, None]
            c_wave = self.pixel_to_wavelength(self.detector.n_pixels/2, o)
            fsr = c_wave/o
            return np.abs(wave - c_wave) < fsr/2
        else:
            x = self.wavelength_to_pixel(wave, self.orders[:, None])
            return (x >= 0) & (x < self.detector.n_pixels)

    def edge_wave(self, fsr=True):
        pix = self.detector.pixel_indices[[0, self.detector.n_pixels // 2, -1]]+.5
        fiducial_waves = self.pixel_to_wavelength(pix, self.orders[:, None])
        if not fsr:
            return fiducial_waves[:, [0, -1]]

        central_fsr = fiducial_waves[:, 1] / self.orders
        fsr_edges = (u.Quantity([-central_fsr / 2, central_fsr / 2]) + fiducial_waves[:, 1]).T

        return fsr_edges

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

    @property
    def average_res(self):
        w = self.edge_wave(fsr=False)
        return (w.mean(1)/(np.diff(w, axis=1).T/self.detector.n_pixels * self.nominal_pixels_per_res_elem)).ravel()

    def plot_echellogram(spec, center_orders=True, title='', blaze=False):
        import matplotlib.pyplot as plt
        w = spec.pixel_center_wavelengths()
        b = spec.blaze(w)

        fig, axes = plt.subplots(2+int(blaze), 1, figsize=(6, 10+4*int(blaze)))
        if title:
            plt.suptitle(title)
        plt.sca(axes[0])
        plt.title(f'a={spec.beta_central_pixel:.1f} m={spec.m0}-{spec.m_max}')
        fsr_edges = spec.edge_wave(fsr=True)
        for ii, i in enumerate(spec.orders):
            waves = w[ii, [0, spec.detector.n_pixels // 2, -1]]
            plt.plot(spec.wavelength_to_pixel(waves, i), [i] * 3, '*', color=f'C{ii}')
            plt.plot(spec.wavelength_to_pixel(fsr_edges[ii], i), [i] * 2, '.', color=f'C{ii}')
        plt.xlabel('Pixel')
        plt.ylabel('Order')
        plt.sca(axes[1])
        for ii, i in enumerate(spec.orders):
            waves = w[ii, [0, spec.detector.n_pixels // 2, -1]]
            oset = waves[1] if center_orders else 0
            plt.plot(waves - oset, [i] * 3, '*', color=f'C{ii}')
            plt.plot(fsr_edges[ii] - oset, [i] * 2, '.', color=f'C{ii}',
                     label=f'$\lambda=${waves[1]:.0f}')
        plt.legend()
        plt.xlabel('Center relative wavelength (nm)')
        plt.ylabel('Order')
        if blaze:
            plt.sca(axes[2])
            plt.plot(w.T, b.T)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Blaze')
        plt.tight_layout()
        plt.show()



GRATING_CATALOG = np.loadtxt('newport_masters.txt', delimiter=',',
                             dtype=[('name', 'U10'), ('l', 'f4'), ('blaze', 'f4'),
                                    ('width', 'f4'), ('height', 'f4'), ('stock', 'U10')])
GRATING_CATALOG['l'] = 1e6/GRATING_CATALOG['l']

NEWPORT_GRATINGS = {x['name']: GratingSetup(None, x['blaze']*u.deg, x['l']* u.nm) for x in GRATING_CATALOG}
