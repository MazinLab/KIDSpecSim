import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u


class SpecSimSettings:
    def __init__(self,
                 R0s_file: str,
                 designR0: float,
                 simpconvol: bool,
                 nolittrow: bool,
                 l0_nm: float,
                 m0: int,
                 m_max: int,
                 minwave_nm: float,
                 maxwave_nm: float,
                 npix: int,
                 pixellim: int,
                 exptime_s: float,
                 telearea_cm2: float,
                 pixelsize_um: float,
                 focallength_mm: float,
                 pixels_per_res_elem: float,
                 temp_K: float,
                 type_spectra: str
                 ):
        """
        :param str R0s_file: Directory/filename of the R0s file.
        :param float designR0: The spectral resolution at the longest wavelength of m0.
        :param bool simpconvol: True if conducting a simplified convolution with MKIDs.
        :param bool nolittrow: True if Littrow configuration is NOT assumed.
        :param float l0_nm: The longest wavelength of any order in use in nm.
        :param int m0: The initial order, at the longer wavelength end.
        :param int m_max: The final order, at the shorter wavelength end.
        :param float minwave_nm: The minimum wavelength of the spectrometer in nm.
        :param float maxwave_nm: The maximum wavelength of the spectrometer in nm.
        :param int npix: The number of pixels in the MKID detector.
        :param int pixellim: Limit to the # of photons per pixel.
        :param float exptime_s: The exposure time of the observation in seconds.
        :param float telearea_cm2: The telescope area of the observation in cm2.
        :param float pixelsize_um: The length of the MKID pixel in the dispersion direction in um.
        :param float focallength_mm: The focal length of the detector in mm.
        :param float pixels_per_res_elem: Number of pixels per spectral resolution element for the spectrograph.
        :param float temp_K: The temperature of the spectrum in K.
        :param str type_spectra: The type of spectrum to be simulated.
        """
        self.R0s_file = R0s_file
        self.designR0 = float(designR0)
        self.simpconvol = simpconvol
        self.nolittrow = nolittrow
        self.l0 = float(l0_nm)*u.nm if not isinstance(l0_nm, u.Quantity) else l0_nm
        self.m0 = int(m0)
        self.m_max = int(m_max)
        self.minwave = float(minwave_nm)*u.nm if not isinstance(minwave_nm, u.Quantity) else minwave_nm
        self.maxwave = float(maxwave_nm)*u.nm if not isinstance(maxwave_nm, u.Quantity) else maxwave_nm
        self.npix = int(npix)
        self.pixellim = int(pixellim)
        self.exptime = float(exptime_s)*u.s if not isinstance(exptime_s, u.Quantity) else exptime_s
        self.telearea = float(telearea_cm2)*u.cm**2 if not isinstance(telearea_cm2, u.Quantity) else telearea_cm2
        self.pixelsize = float(pixelsize_um)*u.micron if not isinstance(pixelsize_um, u.Quantity) else pixelsize_um
        self.focallength = float(focallength_mm)*u.mm if not isinstance(focallength_mm, u.Quantity) else focallength_mm
        self.pixels_per_res_elem = float(pixels_per_res_elem)
        self.temp = float(temp_K)
        self.type_spectra = type_spectra

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False
        else:
            return self.__dict__ == other.__dict__
