import synphot
import specutils
import numpy as np
from astropy.io import fits

from astropy import units as u
from specutils import Spectrum1D
from synphot import SpectralElement, SourceSpectrum
from synphot.models import Box1D, BlackBodyNorm1D

_atm = None


# todo make use of Specutils/and pysynphot.
# https://synphot.readthedocs.io/en/latest/synphot/spectrum.html#specutils


def _get_atm():
    """
    :return: atmospheric transmission as function of wavelength
    """
    global _atm
    if _atm is not None:
        return _atm
    x = np.genfromtxt('transdata_0.5_1_mic')
    y = np.genfromtxt('transdata_1_5_mic')
    x = x[x[:, 1] > 0]
    x[:, 0] = 1e4 / x[:, 0]
    y = y[y[:, 1] > 0]
    y[:, 0] = 1e4 / y[:, 0]
    x = x[::-1]
    y = y[::-1]
    trans = np.vstack((x[x[:, 0] < 1.111], y[y[:, 0] >= 1.111]))
    atmosphere = trans[(trans[:, 0] > .5) & (trans[:, 0] < 1.4)]
    _atm = atmosphere[:, 0] * u.micron, atmosphere[:, 1] * u.dimensionless_unscaled
    return _atm


def AtmosphericTransmission():
    """
    :return: atmospheric transmission as SpectralElement object
    """
    w, t = _get_atm()
    spec = Spectrum1D(spectral_axis=w, flux=t)
    print("\nObtained atmospheric transmission bandpass.")
    return SpectralElement.from_spectrum1d(spec)


def TelescopeTransmission(reflectivity: float = .9):
    """
    :param reflectivity: of the telescope as 1-fraction, (1 means no reflection)
    :return: transmission due to telescope reflectivity, smaller at higher wavelengths, as SpectralElement object
    """
    w = np.linspace(300, 1500, 10000) * u.nm
    t = np.linspace(1, .95, 10000) * reflectivity * u.dimensionless_unscaled
    spec = Spectrum1D(spectral_axis=w, flux=t)
    print(f"Obtained telescope bandpass with reflectivity {reflectivity}.")
    return SpectralElement.from_spectrum1d(spec)


def FilterTransmission(min=400*u.nm, max=800*u.nm):
    """
    :param min: shorter wavelength edge
    :param max: longer wavelength edge
    :return: transmission of 1 between min and max as SpectralElement object
    """
    wid = max - min
    center = (max + min) / 2
    print(f"Obtained {min} to {max} filter bandpass.")
    return SpectralElement(Box1D, amplitude=1, x_0=center, width=wid)


def apply_bandpass(spectrum, cal=False, **kwargs):
    """
    :param spectrum: spectrum to apply bandpasses to
    :param cal: if spectrum is for calibration, no bandpasses will be applied
    :param kwargs: teff, minwave, maxwave, feh, logg, etc.
    :return: original spectrum multiplied with bandpasses
    """
    if cal:
        w = np.linspace(300, 1000, 1400000) * u.nm
        t = np.ones(1400000) * u.dimensionless_unscaled
        ones = Spectrum1D(spectral_axis=w, flux=t)
        spectrum[0] *= SpectralElement.from_spectrum1d(ones)
        return clip_spectrum(spectrum[0], **kwargs)
    else:
        bandpasses = [AtmosphericTransmission(), TelescopeTransmission(),
                      FilterTransmission(**kwargs)]
        for i, s in enumerate(spectrum):
            for b in bandpasses:
                s *= b
            spectrum[i] = s
        return clip_spectrum(spectrum[0], **kwargs)


def PhoenixModel(teff: float, feh=0, logg=4.8, desired_magnitude=None):
    """
    :param float teff: effective temperature of star
    :param feh: distance as z value
    :param logg: log of surface gravity
    :param desired_magnitude: magnitude with which to normalize model spectrum, optional
    :return: Phoenix model of star with given properties as SourceSpectrum object
    """
    from expecto import get_spectrum
    sp = SourceSpectrum.from_spectrum1d(get_spectrum(T_eff=teff, log_g=logg, Z=feh, cache=True))
    if desired_magnitude is not None:
        sp.normalize(desired_magnitude, band=S.ObsBandpass('johnson,v'))
    return sp


def BlackbodyModel(teff: float):
    """
    :param float teff: effective temperature of model star
    :return: blackbody model of star as SourceSpectrum object
    """
    return SourceSpectrum(BlackBodyNorm1D, temperature=teff)


def DeltaModel(minwave=400*u.nm, maxwave=800*u.nm):
    """
    :param minwave: minimum wavelength of detector
    :param maxwave: maximum wavelength of detector
    :return: 'delta'-like model at the central wavelength
    """
    x_0 = (maxwave + minwave) / 2
    width = 10 * u.nm
    return SourceSpectrum(Box1D, amplitude=1e-15, x_0=x_0, width=width)


def get_spectrum(spectrum_type: str, **kwargs):
    """
    :param str spectrum_type: 'blackbody', 'phoenix', or 'delta' only
    :param kwargs: teff, feh, logg, minwave, maxwave, etc.
    :return: SourceSpectrum object of chosen spectrum
    """
    if spectrum_type == 'blackbody':
        return BlackbodyModel(**kwargs)
    elif spectrum_type == 'phoenix':
        return PhoenixModel(**kwargs)
    elif spectrum_type == 'delta':
        return DeltaModel(**kwargs)
    else:
        raise ValueError("Only 'blackbody', 'phoenix', or 'delta' are supported for spectrum_type.")


def clip_spectrum(x, minw, maxw):
    """
    :param x: SourceSpectrum object containing desired spectrum
    :param minw: shorter wavelength edge
    :param maxw: longer wavelength edge
    :return: clipped out SourceSpectrum instead of setting fluxden to 0, different from FilterTransmission
    """
    mask = (x.waveset >= minw) & (x.waveset <= maxw)
    w = x.waveset[mask]
    print(f"Clipped spectrum from {minw} to {maxw}.")
    return SourceSpectrum.from_spectrum1d(Spectrum1D(spectral_axis=w, flux=x(w)))
