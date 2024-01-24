import synphot
import specutils
import numpy as np
from astropy.io import fits
import logging
import pandas as pd

from astropy import units as u
from specutils import Spectrum1D
from synphot import SpectralElement, SourceSpectrum
from synphot.spectrum import BaseSpectrum
from synphot.models import Box1D, BlackBodyNorm1D, ConstFlux1D
from .engine import gauss

_atm = None

u.photlam = u.photon / u.s / u.cm ** 2 / u.AA  # new unit name, photon flux per wavelength


# todo make use of Specutils/and pysynphot.
# https://synphot.readthedocs.io/en/latest/synphot/spectrum.html#specutils


def _get_atm():
    """
    :return: atmospheric transmission as function of wavelength
    """
    global _atm
    if _atm is not None:
        return _atm
    x = np.genfromtxt('transdata_0.5_1_mic')  # TODO doesn't go below 500nm
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
    return SpectralElement.from_spectrum1d(spec)


def TelescopeTransmission(reflectivity: float = .9):
    """
    :param reflectivity: of the telescope as 1-fraction, (1 means no reflection)
    :return: transmission due to telescope reflectivity, smaller at higher wavelengths, as SpectralElement object
    """
    w = np.linspace(300, 1500, 10000) * u.nm
    t = np.linspace(1, .95, 10000) * reflectivity * u.dimensionless_unscaled
    spec = Spectrum1D(spectral_axis=w, flux=t)
    return SpectralElement.from_spectrum1d(spec)


def FilterTransmission(min=400 * u.nm, max=800 * u.nm):
    """
    :param min: shorter wavelength edge
    :param max: longer wavelength edge
    :return: transmission of 1 between min and max as SpectralElement object
    """
    wid = max - min
    center = (max + min) / 2
    return SpectralElement(Box1D, amplitude=1, x_0=center, width=wid)


def FineGrid(min, max):
    w = np.linspace(min.to(u.nm).value-100, max.to(u.nm).value+100, 100000) * u.nm
    t = np.ones(100000) * u.dimensionless_unscaled
    ones = Spectrum1D(spectral_axis=w, flux=t)
    return SpectralElement.from_spectrum1d(ones)


def apply_bandpass(spectra, bandpass):
    """
    :param spectra: spectra to apply bandpasses to, as list or object
    :param bandpass: the filter to be applied, as list or object
    :return: original spectrum multiplied with bandpasses
    """
    if not isinstance(spectra, list):
        spectra = [spectra]
        not_list = True
    if not isinstance(bandpass, list):
        bandpass = [bandpass]
    for i, s in enumerate(spectra):
        for b in bandpass:
            s *= b
        spectra[i] = s
    logging.info(f'Multipled spectrum with given bandpass.')
    if not_list:
        return spectra[0]
    else:
        return spectra


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
    return SourceSpectrum(BlackBodyNorm1D, temperature=teff)*1e16


def FlatModel():
    """
    :return: model which returns the same flux at all wavelengths
    """
    return SourceSpectrum(ConstFlux1D, amplitude=1e5)


def EmissionModel(filename, minwave, maxwave, target_R=50000):
    """
    :param filename: file name of the emission line list, with wavelength in nm, FROM NIST
    :param minwave: the min wave of the desired model, in nm or as u.Quantity
    :param maxwave: the max wave of the desired model, in nm or as u.Quantity
    :param target_R: spectral resolution to diffraction limit line spectrum
    :return: full emission spectrum, intensity converted to photlam
    """
    file = pd.read_csv(filename, delimiter=',')
    flux = np.array(file['intens'])
    wave = np.array(file['obs_wl_air(nm)'])
    try:
        uncert = np.array(file['unc_obs_wl'])
    except KeyError:
        uncert = np.full(wave.shape, 0.0010)
    if isinstance(flux[0], str):
        include = np.full(len(flux), False)
        for n, i in enumerate(flux):
            try:
                flux[n] = float(i[2:-1])
                wave[n] = float(wave[n][2:-1])
                if isinstance(uncert[n], str):
                    uncert[n] = float(uncert[n][2:-1])
                include[n] = True
            except ValueError:
                include[n] = False
        flux = flux[include]
        wave = wave[include]
        uncert = uncert[include]
    target_dl = (wave[0] + wave[-1]) / 2 / target_R
    sigma_factor = target_dl / min(uncert)  # 3 sigma approx to 1st Airy ring

    if isinstance(minwave, u.Quantity):
        minwave = minwave.to(u.nm).value
    if isinstance(maxwave, u.Quantity):
        maxwave = maxwave.to(u.nm).value
    wave_grid = np.arange(minwave, maxwave, target_dl)
    line_gauss = gauss(wave_grid[None, :].astype(float), wave[:, None].astype(float),
                       uncert[:, None].astype(float) * sigma_factor / 3, flux[:, None].astype(float))
    spectrum = np.sum(line_gauss, axis=1)
    return SourceSpectrum.from_spectrum1d(Spectrum1D(flux=spectrum * u.photlam, spectral_axis=wave_grid * u.nm))


def get_spectrum(spectrum_type: str, teff=None, emission_file=None, minwave=None, maxwave=None):
    """
    :param str spectrum_type: 'blackbody', 'phoenix', 'flat', or 'emission' only
    :param teff: effective temperature for blackbody or phoenix model spectrum
    :param emission_file: file name for the desired emission spectrum
    :param minwave: minimum wavelength
    :param maxwave: maximum wavelength
    :return: SourceSpectrum object of chosen spectrum
    """
    if spectrum_type == 'blackbody':
        logging.info(f'\nObtained blackbody model spectrum of {teff} K star.')
        return BlackbodyModel(teff)
    elif spectrum_type == 'phoenix':
        logging.info(f'\nObtained Phoenix model spectrum of {teff} K star.')
        return PhoenixModel(teff)
    elif spectrum_type == 'flat':
        logging.info(f'\nObtained flat flux model spectrum.')
        return FlatModel()
    elif spectrum_type == 'emission':
        logging.info(f'\nObtained {emission_file} emission spectrum.')
        return EmissionModel(emission_file, minwave, maxwave)
    else:
        raise ValueError("Only 'blackbody', 'phoenix', 'flat', or 'emission' are supported for spectrum_type.")


def clip_spectrum(x, clip_range):
    """
    :param x: SourceSpectrum object containing desired spectrum
    :param tuple clip_range: shorter/longer wavelength edge
    :return: clipped out SourceSpectrum instead of setting fluxden to 0, different from FilterTransmission
    """
    mask = (x.waveset >= clip_range[0]) & (x.waveset <= clip_range[-1])
    w = x.waveset[mask]
    logging.info(f"Clipped spectrum to{clip_range}.")
    return SourceSpectrum.from_spectrum1d(Spectrum1D(spectral_axis=w, flux=x(w)))
