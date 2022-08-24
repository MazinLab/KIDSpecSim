import numpy as np
import scipy.interpolate as interp
import scipy
import scipy.signal as sig
import scipy.ndimage as ndi
import numpy as np

from ucsbsim.spectra import PhoenixModel, AtmosphericTransmission, FilterTransmission, TelescopeTransmission
from ucsbsim.spectrograph import GratingSetup, SpectrographSetup
from ucsbsim.detector import MKIDDetector


class Engine:
    def __init__(self, spectrograph: SpectrographSetup):
        self.spectrograph = spectrograph

    def opticaly_broaden(self, wave, flux):
        """
        flux may be a multidimensional array as long as the last dimension matches wave


        The optical PSF comes from effects prior to, from, and after the grating
        In detail the PSF will be both chromatic and non-Gaussian with chromaticity stemming from both chromaticity in the
        optics and from aberrations as a result of slit images taking different paths through the optics. The former would
        slowly vary over the full wavelength domain while the latter would vary over a single order (as well as over
        the wavelengths in the order). It is reasonable to assume that an achromatic gaussian may be used to represent the
        intensity profile of slit image produced by the camera at the for a well-designed optical spectrograph. Since this
        is a effect on the image it can be freely done on a per-order basis without worry about interplay between the orders,
        this facilitates low-cost support for a first order approximation of chromaticity by varying the gaussian width with
        each order.

        Technically a sinc of the grating convolved with the optical spot

        kernel width is function of order data is a function of order

        NB a further approximation can be made by moving a space space with constant sampling in dl/l=c and arguing that the width of the LSF is directly
        proportional to lambda. Doing this does change the effective resolution though so care should be taken that there are
        sufficient samples per pixel, with this approximation the kernel a single kernel of fixed width in dl/l

        For now though just treat it as constant and defined at the midle of the wavelength range
        """
        sample_width = wave.mean() * self.spectrograph.nondimensional_lsf_width / np.diff(wave).mean()
        return ndi.gaussian_filter1d(flux, sample_width, axis=0)*flux.unit

    def build_mkid_kernel(self, n_sigma, width, sampling):
        """
        Return a kernel corresponding to an MKID resolution element of width covering n_sigma on either side

        width/sampling specifies how many points are used, though the kernel will round up to the next odd number
        """
        max_mkid_kernel_width = 2 * n_sigma * width
        mkid_kernel_npoints = np.ceil((max_mkid_kernel_width / sampling).si.value).astype(int)
        if not mkid_kernel_npoints % 2:
            mkid_kernel_npoints += 1
        return sig.gaussian(mkid_kernel_npoints, (width / sampling).si.value)  # width/sampling is the dimensionless width
