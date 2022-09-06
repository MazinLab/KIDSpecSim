import numpy as np
import scipy.interpolate as interp
import scipy
import scipy.signal as sig
import scipy.ndimage as ndi
import astropy.units as u

from spectra import PhoenixModel, AtmosphericTransmission, FilterTransmission, TelescopeTransmission
from spectrograph import GratingSetup, SpectrographSetup
from detector import MKIDDetector


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
        return ndi.gaussian_filter1d(flux, sample_width, axis=0) * flux.unit

    def build_mkid_kernel(self, n_sigma, width, sampling):
        """
        Return a kernel corresponding to an MKID resolution element of width covering n_sigma on either side

        width/sampling specifies how many points are used, though the kernel will round up to the next odd number
        """
        max_mkid_kernel_width = 2 * n_sigma * width
        mkid_kernel_npoints = np.ceil((max_mkid_kernel_width / sampling).si.value).astype(int)
        if not mkid_kernel_npoints % 2:
            mkid_kernel_npoints += 1
        return sig.gaussian(mkid_kernel_npoints,
                            (width / sampling).si.value)  # width/sampling is the dimensionless width

    def determin_apodization(self, pixel_samples_frac, pixel_max_npoints):
        """
        Samples is a bit of a misnomer in that this is the number of normalized mkid kernel width elements that would cover
        the pixel. So 20.48 elements would be 21 sample points with one at the center and 10 to either side. the 10th would be
        sampling the pixel just a bit shy of its edge (.24 * .5/(20.48/2), where the edge of the pixel is at .5
        the next sample is out of the wave domain of the pixel, though its bin might still include some flux from
        within the pixel: (.5-(np.ceil(pixel_samples_frac)-pixel_samples_frac)).clip(0)/2

        Note mkidsigma = dl_pixel*dl_mkid_max/sampling/pixel_samples so MKID R error induced by sampling is
        mkid_sigma_error = -dl_pixel*dl_mkid_max/sampling/pixel_samples_frac**2 * (pixel_samples_frac-pixel_samples_frac).std()
        plt.imshow(mkid_sigma_error/dl_mkid_pixel,aspect=2048/5, origin='lower', interpolation='nearest');plt.colorbar();plt.show()
        but actually this many not be an issue at all ad the samplin isn't the points its where those points HIT the pixel
        wavelengths relative to each other and that is handled by the interpolation!

        A (a*dp/dp)%1 >= .5
        v0   v1     v2                 v3    v4
        |  e  |  e  |    e     ||       |  e  | ....
        0     1     2   2.5  pixedge_a  3     4
        flux/dp = v0 + v1 + v2 + v3*(a-floor(a)-.5)
        coord=floor(nsamp/2)+1

        B (a*dp/dp)%1 <=.5
        v0   v1     v2                 v3    v4
        |  e  |  e  |     ||       e   |  e  | ....
        0     1     2  pixedge_a  2.5  3     4
        flux/dp = v0 + v1 + v2*(a-floor(a)+.5)
        coord=floor(nsamp/2)

        """
        # TODO this many not be quite right, cf. interp_wave
        x = np.linspace(-pixel_max_npoints / 2, pixel_max_npoints / 2, num=pixel_max_npoints)
        in_pixel = np.abs(x[:, None, None]) <= (pixel_samples_frac / 2)

        edge = pixel_samples_frac / 2
        last = edge.astype(int)
        caseb = last == np.round(edge).astype(int)

        apod = edge - edge.astype(int) - .5  # compute case A
        apod[caseb] += 1  # correct values to caseb

        apod_ndx = last + (~caseb)  # Case A needs to go one further
        apod_ndx = np.array([pixel_max_npoints // 2 - apod_ndx, pixel_max_npoints // 2 + apod_ndx])
        apod_ndx.clip(0, pixel_max_npoints - 1, out=apod_ndx)

        interp_apod = np.zeros((pixel_max_npoints,) + pixel_samples_frac.shape)
        ord_ndx = np.arange(interp_apod.shape[1], dtype=int)[None, :, None]
        pix_ndx = np.arange(interp_apod.shape[2], dtype=int)[None, None, :]
        interp_apod[in_pixel] = 1
        interp_apod[apod_ndx[0], ord_ndx, pix_ndx] = apod
        interp_apod[apod_ndx[1], ord_ndx, pix_ndx] = apod

        return interp_apod

    def determine_mkid_convolution_sampling(self, oversampling=10):
        """
        # Each MKID has its own spectral resolution, which we like to approximate as a guassian with width depending
        # on wavelength. The spectrograph is sending only certain wavelengths in each order to each pixel.
        # This incident spectrum can then be convolved with a gaussian of varying width and the resulting distribution
        # sampled to find "seen" wavelengths. If the variation in R_MKID over a single pixel is small then this can be
        # approximated as a convolution with a per-order gaussian and the results summed.

        # kernel width is function of pixel, order
        # data is a function of pixel, order

        # With say an R_MKID_800nm=15 we would have about a 55nm FWHM and targeting a spectral resolution of 3400 w/2.5 pix dpix ~ 0.05 nm
        # well sampling (say 10 samples per pixel) would imply then a 3sigma kernel of about 33000 elements
        # there would be so the kernel array would be [norder, npixel, 3300*osamp ] or about 1.25GB for a float and 10x osamp
        #this would need to be convolved with the data which would be [norder, npixel, osamp] (negligible)

        #However here we can almost certainly play with the sampling to do this more efficiently. By adopting a nonuniform
        # sampling (e.g. by allowing dl to be different for each pixel) we are effectively scaling the width of the applied
        # gaussian, which could then be a single oversampled gaussian out to some nsigma in normalized coordinates.

        # In the spectrograph the sampling at a given pixel flows from the spectrograph design
        # dl_pixel = Dbeta_subtended*lambda(pixel)/2/tan(beta(pixel))=Dbeta_subtended*sigma_grating*cos(beta(pixel))/m
        # which by design will be ~ FSR/npix for the last order at the center beta will vary about 6 degrees over the order and
        # which will translate to a <±10% effect for likely grating angles so call the pixel sampling that of the order average
        # which can be found to be ~ l_center_limiting_order/Npix/order about 0.0355 nm to 0.078 nm
        # (for a 400-800 m=5-9 spectrograph).
        # The dl_MKID will be l^2/R0/l0. So we require (2*n_sigma_mkid*l_max^2/R0_min/l0) / min(dl_pixel) * osamp
        # points across the kernel in the worst case.
        # For pixels we want at least osamp samples across the pixel's so that would mean the sampling will need to be at least
        #  min(dl_pixel)/osamp and at most 10*max(dl_pixel)/min(dl_pixel) samples across a pixel


        # so for the extrema:
        # 2*sigma*dl_MKID_800 = 400  # this is the smallest gaussian kernel domain
        # 2*sigma*dl_MKID_400 = 67  # R0_max not min, this is the smallest gaussian kernel domain
        # dl_pix_800 = 0.0781   # the maximum change in wavelength across a pixel
        # dl_pix_400 = 0.0373   # l0_center/npix*(1-c_beta*m0/m_max)/m_max  #the minimum change in wavelength across a pixel, note the lower angular width at higher order
        """
        spectrograph = self.spectrograph
        detector = self.spectrograph.detector
        max_beta_m0 = spectrograph.grating.beta(spectrograph.l0, spectrograph.m0)
        min_beta_mmax = spectrograph.grating.beta(spectrograph.minimum_wave, spectrograph.m_max)
        # The maximum and minimum change in wavelength across a pixel
        #  NB You can get a crude approximation by using the following where c_beta the fractional change in beta across the order
        #   e.g. for 6 degrees with acenteral angle of 45 about .1 in the minimum order
        #   dl_pix_min_wave = spectrograph.central_wave(m0)/npix*(1-c_beta*m0/m_max)/m_max
        #   dl_pix_max_wave = spectrograph.central_wave(m0)/npix*(1+c_beta)/m0
        dl_pix_max_wave = spectrograph.pixel_scale / spectrograph.grating.angular_dispersion(spectrograph.m0,
                                                                                             max_beta_m0)
        dl_pix_min_wave = spectrograph.pixel_scale / spectrograph.grating.angular_dispersion(spectrograph.m_max,
                                                                                             min_beta_mmax)

        # We need to compute the worst case kernel width which will be towards the long wavelength end in the lowest order
        # wave_ord0=np.linspace(-.5,.5, num=osamp*detector.n_pixels)*spectrograph.fsr(m0)+spectrograph.central_wave(m0)
        # dl_mkid_max = (wave_ord0**2/detector.mkid_constant(spectrograph.wavelength_to_pixel(wave_ord0))).max()  #sigma not fwhm
        dl_mkid_max = (spectrograph.l0 ** 2 / detector.mkid_constant(detector.pixel_indices)).max()

        sampling = dl_pix_min_wave / oversampling

        # pixel wavelength centers (nord, npixel)
        # linear array for lowest order, then broadcast andscale via grating equation i.e. m/m'

        # lambda_pixel = (np.linspace(-.5, .5, num=detector.n_pixels) * spectrograph.fsr(m0) + spectrograph.central_wave(m0)) * (m0/spectrograph.orders)[:,None]
        # lambda_pixel_pad1 = ((np.linspace(-.5,.5, num=detector.n_pixels+1)-1/detector.n_pixels/2) * spectrograph.fsr(m0) + spectrograph.central_wave(m0)) * (m0/spectrograph.orders)[:,None]
        # dl_pixel = np.diff(lambda_pixel_pad1, axis=1)  #dl spectrograph subtended by pixel

        lambda_pixel = spectrograph.pixel_center_wavelengths()
        dl_pixel = spectrograph.pixel_scale / spectrograph.angular_dispersion()
        dl_mkid_pixel = detector.mkid_resolution_width(lambda_pixel, detector.pixel_indices)

        # Each pixel width is some fraction of the kernel's sigma, to handle the stretching properly we need to make sure the
        # correct number of samples are used for the pixel flux. dl_mkid_max/sampling points is mkid sigma
        # so a pixel of dl_pixel width with a mkid sigma of dl_mkid_pixel has dl_mkid_max/sampling points in dl_mkid_pixel
        # so `dl_pixel` nanometers corresponds to dl_pixel/dl_mkid_pixel * dl_mkid_max/sampling points
        # so the sampling of that pixel is dl_mkid_pixel*sampling/dl_mkid_max
        # pixel_samples = dl_pixel/dl_mkid_pixel * dl_mkid_max/sampling
        pixel_rescale = dl_mkid_pixel * sampling / dl_mkid_max

        pixel_samples_frac = (dl_pixel / pixel_rescale).si.value
        pixel_max_npoints = np.ceil(pixel_samples_frac.max()).astype(int)
        if not pixel_max_npoints % 2:  # ensure there is a point at the pixel center
            pixel_max_npoints += 1

        return (pixel_samples_frac, pixel_max_npoints, pixel_rescale, dl_pixel, lambda_pixel,
                dl_mkid_max, sampling)

    def convolve_mkid_response(self, wave, spectral_fluxden,
                               pixel_samples_frac, pixel_max_npoints, pixel_rescale,
                               dl_pixel, lambda_pixel, dl_mkid_max, sampling, n_sigma_mkid=3):
        """

        :param wave:
        :param spectral_fluxden:
        :param pixel_samples_frac:
        :param pixel_max_npoints:
        :param pixel_rescale:
        :param dl_pixel:
        :param lambda_pixel:
        :param dl_mkid_max:
        :param sampling:
        :param n_sigma_mkid:
        :return: result wavelengths and flux arrays [nsamples, norder, npixel] each
        """

        x = np.linspace(-pixel_max_npoints // 2, pixel_max_npoints // 2, num=pixel_max_npoints)
        interp_wave = (x[:, None, None] / pixel_samples_frac) * dl_pixel + lambda_pixel

        interp_apod = self.determin_apodization(pixel_samples_frac, pixel_max_npoints)

        # Compute the convolution data
        convoldata = np.zeros(interp_apod.shape)
        for i, bs in enumerate(spectral_fluxden):
            specinterp = interp.interp1d(wave.to('nm').value, bs, fill_value=0, bounds_error=False, copy=False)
            convoldata[:, i, :] = specinterp(interp_wave[:, i, :].to('nm').value)

        # Apodize to handle fractional bin flux apportionment
        convoldata *= interp_apod

        # Do the convolution
        mkid_kernel = self.build_mkid_kernel(n_sigma_mkid, dl_mkid_max, sampling)
        sl = slice(pixel_max_npoints // 2, -(pixel_max_npoints // 2))
        result = sig.oaconvolve(convoldata, mkid_kernel[:, None, None], mode='full', axes=0)[sl]

        result *= u.photlam

        # TODO I'm not sure this is right, might should be simply sampling
        result *= pixel_rescale[None, ...]  # To photons
        # Compute the wavelengths for the output, converting back to the original sampling, cleverness is done
        result_wave = (np.arange(result.shape[0]) - result.shape[0] // 2)[:, None, None] * pixel_rescale[
            None, ...] + lambda_pixel

        return result_wave, result

    def multiply_mkid_response(self, wave, spectral_fluxden, oversampling=10, n_sigma_mkid=3):
        """
        Instead of convolving the MKID response just multiply it by the average fluxdensity in the pixel

        """

        spectrograph = self.spectrograph
        detector = self.spectrograph.detector
        lambda_pixel = spectrograph.pixel_center_wavelengths()
        dl_pixel = spectrograph.pixel_scale / spectrograph.angular_dispersion()
        dl_mkid_pixel = detector.mkid_resolution_width(lambda_pixel, detector.pixel_indices)

        sampling = dl_pixel.min() / oversampling

        pixel_rescale = dl_mkid_pixel * sampling / dl_mkid_pixel.max()

        flux = np.zeros(dl_pixel.shape)

        for i, bs in enumerate(spectral_fluxden):
            specinterp = interp.interp1d(wave.to('nm').value, bs, fill_value=0, bounds_error=False, copy=False)
            flux[i, :] = specinterp(lambda_pixel[i, :].to('nm').value)

        mkid_kernel = self.build_mkid_kernel(n_sigma_mkid, dl_mkid_pixel.max(), sampling)

        result = flux * mkid_kernel[:, None, None]

        # Compute the wavelengths for the output, converting back to the original sampling, cleverness is done
        result_wave = (np.arange(result.shape[0]) - result.shape[0] // 2)[:, None, None] * pixel_rescale[
            None, ...] + lambda_pixel

        result *= pixel_rescale[None, ...]  # To photons

        return result_wave, result

    def draw_photons(self, result_wave, result, area=np.pi * (4 * u.cm) ** 2, exptime=1 * u.s, limit_to=1000):
        """result and result_wave are [nconvolution_output, nords, npixels] arrays, units of result should be
        photlambda*dwave

        limit
        """

        # Now, compute the CDF from dNdE and set up an interpolating function
        cdf_shape = int(np.prod(result.shape[:2])), result.shape[-1]
        result_pix = result.reshape(cdf_shape)
        wave_pix = result_wave.reshape(cdf_shape)
        cdf = np.cumsum(result_pix, axis=0)
        rest_of_way_to_photons = area * exptime
        cdf *= rest_of_way_to_photons
        cdf = cdf.decompose()  # todo this is a slopy copy
        total_photons = cdf[-1, :]

        # putting Poisson draw after limiting because int64 error
        if total_photons.value.max() > limit_to:
            print(f'Limiting to {limit_to} photons per pixel max')
            total_photons_ltd = (total_photons.value / total_photons.value.max() * limit_to).astype(int)  # crappy limit to 1000 photons per pixel for testing
            N = np.random.poisson(total_photons_ltd)  # Now assume that you want N photons as a Poisson random number
        else:
            N = np.random.poisson(total_photons.value)

        cdf /= total_photons
        # Decide on wavelengths and times
        l_photons, t_photons = [], []

        for i, (x, n) in enumerate(zip(cdf.T, N)):  # for each pixel
            cdf_interp = interp.interp1d(x, wave_pix[:, i].to('nm').value, fill_value=0, bounds_error=False, copy=False)
            l_photons.append(cdf_interp(np.random.uniform(0, 1, size=n)) * u.nm)
            t_photons.append(np.random.uniform(0, 1, size=n) * exptime)
        return t_photons, l_photons
