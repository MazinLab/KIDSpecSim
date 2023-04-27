import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.interpolate as interp
import astropy.units as u
import time

from spectra import PhoenixModel, AtmosphericTransmission, FilterTransmission, TelescopeTransmission, clip_spectrum
from spectrograph import GratingSetup, SpectrographSetup
from detector import MKIDDetector
from engine import Engine


class Simulator:
    def __init__(
            self,
            spectrograting: str = 'default',
            type_of_spectra: str = 'phoenix',
            convol: str = 'full',
            pixel_lim: int = 5000,
            exptime: u.Quantity = 200 * u.s,
            minwave: u.Quantity = 400 * u.nm,
            maxwave: u. Quantity = 800 * u.nm,
            **properties
    ) -> None:
        """
        Simulator for the MKID spectrometer.

        :param spectrograting: 'new' if changing fundamental properties of the default spectrograph or grating,
            'default' if keeping default properties
        :param type_of_spectra: Type of spectrum being simulated, can be:
            'phoenix' -     Phoenix model spectrum of a 4300 K star,
            'blackbody' -   Blackbody model spectrum of a 4300 K star with R_sun at 1kpc,
            'delta' -       Narrow-width delta-like spectrum at the central wavelength 600 nm
            'calibration' - if generating a calibration spectrum, this will override any existing MKID R0s
        :param convol: 'full' if conducting a full convolution with the MKID response, 'simple' otherwise
        :param pixel_lim: Total number-of-photons limit for any pixel, 1000 takes a few minutes for full simulation
        :param exptime: Total exposure time for observation in astropy units of time, longer results in fewer merges
        :param minwave: Shorter wavelength minimum value
        :param maxwave: Longer wavelength maximum value
        :param properties: keyword settings when changing spectrometer properties, ALL must be supplied if so.
            The default MKID detector, spectrograph, and grating have:
            npix: int = 2048,
            R0: float = 15,
            l0: u.Quantity = 800 * u.nm,
            m0: int = 5,
            m_max: int = 9,
            pixels_per_res_elem: float = 2.5,
            pixel_size: u.Quantity = 20 * u.micron,
            focal_length: u.Quantity = 350 * u.mm,
            littrow: bool = True
        :return: saves h5 file containing observed spectrum, syntax: '{type_of_spectra}_{pixel_lim}_R0{R0}.h5'
        :rtype: None
        """
        if spectrograting == 'new':
            self.new_spectrograting = True
        elif spectrograting == 'default':
            self.new_spectrograting = False
        self.type_of_spectra = type_of_spectra
        if type_of_spectra == 'calibration':
            self.type_of_spectra = 'blackbody'
            self.gen_cal = True
        else:
            self.gen_cal = False
        if convol == 'full':
            self.full_convolution = True
        elif convol == 'simple':
            self.full_convolution = False
        self.pixel_lim = pixel_lim
        self.exptime = exptime
        self.minwave = minwave
        self.maxwave = maxwave
        if self.new_spectrograting:
            acceptable_keys = ['npix', 'R0', 'l0', 'm0', 'm_max', 'pixels_per_res_elem', 'pixel_size',
                                   'focal_length', 'littrow']
            self.__dict__.update({k: v for k, v in properties.items() if k in acceptable_keys})

    def setup_spectrometer(self):
        """
        :return: detector, grating, and spectrograph class objects, and R0
        """
        if self.new_spectrograting:
            detector = MKIDDetector(npix=self.npix, pixel_size=self.pixel_size, R0=self.R0, l0=self.l0,
                                    generate_R0=self.gen_cal)
            grating = GratingSetup(self.l0, self.m0, self.pixel_size, self.npix, self.focal_length,
                                   littrow=self.littrow)
            spectrograph = SpectrographSetup(grating, detector, self.m0, self.m_max, self.l0, self.pixels_per_res_elem,
                                             self.focal_length, littrow=self.littrow)
            R0 = detector.design_R0
        else:
            grating = GratingSetup()
            detector = MKIDDetector(generate_R0=self.gen_cal)
            spectrograph = SpectrographSetup(grating, detector)
            R0 = detector.design_R0
        return detector, grating, spectrograph, R0

    def setup_spectrum(self):
        """
        :return: SourceSpectrum object given type of spectra from Simulator class object
        """
        if self.type_of_spectra == 'phoenix':
            spectra = PhoenixModel(4300, 0, 4.8)
            print(f"\nObtained Phoenix model spectrum of star with T_eff of 4300 K.")
        elif self.type_of_spectra == 'blackbody':
            # Optional comparison with a blackbody model:
            from synphot import SourceSpectrum
            from synphot.models import BlackBodyNorm1D
            spectra = SourceSpectrum(BlackBodyNorm1D, temperature=4300)
            print(f"\nObtained blackbody model spectrum of 4300 K star.")
        elif self.type_of_spectra == 'delta':
            # Optional sanity check: the below produces a single block spectrum at 600 nm that is 40 nm wide.
            from synphot import SourceSpectrum
            from specutils import Spectrum1D

            w = np.linspace(400, 800, 4000) * u.nm
            tens = np.full(1950, 0)
            sp = np.full(100, 0.01)
            f = np.append(tens, sp)
            f = np.append(f, tens) * u.photlam
            spectra = SourceSpectrum.from_spectrum1d(Spectrum1D(spectral_axis=w, flux=f)) / 1e20
            print(f"\nObtained 0.01 photlam narrow width spectrum at 600 nm "
                  f"that is 10 nm wide with 0 photlam elsewhere.")
        else:
            raise ValueError("type_of_spectra must be 'phoenix,' 'blackbody,' or 'delta.'")
        return spectra

    def apply_bandpass(self, spectrum):
        """
        :param spectrum: SourceSpectrum object
        :return: same spectrum multiplied by given bandpasses
        """
        spectra = [spectrum]
        bandpasses = [AtmosphericTransmission(), TelescopeTransmission(reflectivity=.9),
                      FilterTransmission(self.minwave, self.maxwave)]
        if self.gen_cal:
            from specutils import Spectrum1D
            from synphot import SpectralElement

            # finer grid spacing
            w = np.linspace(300, 1000, 1400000) * u.nm
            t = np.ones(1400000) * u.dimensionless_unscaled
            ones = Spectrum1D(spectral_axis=w, flux=t)
            spectra[0] *= SpectralElement.from_spectrum1d(ones)
        else:  # don't apply bandpasses to calibration spectra
            for i, s in enumerate(spectra):
                for b in bandpasses:
                    s *= b
                spectra[i] = s
        return spectra

    def apply_blaze(self, spectrograph, spectra):
        """
        :param spectrograph: Spectrograph class object
        :param spectra: SourceSpectrum object
        :return: unmasked and masked (w/ wavelength masks) blazed spectra
        """
        # Grating throughput impact is a function of wavelength and grating angles, handled for each order
        blaze_efficiencies = spectrograph.blaze(spectra.waveset)
        # spectrograph.blaze returns 2D array of blaze efficiencies [wave.size, norders]
        order_mask = spectrograph.order_mask(spectra.waveset.to(u.nm), fsr_edge=False)
        blazed_spectrum = blaze_efficiencies * spectra(spectra.waveset)
        print(f"Multiplied blaze efficiencies with spectrum.")

        masked_blaze = [blazed_spectrum[i, order_mask[i]] for i in range(len(spectrograph.orders))]
        mask_waves = [spectra.waveset[order_mask[i]].to(u.nm) for i in range(len(spectrograph.orders))]
        return blazed_spectrum, mask_waves, masked_blaze  # purpose of returning masked lists is for comparison

    def apply_convolution(self, spectra, broadened, engine, n_sigma_mkid, osamp):
        """
        :param spectra: SourceSpectrum object
        :param broadened: ndarray with flux unit
        :param engine: Engine class object
        :param n_sigma_mkid: number of sigma to calculate convolution with
        :param osamp: how much to oversample using smallest pixel extent
        :return: important intermediate values, resulting convolution arrays/wavelengths, MKID kernel
        """
        sampling_data = engine.determine_mkid_convolution_sampling(oversampling=osamp)
        if self.full_convolution:
            result_wave, result, mkid_kernel = \
                engine.convolve_mkid_response(spectra.waveset, broadened, *sampling_data, n_sigma_mkid=n_sigma_mkid)
        else:
            result_wave, result, mkid_kernel = \
                engine.multiply_mkid_response(spectra.waveset, broadened, oversampling=osamp, n_sigma_mkid=n_sigma_mkid)
        return result_wave, result, mkid_kernel

    def simulate_spectrum(self, plot_pdf: bool = False):
        """
        :param bool plot_pdf: True if saving relevant intermediate plots to pdf
        :return: None (saves observed spectrum to h5 file)
        """
        tic = time.time()
        print("Simulating spectrum from MKID Spectrometer.")
        u.photlam = u.photon / u.s / u.cm ** 2 / u.AA  # photon flux per wavelength

        N_SIGMA_MKID = 3
        THREESIG = 0.9973
        OSAMP = 10
        SIGMA_TO_WIDTH = 2.355

        detector, grating, spectrograph, R0 = self.setup_spectrometer()
        nord = len(spectrograph.orders)
        npix = detector.n_pixels

        spectra_0 = self.setup_spectrum()
        engine = Engine(spectrograph)
        spectra = self.apply_bandpass(spectra_0)
        inbound = clip_spectrum(spectra[0], self.minwave, self.maxwave)
        blazed_spectrum, mask_waves, masked_blaze = self.apply_blaze(spectrograph, inbound)
        broadened_spectrum = engine.optically_broaden(inbound.waveset, blazed_spectrum)

        masked_broad = [engine.optically_broaden(mask_waves[i], masked_blaze[i], axis=0) for i in range(nord)]

        if plot_pdf:
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.5, 11), dpi=300)
            axes = axes.ravel()
            plt.suptitle("Intermediate plots for MKID spectrometer simulation "
                         f"of {self.type_of_spectra} spectrum", fontweight='bold')
            p = 0
            axes[p].grid()
            axes[p].plot(spectra_0.waveset.to('nm'), spectra_0(spectra_0.waveset), color="C0", label="Original")
            axes[p].plot(inbound.waveset.to('nm'), inbound(inbound.waveset), color="C1", label="Bandpassed")
            axes[p].set_xlim([self.minwave.value-25, self.maxwave.value+25])
            for i in range(nord):
                axes[p].plot(mask_waves[i], masked_blaze[i], color=f"C{i + 2}", label=f'Blazed O{i + spectrograph.m0}')
            for i in range(nord):
                axes[p].plot(mask_waves[i], masked_broad[i], 'k')
            axes[p].plot(mask_waves[-1], masked_broad[-1], 'k', label="Opt.-broadened")
            axes[p].set_ylabel(r"Flux Density (phot $cm^{-2} s^{-1} \AA^{-1})$")
            axes[p].set_title("Original through Optically-Broadened")
            axes[p].legend(loc='upper left')
            p += 1

        convol_wave, convol, mkid_kernel = self.apply_convolution(inbound, broadened_spectrum, engine, N_SIGMA_MKID, OSAMP)
        pixel_samples_frac, pixel_max_npoints, pixel_rescale, dl_pixel, lambda_pixel, dl_mkid_max, sampling = \
            engine.determine_mkid_convolution_sampling(oversampling=OSAMP)

        if plot_pdf:
            pix_leftedge = spectrograph.pixel_to_wavelength(detector.pixel_indices, spectrograph.orders[:, None])
            split_indices = np.empty([nord, npix])
            filter_mask = np.where(lambda_pixel[-1, :].to(u.nm).value > self.minwave.value)[0]
            split_indices[-1, filter_mask] = [np.where(np.abs((mask_waves[-1].to(u.nm) -
                                                              pix_leftedge[-1, j].to(u.nm)).value) < 1e-3)[0][0]
                                              for j in filter_mask]
            split_indices[:-1, :] = [[np.where(np.abs((mask_waves[i].to(u.nm) -
                                                      pix_leftedge[i, j].to(u.nm)).value) < 1e-3)[0][0]
                                      for j in range(npix)] for i in range(nord-1)]
            lamtopixel = np.empty([nord, npix])
            # regaining flux density by summing flux by previous indices and multiplying by dx
            lamtopixel[:, :-1] = [[
                scipy.integrate.trapz(
                    masked_broad[i][int(split_indices[i, j]):int(split_indices[i, j + 1])].to(u.photlam).value,
                    x=mask_waves[i][int(split_indices[i, j]):int(split_indices[i, j + 1])].to(u.nm).value)
                for j in range(npix - 1)] for i in range(nord)]
            lamtopixel[:, -1] = lamtopixel[:, -2]

            x = np.array([[np.linspace(-N_SIGMA_MKID * (pixel_rescale[i, j].to(u.nm) * dl_mkid_max.to(u.nm) /
                                                        sampling.to(u.nm)).value / SIGMA_TO_WIDTH,
                                       N_SIGMA_MKID * (pixel_rescale[i, j].to(u.nm) * dl_mkid_max.to(u.nm) /
                                                       sampling.to(u.nm)).value / SIGMA_TO_WIDTH,
                                       len(mkid_kernel)) for j in range(npix)] for i in range(nord)])
            dx_norm = x[:, :, 1] - x[:, :, 0]
            norms = np.sum(mkid_kernel) * dx_norm / THREESIG  # since 3 sigma on either side is not exactly 1
            convol_plot = (convol / (norms * u.nm)).to(u.photlam)

            dx = [[convol_wave[1, i, j].to(u.nm).value - convol_wave[0, i, j].to(u.nm).value
                   for j in range(npix)] for i in range(nord)]
            convol_summed = np.sum(convol_plot.to(u.photlam).value, axis=0) * dx

            axes[p].grid()
            for i in range(nord):
                axes[p].plot(lambda_pixel[i, :], convol_summed[i, :], color='C3')
            axes[p].plot(lambda_pixel[i, :], convol_summed[i, :], color='C3', label="Conv.+Int.")
            for i in range(nord - 1):
                axes[p].plot(lambda_pixel[i, :].to(u.nm).value, lamtopixel[i, :], 'blue', alpha=0.5)
            axes[p].plot(lambda_pixel[-1, :].to(u.nm).value, lamtopixel[-1, :], 'blue', alpha=0.5,
                         label=r'$\lambda$-to-Pixel Int.')
            axes[p].set_ylim(top=5e16)
            axes[p].set_ylabel(r"Flux (phot $cm^{-2} s^{-1})$")
            axes[p].set_title(r"Convolved+Integrated & $\lambda$-to-Pixel Integrated")
            axes[p].legend()
            p += 1

        time_photons, lambda_photons = engine.draw_photons(convol_wave, convol, exptime=self.exptime, limit_to=self.pixel_lim)

        photons, observed = detector.observe(time_photons, lambda_photons)

        if plot_pdf:
            lambda_pixel = lambda_pixel[::-1, :]  # flipping order of arrays since photon draw is list of wavelengths
            convol_summed = convol_summed[::-1, :]
            fsr = spectrograph.fsr(spectrograph.orders)[::-1]
            pixel_rescale = pixel_rescale[::-1, :]

            hist_bins = np.empty((nord + 1, npix))  # choosing rough histogram bins by using FSR of each pixel/wave
            hist_bins[0, :] = (lambda_pixel[0, :] - fsr[0] / 2).to(u.nm).value
            hist_bins[1:nord, :] = [(lambda_pixel[i, :] + fsr[i] / 2).to(u.nm).value for i in range(nord - 1)]
            hist_bins[nord, :] = np.full(npix, self.maxwave.value + 100)

            photon_list = []
            resid_map = np.arange(npix, dtype=int) * 10 + 100
            for i in range(npix):  # sorting photons by resID (i.e. pixel)
                idx = np.where(photons.resID == resid_map[i])
                photon_list.append(photons[:observed].wavelength[idx].tolist())

            photons_binned = np.empty((nord, npix))
            for j in range(npix):  # sorting by histogram bins as before
                photons_binned[:, j], _ = np.histogram(photon_list[j], bins=hist_bins[:, j], density=False)

            axes[p].grid()
            for i in range(nord - 1):
                axes[p].plot(lambda_pixel[i, :], photons_binned[i, :], color='C3')
            axes[p].plot(lambda_pixel[-1, :], photons_binned[-1, :], color='C3', label='Observed+FSR-Binned')
            ax_0 = axes[p].twinx()
            for i in range(nord - 1):
                ax_0.plot(lambda_pixel[i, :], convol_summed[i, :], 'blue', alpha=0.5)
            ax_0.plot(lambda_pixel[-1, :], convol_summed[-1, :], 'blue', alpha=0.5, label='Conv.+Int. Input')
            axes[p].set_ylabel("Photon Count")
            ax_0.set_ylabel(r"Flux (phot $cm^{-2} s^{-1})$")
            ax_0.set_ylim(top=4.72e16)
            axes[p].tick_params(axis="y", labelcolor=f'C3')
            ax_0.tick_params(axis="y", labelcolor='blue')
            axes[p].set_xlabel("Wavelength (nm)")
            axes[p].set_title("Convolved+Integrated & Observed Photons")
            axes[p].legend(loc='upper left')
            ax_0.legend(loc='lower right')
            fig.tight_layout()
            plt.subplots_adjust(top=0.92, right=0.6, left=0.1)
            fig.savefig(f'plots/intplots_{self.type_of_spectra}_R0{R0}.pdf')
            plt.show()
            print(f"Intermediate plots have been saved to: plots/intplots_{self.type_of_spectra}_R0{R0}.pdf.")

        # Dump to HDF5
        # TODO this will need work as the pipeline will probably default to MEC HDF headers
        from mkidpipeline.steps import buildhdf

        buildhdf.buildfromarray(photons[:observed],
                                user_h5file=f'h5_output/{self.type_of_spectra}_{self.pixel_lim}_R0{R0}.h5')
        # at this point we are simulating the pipeline and have gone past the "wavecal" part. Next is >to spectrum.
        print(f"\nCompiled data to h5 file: h5_output/{self.type_of_spectra}_{self.pixel_lim}_R0{R0}.h5.")

        toc = time.time()
        print(f"\nMKID Spectrometer spectral simulation completed in {round((toc - tic) / 60, 2)} minutes.")


if __name__ == '__main__':
    # example spectral simulation with fewer pixels
    sim = Simulator(pixel_lim=1000)
    sim.simulate_spectrum(plot_pdf=True)
