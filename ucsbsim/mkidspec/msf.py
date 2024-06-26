import numpy as np
import matplotlib.pyplot as plt


class MKIDSpreadFunction:
    def __init__(
            self,
            bin_edges=None,
            cov_matrix=None,
            waves=None,
            filename: str = '',
            sim_settings=None
    ):
        """
        :param bin_edges: n_ord+1 by n_pix array of wavelength or phase bins
        :param cov_matrix: n_ord by n_ord by n_pix array of covariance fractions belonging to another order
        :param waves: n_ord by n_pix array of retrieved wavelengths or phases
        :param str filename: where the MKIDSpreadFunction exists or where to save a new file
        :param sim_settings: simulation settings for the spectrograph
        """
        self.filename = filename
        if filename:
            self._load()
        else:
            assert bin_edges is not None and cov_matrix is not None and waves is not None, \
                'Must provide all other arguments if not specifying file.'
            self.bin_edges = bin_edges
            self.cov_matrix = cov_matrix
            self.waves = waves
            self.sim_settings = sim_settings
            nord = self.sim_settings.order_range[1] - self.sim_settings.order_range[0] + 1
            assert self.waves.shape[0] == nord, 'Wavelengths/phases do not have the correct number of orders.'
            assert self.cov_matrix.shape[0] == nord, \
                "Covariance matrix doesn't have the correct number of orders."
            assert self.cov_matrix.shape[0] == self.cov_matrix.shape[1], 'Covariance matrix is not square.'
            assert self.bin_edges.shape[0]-1 == nord, 'Bin edges do not have the correct number of orders.'
            assert self.bin_edges.shape[-1] == self.waves.shape[-1], \
                'Bin edges and wavelengths/phases have unequal number of pixels.'
            assert self.bin_edges.shape[-1] == self.cov_matrix.shape[-1],\
                'Bin edges and covariance have unequal number of pixels.'

    def save(self, filename=''):
        fn = filename or self.filename
        assert fn, "'filename' must be specified."
        np.savez(fn,
                 bin_edges=self.bin_edges,
                 cov_matrix=self.cov_matrix,
                 waves=self.waves,
                 sim_settings=self.sim_settings)

    def _load(self):
        msf = np.load(self.filename, allow_pickle=True)
        assert msf['bin_edges'].any(), 'File is missing bin edges or syntax is incorrect.'
        self.bin_edges = msf['bin_edges']
        assert msf['cov_matrix'].any(), 'File is missing covariance matrix or syntax is incorrect.'
        self.cov_matrix = msf['cov_matrix']
        assert msf['sim_settings'], 'File is missing simulation settings or syntax is incorrect.'
        self.sim_settings = msf['sim_settings']
        assert msf['waves'].any(), 'File is missing wavelengths/phases or syntax is incorrect.'
        self.waves = msf['waves']

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8.5, 11), dpi=300)
        # add more

