import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot(ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8.5, 11), dpi=300)
    # add more


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

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.bin_edges, f)
            pickle.dump(self.cov_matrix, f)
            pickle.dump(self.waves, f)
            pickle.dump(self.sim_settings, f)

    def _load(self):
        with open(self.filename, 'rb') as f:
            self.bin_edges = pickle.load(f)
            self.cov_matrix = pickle.load(f)
            self.waves = pickle.load(f)
            self.sim_settings = pickle.load(f)
