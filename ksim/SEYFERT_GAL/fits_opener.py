# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:00:24 2020

@author: BVH
"""

from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt







fits_file = fits.open('mrk348_hk_scaled.FITS')

a = fits_file[0].data[0,:,:][0]

x = np.arange(13889.96248,(13889.96248+(9.50505*len(a))),9.50505)

fits_file.close()
plt.figure()
plt.plot(x,a)


fits_file = fits.open('mrk348_zj_scaled.FITS')




