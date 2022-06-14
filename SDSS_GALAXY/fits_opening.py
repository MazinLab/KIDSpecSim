# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:20:09 2020

@author: BVH
"""


from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt

fits_file = fits.open('spec-0392-51793-0106.FITS')
specx = 10**fits_file[1].data['loglam']
specy = fits_file[1].data['flux'] *10e-17
fits_file.close()

plt.figure()
plt.plot(specx,specy,'k-')
plt.xlabel('Wavelength / $\AA$')
plt.ylabel('Flux / $ergcm^{-2}s^{-1}\AA^{-1}$')
plt.title('SDSS J003948.20+000814.6')



