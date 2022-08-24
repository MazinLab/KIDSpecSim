# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:19:56 2020

@author: BVH
"""



import numpy as np

from parameters import folder, object_name, n_pixels, pix_mult
from useful_funcs import where_in_array
from matplotlib import pyplot as plt

object_photons = (np.load('%s/%s_data/%s_wavelengths.npy'%(folder,object_name,object_name)),
                      np.load('%s/%s_data/%s_photon_counts.npy'%(folder,object_name,object_name)))

'''
pixel = 6222
gauss = np.load('%s/Pixel_Data/%s_gaussian_array.npy'%(folder,pixel)) #loading in the gaussian for that pixel
where_greater = np.max(gauss,axis=0)
plt.figure()
plt.plot(object_photons[0][::3]*1e7,where_greater[::3])
'''

Y = np.linspace(1,n_pixels,n_pixels)
X = object_photons[0]*1e7


data_gaussians = np.zeros((n_pixels,len(object_photons[0])))

perc_complete = 0

for pix in range(n_pixels):
    
    gauss = np.load('%s/Pixel_Data/%s_gaussian_array.npy'%(folder,pix))
    
    max_values = np.max(gauss,axis=0)
    
    maximum = np.max(max_values)
    
    max_values = max_values / maximum
    
    data_gaussians[pix,:] = max_values
    
    if (pix == pix_mult).any():
        perc_complete += 10
        print('Computing data for 2D plot - %i%% complete'%perc_complete)

print('Computing data for 2D plot - 100% complete')

plt.figure()
plt.contourf(X,Y,data_gaussians,levels=20)
plt.ylim(8000,1)
plt.colorbar()
plt.xlabel('Wavelength / nm')
plt.ylabel('Pixel Number')
