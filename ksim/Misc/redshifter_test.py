# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 10:09:38 2021

@author: BVH
"""


import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import L_sun, c
from sedpy.observate import getSED


data = np.load('MRK348_DATA_SPEC.npy')

plt.figure()
plt.plot(data[0],data[1])
plt.xlabel('Wavelength / nm')
plt.ylabel('Flux')

curr_wls = data[0]*10 #in Angstrom
curr_flux = data[1]
# z(~) 0.000017 [0.000015] hd212442 |||| mrk348 0.015
curr_z = 0.015
d_L_pre_rshift = cosmo.luminosity_distance(curr_z).to("cm").value
curr_L = 4*np.pi*curr_flux*(d_L_pre_rshift**2)

reds = np.linspace(0.00002,2,num=50)
plt.figure()


zred = 0.7 # the redshift
    
# Get spectrum from a fsps.StellarPopulation object with desired parameters already set.
# These have units of angstroms and L_\sun/Hz (flux density), usually for one solar mass formed.
#wave_rest_aa, L_nu_rest = sp.get_spectrum()

# redshift the wavelengths.
a = 1 + zred
rshift_wls = curr_wls * a

d_L = cosmo.luminosity_distance(zred).to("cm").value
rshift_flux = curr_L / (4 * a * np.pi * d_L**2)


# Now get magnitudes.
#filters = ["jwst_f090w"]
#apparent_mags_AB = getSED(rshift_wls, rshift_flux, filterlist=filters)
plt.plot(rshift_wls/10,rshift_flux)

'''
for i in range(len(reds)):

    zred = reds[i] # the redshift
    
    # Get spectrum from a fsps.StellarPopulation object with desired parameters already set.
    # These have units of angstroms and L_\sun/Hz (flux density), usually for one solar mass formed.
    #wave_rest_aa, L_nu_rest = sp.get_spectrum()
    
    # redshift the wavelengths.
    a = 1 + zred
    rshift_wls = curr_wls * a
    
    d_L = cosmo.luminosity_distance(zred).to("cm").value
    rshift_flux = curr_L / (4 * a * np.pi * d_L**2)
    
    
    # Now get magnitudes.
    #filters = ["jwst_f090w"]
    #apparent_mags_AB = getSED(rshift_wls, rshift_flux, filterlist=filters)
    plt.semilogy(rshift_wls/10,rshift_flux)
'''
    
plt.xlabel('Wavelength / nm')
plt.ylabel('Flux')

