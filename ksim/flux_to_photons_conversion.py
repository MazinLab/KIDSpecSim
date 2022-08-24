# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:34:26 2020

@author: BVH
"""

import numpy as np
from scipy import interpolate

from matplotlib import pyplot as plt

from parameters import binstep,mirr_diam,h,c,exposure_t,object_x,object_y,n_pixels
from useful_funcs import nearest

def photons_conversion(spec,orig_spec,plotting=False):
    
    flux = spec[1]
    
    area_mirr = np.pi * ((0.5*mirr_diam)**2.) #collecting area
    
    numerator = flux * binstep * area_mirr * exposure_t #binstep should be in units of cm, but is in units of nm from seperation of wavelength data 
                                                        #entries from the spectrum data
    
    wavelengths = spec[0] * 10 #converting from nm to Ang since spectrum flux comes in Ang but wavelengths come in nm 
    
    denominator =  (h*c) / wavelengths #this is the photon energy portion hc/lambda
    
    photon_spec = numerator / denominator #final calculation for photon number
    
    if plotting == True:
        
        plt.figure()
        
        plt.subplot(211)
        plt.plot(orig_spec[0],orig_spec[1],'r-',markersize=1,label='Spectrum from data file')
        plt.xlabel(object_x)
        plt.ylabel(object_y)
        plt.legend(loc='best')
        
        plt.subplot(212)
        plt.plot(spec[0],photon_spec,'b-',markersize=1,label='Photon converted spectrum')
        plt.xlabel(object_x)
        plt.ylabel('Number of photons')
        plt.legend(loc='best')
        
        
    return spec[0],photon_spec

def photons_conversion_with_bin_calc(spec):    
    
    bin_width = (spec[0][-1] - spec[0][0]) / len(spec[0])
    
    flux = spec[1]
    
    area_mirr = np.pi * ((0.5*mirr_diam)**2.) #collecting area
    
    numerator = flux * bin_width * area_mirr * exposure_t #binstep should be in units of cm, but is in units of nm from seperation of wavelength data 
                                                        #entries from the spectrum data
    
    wavelengths = spec[0] * 10 #converting from nm to Ang since spectrum flux comes in Ang but wavelengths come in nm 
    
    denominator =  (h*c) / wavelengths #this is the photon energy portion hc/lambda
    
    photon_spec = numerator / denominator #final calculation for photon number
    
    #photon_spec = spec[1] / (np.max(spec[1])*1e-2)
    
    out_spec = np.zeros((2,len(spec[0])))
    out_spec[0] += spec[0]
    out_spec[1] += photon_spec
    
    return out_spec



def flux_conversion(spec):
    
    photons = spec[1]
    
    area_mirr = np.pi*((0.5*mirr_diam)**2.) #collecting area
    
    denominator = (binstep) * area_mirr * exposure_t #binstep * area_mirr * exposure_t 
    
    wavelengths = spec[0] * 10 #converting from nm to Ang since spectrum flux comes in Ang but wavelengths come in nm 
    
    numerator =  photons * ((h*c) / wavelengths) #this is the photon energy portion hc/lambda
    
    flux_spec = numerator / denominator #final calculation for photon number    
    
    out_spec = np.zeros((2,len(spec[0])))
    out_spec[0] += spec[0]
    out_spec[1] += flux_spec
    
    return out_spec


def flux_conversion_2(spec,wl_occ,w_o,w_o_bins_ir,w_o_bins_opt,plotting=False):
    
    wls = spec[0]
    binsteps = np.zeros(len(wls))
    ord_waves_tot = wl_occ[:,0]
    
    for i in range(len(spec[0])):
        
        if spec[0][i] < 1000:
            ord_waves = ord_waves_tot[7*2500:]
            w_o_bins = w_o_bins_opt
        else:
            ord_waves = ord_waves_tot[:(7*2500)]
            w_o_bins = w_o_bins_ir
        
        
        appearances = np.where(ord_waves == spec[0][i])[0] #was nearest
        no_appearances = len(appearances)            
        bins = np.zeros(no_appearances)
        for j in range(no_appearances):
            if appearances[j]+1 == len(ord_waves) or ord_waves[appearances[j]] > ord_waves[appearances[j]+1]:
                order = int(appearances[j] / n_pixels)
                bins[j] = w_o_bins[order,-1] - w_o_bins[order,-2]
            else:
                bins[j] += (ord_waves[appearances[j]+1] - ord_waves[appearances[j]])
            
        
        binsteps[i] = np.mean(bins)#*no_appearances/number_ord#tot_count#*no_appearances
        
    
    photons = spec[1]
    
    area_mirr = (0.5*mirr_diam)**2. #collecting area
    
    wavelengths = spec[0] * 10 #converting from nm to Ang since spectrum flux comes in Ang but wavelengths come in nm 
    
    denominator = area_mirr * exposure_t * (binsteps*1e-7)
    
    numerator =  photons * ((h*c) / wavelengths) #this is the photon energy portion hc/lambda
    
    fluxes = (numerator / denominator) #final calculation for photon number    

    flux_spec = np.zeros((2,len(wls)))
    flux_spec[0] += wls
    flux_spec[1] += fluxes
    
    #plt.figure()
    #plt.plot(flux_spec[0],flux_spec[1])
    
    
    return flux_spec,binsteps



def flux_conversion_3(spec):
    
    wls = spec[0]
    
    binstep = np.zeros(len(spec[0]))
    for i in range(len(binstep)):
        if i == len(binstep)-1:
            binstep[i] += binstep[i-1]
        else:
            binstep[i] += spec[0][i+1] - spec[0][i]
    
    photons = spec[1]
    
    area_mirr = np.pi*((0.5*mirr_diam)**2.) #collecting area
    
    wavelengths = spec[0] * 10 #converting from nm to Ang since spectrum flux comes in Ang but wavelengths come in nm 
    
    denominator = area_mirr * exposure_t * (binstep*1e-7)
    
    numerator =  photons * ((h*c) / wavelengths) #this is the photon energy portion hc/lambda
    
    fluxes = (numerator / denominator) #final calculation for photon number    

    flux_spec = np.zeros((2,len(wls)))
    flux_spec[0] += wls
    flux_spec[1] += fluxes
    
    return flux_spec




def flux_conversion_4(spec,bin_widths):
    
    wls = spec[0]
    
    photons = spec[1]
    
    area_mirr = np.pi * (0.5*mirr_diam)**2. #collecting area
    
    wavelengths = spec[0] * 10 #converting from nm to Ang since spectrum flux comes in Ang but wavelengths come in nm 
    
    denominator = area_mirr * exposure_t * (bin_widths*1e-7)
    
    numerator =  photons * ((h*c) / wavelengths) #this is the photon energy portion hc/lambda
    
    fluxes = (numerator / denominator) #final calculation for photon number    

    flux_spec = np.zeros((2,len(wls)))
    flux_spec[0] += wls
    flux_spec[1] += fluxes
    
    return flux_spec



