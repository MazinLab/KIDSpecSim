# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:00:26 2020

@author: BVH
"""

import numpy as np

from parameters import folder,lambda_low_val,lambda_high_val
from useful_funcs import nearest, wavelength_array_maker,cutoff_sorter,where_in_array,total_eff_calc
from scipy import interpolate


def resampler(order_list,spec,pixel,w_o,w_o_arm,eff,sky=False,IR=False,PTS=False):

    if sky == True:
        if IR == True:
            manip_array_misident = np.load('%s/Overlaps/misident_photons_pixel_IR_sky_%s.npy'%(folder,pixel))# for misidentified spectrum
            manip_array_order = np.load('%s/Overlaps/order_photons_pixel_IR_sky_%s.npy'%(folder,pixel))#for photon spectrum
        else:
            manip_array_misident = np.load('%s/Overlaps/misident_photons_pixel_OPT_sky_%s.npy'%(folder,pixel))# for misidentified spectrum
            manip_array_order = np.load('%s/Overlaps/order_photons_pixel_OPT_sky_%s.npy'%(folder,pixel))#for photon spectrum
    
    else:
        if IR == True:
            manip_array_misident = np.load('%s/Overlaps/misident_photons_pixel_IR_%s.npy'%(folder,pixel))# for misidentified spectrum
            manip_array_order = np.load('%s/Overlaps/order_photons_pixel_IR_%s.npy'%(folder,pixel))#for photon spectrum
        else:
            manip_array_misident = np.load('%s/Overlaps/misident_photons_pixel_OPT_%s.npy'%(folder,pixel))# for misidentified spectrum
            manip_array_order = np.load('%s/Overlaps/order_photons_pixel_OPT_%s.npy'%(folder,pixel))#for photon spectrum

    
    
    pixel_spec_ord = wavelength_array_maker(w_o) #data store for the pixel results for spec data for order photons
    pixel_spec_mis = wavelength_array_maker(w_o) #data store for the pixel results for spec data for misidentified photons
    
    
    for i in range(len(order_list)):
        
        A = nearest(pixel_spec_ord[0], w_o_arm[i,pixel],'coord') #lower bin edge index
        
        eff_over_range = eff[i,pixel]
        if eff_over_range == 0.0 or eff_over_range == 0:
            eff_over_range = 1
        
        #if mult_out[i,pixel] > 0:
            #eff_over_range = (mult_out[i,pixel]/mult_wls[i,pixel])
        #    eff_over_range = total_eff_calc(eff,w_o_arm,w_o_arm[i,pixel])
        
        #checking for NaNs
        if np.isnan(manip_array_misident[1,i]):
            manip_array_misident[1,i] = 0
            
        if np.isnan(manip_array_order[1,i]):
            manip_array_order[1,i] = 0
      
        
        resamp_mis = manip_array_misident[1,i]  #
        resamp_ord = manip_array_order[1,i]     #photons spread over the relevant spectral range (from indexes)
        
        pixel_spec_ord[1,A] += (resamp_ord )#/ eff_over_range)
        pixel_spec_mis[1,A] += (resamp_mis )#/ eff_over_range)

    return pixel_spec_ord,pixel_spec_mis



def recreator(spec,n_pixels,w_o,file_path,sky=False):
    
    rec_spec = wavelength_array_maker(w_o)
    
    mis_spec = wavelength_array_maker(w_o)

    if sky == True:
        for pixel in range(n_pixels):
            
            rec_spec[1] += np.load('%sspectrum_order_pixel_IR_sky_%i.npy'%(file_path,pixel))[1]    #looping through all pixels and adding their spectrums together to 
            mis_spec[1] += np.load('%sspectrum_misident_pixel_IR_sky_%i.npy'%(file_path,pixel))[1] #the final spectrum
            
            rec_spec[1] += np.load('%sspectrum_order_pixel_OPT_sky_%i.npy'%(file_path,pixel))[1]    #looping through all pixels and adding their spectrums together to 
            mis_spec[1] += np.load('%sspectrum_misident_pixel_OPT_sky_%i.npy'%(file_path,pixel))[1] #the final spectrum
    
    else:
        for pixel in range(n_pixels):
            
            rec_spec[1] += np.load('%sspectrum_order_pixel_IR_%i.npy'%(file_path,pixel))[1]    #looping through all pixels and adding their spectrums together to 
            mis_spec[1] += np.load('%sspectrum_misident_pixel_IR_%i.npy'%(file_path,pixel))[1] #the final spectrum
            
            rec_spec[1] += np.load('%sspectrum_order_pixel_OPT_%i.npy'%(file_path,pixel))[1]    #looping through all pixels and adding their spectrums together to 
            mis_spec[1] += np.load('%sspectrum_misident_pixel_OPT_%i.npy'%(file_path,pixel))[1] #the final spectrum
        
    low = nearest(rec_spec[0],lambda_low_val,'coord')
    high = nearest(rec_spec[0],lambda_high_val,'coord')

    perc_misident_pp = np.nanmedian(mis_spec[1][low:high] / (rec_spec[1][low:high]+mis_spec[1][low:high])) * 100 #to ignore NaNs
    perc_misident_tot = (np.sum(np.nan_to_num(mis_spec[1][low:high])) / (np.sum(np.nan_to_num(rec_spec[1][low:high]))+np.sum(np.nan_to_num(mis_spec[1][low:high])))) * 100
    no_misident = np.sum(np.abs(mis_spec[1]))

    
    spec_low_coord = nearest(rec_spec[0],spec[0][0],'coord')+1 #trimming spectrum to the data spectrum range
    spec_high_coord = nearest(rec_spec[0],spec[0][-1],'coord')-1
    
    rec_spec = rec_spec[:,spec_low_coord:spec_high_coord]
    mis_spec = mis_spec[:,spec_low_coord:spec_high_coord]

    for i in range(len(rec_spec[1])):
        if rec_spec[1][i] < 0.0:
            rec_spec[1][i] = 0.0

    return rec_spec,mis_spec,perc_misident_pp,perc_misident_tot,no_misident





