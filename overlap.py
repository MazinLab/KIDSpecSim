# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:16:30 2020

@author: BVH
"""


import numpy as np
from matplotlib import pyplot as plt 

from parameters import folder


def gaussian_overlaps(pixel,order_list,w_o,spec,sky=False,IR=False,plotting=False):
    
    if sky == True:
        if IR == True:
            gaussian = np.load('%s/Pixel_Gaussians/gaussian_pixel_IR_sky_%s.npy'%(folder,pixel)) #loading in the gaussian for that pixel
        else:
            gaussian = np.load('%s/Pixel_Gaussians/gaussian_pixel_OPT_sky_%s.npy'%(folder,pixel))
    else:
        if IR == True:
            gaussian = np.load('%s/Pixel_Gaussians/gaussian_pixel_IR_%s.npy'%(folder,pixel)) #loading in the gaussian for that pixel
        else:
            gaussian = np.load('%s/Pixel_Gaussians/gaussian_pixel_OPT_%s.npy'%(folder,pixel)) #loading in the gaussian for that pixel
        
    total_response = np.sum(gaussian,axis=0)

    max_vals = np.max(gaussian,axis=0)
    
    
    #creating data arrays to hold the results for the number of photons on a pixel from a given order, and the misidentified photon count
    #holds wavelength (0) and photons (1)
    
    order_photons = np.zeros((2,len(order_list)))
    misident_photons = np.zeros((2,len(order_list)))

    if plotting == True:
        fig = plt.figure()
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Intensity')
        fig.text(0.14,0.84,'Pixel: %i'%pixel)
        
        
    for i in range(len(order_list)): #cycling through relevant orders
        
        coincidence = np.logical_not(np.isin(max_vals,gaussian[i])) #checking where the max values for each column occur across the order row
        #needed the logical not since ma.array sees True as invalid data 
        #returns the coordinates of the values in the order row which are the maximum in each column 

        masked_total_response = np.ma.array(total_response,mask=coincidence) #applying the coordinates to the total sum of each column, 
                                                                        #returns array of values which correspond to the coordinates previously found
        
        x_values = np.ma.array(spec[0],mask=coincidence)
        
        
        order_photons[0,i] = w_o[i,pixel] #wavelengths
        order_photons[1,i] = np.sum(masked_total_response)
        
        if np.isnan(order_photons[1,i]).any() or order_photons[1,i] < 0.0001: #checking for NaNs and partial photon counts
            order_photons[1,i] = 0                                  #since some gaussians are smaller than others in size
        
        
        #now sorting out the misidentified photons array
        misident_photons[0,i] = w_o[i,pixel]
        misident_photons[1,i] = order_photons[1,i] - np.sum( np.ma.array(gaussian[i],mask=coincidence) ) #order response - masked contribution from current order
                                                                                                    #note it is not summed here, which raises the comparison
                                                                                                    
        
        if plotting == True:
            plt.plot(x_values,masked_total_response,'-',label='Order %i'%order_list[i])
    
    
    
    if plotting == True:
        plt.legend(loc='best')
    
    return order_photons,misident_photons




