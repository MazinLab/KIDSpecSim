# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:07:55 2020

@author: BVH
"""


import numpy as np
from matplotlib import pyplot as plt

from useful_funcs import gaussian_eq
from parameters import ER_band_low,lambda_low_val,lambda_high_val,pix_mult






def apply_gaussian(order_list,w_o,spec,pixel,pixel_sums,eff,pixel_R_Es,generate_R_Es=False,IR=False,plot=False,dual_arm=True):

    pix_gauss = np.zeros((len(order_list),len(spec[1]))) #array to store gaussian for this pixel
    
    for i in range(len(order_list)):
        
        mu = w_o[i,pixel] #mean of gaussian for this order
        
        if generate_R_Es == False and IR == False:
            rand_R_E = np.load('R_E_PIXELS/R_E_PIXELS_OPT.npy')[int(pixel)]
        elif generate_R_Es == False and IR == True:
            rand_R_E = np.load('R_E_PIXELS/R_E_PIXELS_IR.npy')[int(pixel)]
        else:
            if pixel_R_Es[int(pixel)] != 0:
                rand_R_E = pixel_R_Es[int(pixel)]
            else:
                rand_R_E = np.random.normal(loc=ER_band_low,scale=3)
                pixel_R_Es[int(pixel)] += rand_R_E
        
        if pixel_R_Es[int(pixel)] == 0:
            pixel_R_Es[int(pixel)] += rand_R_E
        
        if dual_arm==True:
            KID_R =  rand_R_E / np.sqrt(mu/400)
        else:
            KID_R =  rand_R_E / (mu/400)
        
        sig = mu / (KID_R * (2*np.sqrt( 2*np.log(2) )))  #using an equation from 'GIGA-Z: A 100,000 OBJECT SUPERCONDUCTING SPECTROPHOTOMETER FOR LSST FOLLOW-UP' 
        
        gaussian = gaussian_eq(spec[0],mu,sig) #creating gaussian
        
        normalise_fac = np.sum(gaussian) 
        
        if normalise_fac < 1e-15: #just in case the normalisation factor is very small
            normalise_fac = 1                                                           #the normalise and noise factors return the gaussian to a actual photon count
                                                                                         #so that the sum of the gaussian returns the pixel_sum value
        noise =  pixel_sums[pixel,i]#np.random.poisson(lam = pixel_sums[pixel,i]) #adding photon shot noise 
        
        gaussian = gaussian*(noise/normalise_fac) #applying noise and normalisation factor
    
        pix_gauss[i,:] += gaussian #adding to pixel's array
        
        
        
    if plot == True and (pixel == pix_mult).any():
        
        
        
        fig = plt.figure()
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Intensity')
        fig.text(0.14,0.84,'Pixel: %i'%pixel)
        
        for i in range(len(order_list)):

            plt.plot(spec[0],pix_gauss[i],'-',label='Order %i'%order_list[i])

        plt.legend(loc='best')

    


    return pix_gauss,pixel_R_Es


