# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:45:14 2021

@author: BVH
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import os

from useful_funcs import gaussian_eq,nearest,wavelength_array_maker
from parameters import ER_band_low,lambda_low_val,lambda_high_val,pix_mult,folder,r_e_spread,ER_band_wl,IR_arm


#####################################################################################
#APPLYING THE GAUSSIAN TO EACH ORDER ON AN MKID
######################################################################################

def apply_gaussian(order_list,w_o,spec,pixel,pixel_sums,pixel_R_Es,generate_R_Es=False,IR=False,plot=False,dual_arm=True):

    pix_gauss = np.zeros((len(order_list),10000)) #array to store gaussian for this pixel
    
    wavelength_range = np.linspace(lambda_low_val,lambda_high_val,10000)
    
    #looping through each order for the current MKID
    for i in range(len(order_list)):
        
        mu = w_o[i,pixel] #mean of gaussian for this order
        
        #loading any MKID energy resolution variances

        if generate_R_Es == False:
            rand_R_E = ER_band_low
        else:
            if pixel_R_Es[int(pixel)] != ER_band_low:
                rand_R_E = pixel_R_Es[int(pixel)]
            else:
                rand_R_E = np.random.normal(loc=ER_band_low,scale=3)
                pixel_R_Es[int(pixel)] = rand_R_E
                if IR == False:
                    np.save('R_E_PIXELS/R_E_PIXELS_OPT.npy',pixel_R_Es)
                else:
                    np.save('R_E_PIXELS/R_E_PIXELS_IR.npy',pixel_R_Es)
                    
        if pixel_R_Es[int(pixel)] == 0:
            pixel_R_Es[int(pixel)] += rand_R_E
        
        if dual_arm==True:
            KID_R =  rand_R_E / np.sqrt(mu/ER_band_wl)
        else:
            KID_R =  rand_R_E / (mu/ER_band_wl)
        
        #defining gaussian 1sigma
        sig = mu / (KID_R * (2*np.sqrt( 2*np.log(2) )))  #using an equation from 'GIGA-Z: A 100,000 OBJECT SUPERCONDUCTING SPECTROPHOTOMETER FOR LSST FOLLOW-UP' 
        
        gaussian = gaussian_eq(wavelength_range,mu,sig) #creating gaussian
        
        normalise_fac = np.sum(gaussian) 
        
        if normalise_fac < 1e-15: #just in case the normalisation factor is very small
            normalise_fac = 1                                                           #the normalise and noise factors return the gaussian to a actual photon count
                                                                                         #so that the sum of the gaussian returns the pixel_sum value
        incoming_photons =  pixel_sums[pixel,i]
        
        gaussian = np.copy(gaussian)*(incoming_photons/normalise_fac) #applying noise and normalisation factor
    
        pix_gauss[i,:] += gaussian #adding to pixel's array
        
        
        
    if plot == True: #and (pixel == pix_mult).any():
        
        
        
        fig = plt.figure()
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Intensity')
        fig.text(0.14,0.84,'Pixel: %i'%pixel)
        
        for i in range(len(order_list)):

            plt.plot(wavelength_range,pix_gauss[i],'-',label='Order %i'%order_list[i])

        plt.legend(loc='best')



    return pix_gauss,pixel_R_Es


######################################################################################################
#SORTING ANY OVERLAP BETWEEN ORDER GAUSSIANS
######################################################################################################

def gaussian_overlaps(gaussian,pixel,order_list,w_o,spec,plotting=False):
    
    total_response = np.sum(gaussian,axis=0)

    max_vals = np.max(gaussian,axis=0)
    
    wavelength_range = np.linspace(lambda_low_val,lambda_high_val,10000)
    
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
        
        x_values = np.ma.array(wavelength_range,mask=coincidence)
        
        
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


######################################################################################################
#RESAMPLING TO 1D SPECTRUM
######################################################################################################


def resampler(manip_array_order,manip_array_misident,order_list,pixel,w_o,w_o_arm):
    
    pixel_spec_ord = wavelength_array_maker(w_o) #data store for the pixel results for spec data for order photons
    pixel_spec_mis = wavelength_array_maker(w_o) #data store for the pixel results for spec data for misidentified photons
    
    
    for i in range(len(order_list)):
        
        A = nearest(pixel_spec_ord[0], w_o_arm[i,pixel],'coord') #lower bin edge index
        
        #checking for NaNs
        if np.isnan(manip_array_misident[1,i]):
            manip_array_misident[1,i] = 0
            
        if np.isnan(manip_array_order[1,i]):
            manip_array_order[1,i] = 0
      
        
        resamp_mis = manip_array_misident[1,i]  #
        resamp_ord = manip_array_order[1,i]     #photons spread over the relevant spectral range (from indexes)
        
        pixel_spec_ord[1,A] += (resamp_ord )
        pixel_spec_mis[1,A] += (resamp_mis )

    return pixel_spec_ord,pixel_spec_mis




######################################################################################################
#USING PREVIOUS FUNCTIONS TO SIMULATE MKID RESPONSE
######################################################################################################


def MKID_response(spec,order_list,w_o,w_o_arm,n_pixels,pixel_sums,
                  IR=False,sky=False,dual_arm_=True,make_folder=False):
    
    if make_folder == True:
        os.mkdir('%s/Resample/'%(folder))
    file_path_resample = '%s/Resample/'%(folder)
    
    if r_e_spread == True and IR == False:
        pixel_R_Es = np.load('R_E_PIXELS/R_E_PIXELS_OPT.npy')
    elif r_e_spread == True and IR == True:
        pixel_R_Es = np.load('R_E_PIXELS/R_E_PIXELS_IR.npy')
    else:
        pixel_R_Es = np.ones(n_pixels)*ER_band_low
    
    int_steps = np.ndarray.astype((np.linspace(0,n_pixels,100)),dtype='int') 
    prog = 0
    
    print('\n')
    for pixel in range(n_pixels): #loop applies gaussian, finds any overlap, then resamples to 1d spectrum format, then saves
    
        #if pixel == 3479:
        #    pix_gauss,pixel_R_Es = apply_gaussian(order_list,w_o,spec,pixel,pixel_sums,pixel_R_Es,generate_R_Es=r_e_spread,IR=IR,plot=True,dual_arm=dual_arm_)
        #    pixel_spec_ord,pixel_spec_mis = gaussian_overlaps(pix_gauss,pixel,order_list,w_o,spec,plotting=True)
        #else:
        pix_gauss,pixel_R_Es = apply_gaussian(order_list,w_o,spec,pixel,pixel_sums,pixel_R_Es,generate_R_Es=r_e_spread,IR=IR,plot=False,dual_arm=dual_arm_)
        pixel_spec_ord,pixel_spec_mis = gaussian_overlaps(pix_gauss,pixel,order_list,w_o,spec,plotting=False)
        
        #saving resampled result
        if IR == True and sky == False:
            np.save('%s/spectrum_order_pixel_IR_%i.npy'%(file_path_resample,pixel),pixel_spec_ord)
            np.save('%s/spectrum_misident_pixel_IR_%i.npy'%(file_path_resample,pixel),pixel_spec_mis)
        
        if IR == True and sky == True:
            np.save('%s/spectrum_order_pixel_IR_sky_%i.npy'%(file_path_resample,pixel),pixel_spec_ord)
            np.save('%s/spectrum_misident_pixel_IR_sky_%i.npy'%(file_path_resample,pixel),pixel_spec_mis)

        if IR == False and sky == False:
            np.save('%s/spectrum_order_pixel_OPT_%i.npy'%(file_path_resample,pixel),pixel_spec_ord)
            np.save('%s/spectrum_misident_pixel_OPT_%i.npy'%(file_path_resample,pixel),pixel_spec_mis)
        
        if IR == False and sky == True:
            np.save('%s/spectrum_order_pixel_OPT_sky_%i.npy'%(file_path_resample,pixel),pixel_spec_ord)
            np.save('%s/spectrum_misident_pixel_OPT_sky_%i.npy'%(file_path_resample,pixel),pixel_spec_mis)
    
    
        if (pixel == int_steps).any():
            prog += 1
            print('\r%i%% of pixels complete.'%prog,end='',flush=True)
    
    if r_e_spread == True:
        if IR == True:
            np.save('R_E_PIXELS/R_E_PIXELS_IR.npy',pixel_R_Es)
        else:
            np.save('R_E_PIXELS/R_E_PIXELS_OPT.npy',pixel_R_Es)
    
    return

def MKID_response_V2(spec,order_list,w_o,w_o_arm,n_pixels,pixel_sums,
                  IR=False,sky=False,dual_arm_=True):
    
    
    if r_e_spread == True and IR == False:
        pixel_R_Es = np.load('R_E_PIXELS/R_E_PIXELS_OPT.npy')
    elif r_e_spread == True and IR == True:
        pixel_R_Es = np.load('R_E_PIXELS/R_E_PIXELS_IR.npy')
    else:
        pixel_R_Es = np.ones(n_pixels)*ER_band_low
    
    int_steps = np.ndarray.astype((np.linspace(0,n_pixels,100)),dtype='int') 
    prog = 0
    
    resp_grid = np.zeros_like(w_o_arm)
    resp_grid_mis = np.zeros_like(w_o_arm)
    
    print('\n')
    for pixel in range(n_pixels): #loop applies gaussian, finds any overlap, then resamples to 1d spectrum format, then saves
    
        pix_gauss,pixel_R_Es = apply_gaussian(order_list,w_o,spec,pixel,pixel_sums,pixel_R_Es,generate_R_Es=r_e_spread,IR=IR,plot=False,dual_arm=dual_arm_)
        pixel_spec_ord,pixel_spec_mis = gaussian_overlaps(pix_gauss,pixel,order_list,w_o,spec,plotting=False)
        
        if IR_arm == True and IR == True:
            resp_grid[:,pixel] += pixel_spec_ord[1]
            resp_grid_mis[:,pixel] += pixel_spec_mis[1]
        elif IR_arm == False and IR == False:
            resp_grid[:,pixel] += pixel_spec_ord[1]
            resp_grid_mis[:,pixel] += pixel_spec_mis[1]
    
        if (pixel == int_steps).any():
            prog += 1
            print('\r%i%% of pixels complete.'%prog,end='',flush=True)
    
    if r_e_spread == True:
        if IR == True:
            np.save('R_E_PIXELS/R_E_PIXELS_IR.npy',pixel_R_Es)
        else:
            np.save('R_E_PIXELS/R_E_PIXELS_OPT.npy',pixel_R_Es)
    
    return resp_grid,resp_grid_mis




######################################################################################################################
#RECREATING SPECTRUM BY TOTALLING ALL MKID PIXEL RESPONSES
#################################################################################################################################



def recreator(spec,n_pixels,w_o,order_list_ir,sky=False):
    
    file_path = '%s/Resample/'%(folder)
    
    rec_spec = np.zeros_like(w_o)
    
    mis_spec = np.zeros_like(w_o)
    
    if order_list_ir[0] == 200:
        ord_arm_sep = 0
    else:
        ord_arm_sep = len(order_list_ir)

    if sky == True:
        for pixel in range(n_pixels):
            
            rec_spec[:ord_arm_sep,pixel] += np.load('%sspectrum_order_pixel_IR_sky_%i.npy'%(file_path,pixel))[1]    #looping through all pixels and adding their spectrums together to 
            mis_spec[:ord_arm_sep,pixel] += np.load('%sspectrum_misident_pixel_IR_sky_%i.npy'%(file_path,pixel))[1] #the final spectrum
            
            rec_spec[ord_arm_sep:,pixel] += np.load('%sspectrum_order_pixel_OPT_sky_%i.npy'%(file_path,pixel))[1]    #looping through all pixels and adding their spectrums together to 
            mis_spec[ord_arm_sep:,pixel] += np.load('%sspectrum_misident_pixel_OPT_sky_%i.npy'%(file_path,pixel))[1] #the final spectrum
    
    else:
        for pixel in range(n_pixels):
            rec_spec[:ord_arm_sep,pixel] += np.load('%sspectrum_order_pixel_IR_%i.npy'%(file_path,pixel))[1]    #looping through all pixels and adding their spectrums together to 
            mis_spec[:ord_arm_sep,pixel] += np.load('%sspectrum_misident_pixel_IR_%i.npy'%(file_path,pixel))[1] #the final spectrum
            
            rec_spec[ord_arm_sep:,pixel] += np.load('%sspectrum_order_pixel_OPT_%i.npy'%(file_path,pixel))[1]    #looping through all pixels and adding their spectrums together to 
            mis_spec[ord_arm_sep:,pixel] += np.load('%sspectrum_misident_pixel_OPT_%i.npy'%(file_path,pixel))[1] #the final spectrum

    perc_misident_pp = np.nanmedian(np.sum(np.nan_to_num(mis_spec),axis=0) / (np.sum(np.nan_to_num(rec_spec),axis=0)+np.sum(np.nan_to_num(mis_spec),axis=0))) * 100 #to ignore NaNs, average misidentfied photons
    perc_misident_tot = (np.sum(np.nan_to_num(mis_spec)) / (np.sum(np.nan_to_num(rec_spec))+np.sum(np.nan_to_num(mis_spec)))) * 100
    no_misident_photons = np.sum(np.abs(np.nan_to_num(mis_spec)))

    return rec_spec,np.nan_to_num(mis_spec),perc_misident_pp,perc_misident_tot,no_misident_photons


def recreator_one_arm(spec,n_pixels,w_o,order_list_ir,sky=False):
    
    file_path = '%s/Resample/'%(folder)
    
    rec_spec = np.zeros_like(w_o)
    
    mis_spec = np.zeros_like(w_o)
    
    if order_list_ir[0] == 200:
        ord_arm_sep = 0
    else:
        ord_arm_sep = len(order_list_ir)

    if sky == True:
        for pixel in range(n_pixels):
            
            rec_spec[ord_arm_sep:,pixel] += np.load('%sspectrum_order_pixel_OPT_sky_%i.npy'%(file_path,pixel))[1]    #looping through all pixels and adding their spectrums together to 
            mis_spec[ord_arm_sep:,pixel] += np.load('%sspectrum_misident_pixel_OPT_sky_%i.npy'%(file_path,pixel))[1] #the final spectrum
    
    else:
        for pixel in range(n_pixels):
            rec_spec[ord_arm_sep:,pixel] += np.load('%sspectrum_order_pixel_OPT_%i.npy'%(file_path,pixel))[1]    #looping through all pixels and adding their spectrums together to 
            mis_spec[ord_arm_sep:,pixel] += np.load('%sspectrum_misident_pixel_OPT_%i.npy'%(file_path,pixel))[1] #the final spectrum

    perc_misident_pp = np.nanmedian(mis_spec[1] / (rec_spec[1]+mis_spec[1])) * 100 #to ignore NaNs, average misidentfied photons
    perc_misident_tot = (np.sum(np.nan_to_num(mis_spec)) / (np.sum(np.nan_to_num(rec_spec))+np.sum(np.nan_to_num(mis_spec)))) * 100
    no_misident_photons = np.sum(np.abs(np.nan_to_num(mis_spec)))

    return rec_spec,np.nan_to_num(mis_spec),perc_misident_pp,perc_misident_tot,no_misident_photons















