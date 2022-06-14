# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:49:30 2021

@author: BVH
"""

import numpy as np 
from matplotlib import pyplot as plt
from scipy import interpolate
import time

from parameters import *
from useful_funcs import where_in_array,nearest 
                       
                            


def grating_orders_2_arms(model,cutoff,plotting=False):
    
    #<cutoffnm is OPT
    #>cutoffnm is IR
    #cutoff = 2500 #1000
    
    if model not in eff_models: #making sure the efficiency model exists in the list of supported models
        raise ValueError('Model must be one of ', eff_models)
    
    #getting variables from parameter file
    lambda_low = lambda_low_val #lambda = wavelength
    lambda_high = lambda_high_val #these two variables make up the wavelength range that will be covered
    alpha = alpha_val * (np.pi / 180) #angle of incidence
    phi = phi_val * (np.pi / 180) #grating blaze angle
    refl_start = refl_start_val * (np.pi / 180) #reflected angle (lowest)(end is highest)
    refl_end = refl_end_val * (np.pi / 180)  #converting to radians
    alpha_IR = IR_alpha * (np.pi / 180) #angle of incidence
    phi_IR = IR_phi * (np.pi / 180) #grating blaze angle    
    refl_start_ir = refl_start_val_ir * (np.pi / 180)
    refl_end_ir = refl_end_val_ir * (np.pi / 180)

    reflected_angles = np.linspace(refl_start,refl_end,n_pixels) #angles which will fall on each pixel
    reflected_angles_ir = np.linspace(refl_start_ir,refl_end_ir,n_pixels) #angles which will fall on each pixel
    
    order_list_opt = [] #orders which fall in the required range will be added to this list
    wavelengths_orders_opt = [] #their wavelengths will be added here
    
    order_list_ir = [] #orders which fall in the required range will be added to this list
    wavelengths_orders_ir = [] #their wavelengths will be added here
    
    
    
    for i in range(1,int(norders)):
        
        wavelengths_in_order_OPT = (((np.sin(reflected_angles) + np.sin(np.abs(alpha))) * OPT_a) / (i)) * 1000 #grating equation, 1000 is for groove conversion to nm
        wavelengths_in_order_IR = (((np.sin(reflected_angles_ir) + np.sin(np.abs(alpha_IR))) * IR_a) / (i)) * 1000 #grating equation, 1000 is for groove conversion to nm
        
        waves_in_order_OPT = len(list(x for x in wavelengths_in_order_OPT if lambda_low < x < cutoff and x < lambda_high)) #checking if any wavelengths fall in the range setout in parameters
        waves_in_order_IR = len(list(x for x in wavelengths_in_order_IR if cutoff < x < lambda_high and x > lambda_low)) #checking if any wavelengths fall in the range setout in parameters
        
        if waves_in_order_OPT != 0:
            order_list_opt.append(i)
            wavelengths_in_order_OPT = np.round(np.asarray(wavelengths_in_order_OPT),decimals=4)
            wavelengths_orders_opt.append(np.linspace(wavelengths_in_order_OPT[0],wavelengths_in_order_OPT[-1],len(wavelengths_in_order_OPT)))#wavelengths_in_order_OPT) #adding relevant orders and their wavelengths to lists
        
        if waves_in_order_IR != 0:
            order_list_ir.append(i)
            wavelengths_in_order_IR = np.round(np.asarray(wavelengths_in_order_IR),decimals=4)
            wavelengths_orders_ir.append(np.linspace(wavelengths_in_order_IR[0],wavelengths_in_order_IR[-1],len(wavelengths_in_order_IR))) #adding relevant orders and their wavelengths to lists
    
    wavelengths_orders_opt = np.asarray(wavelengths_orders_opt)# np.round(np.asarray(wavelengths_orders_opt),decimals=2) #number of decimals here depends on how much can be resolved
    order_list_opt = np.asarray(order_list_opt)
    
    wavelengths_orders_ir = np.asarray(wavelengths_orders_ir) #number of decimals here depends on how much can be resolved
    order_list_ir = np.asarray(order_list_ir)
    
    order_list = np.concatenate((order_list_ir,order_list_opt))
    
    if len(wavelengths_orders_ir) > 0:
        wavelengths_orders = np.concatenate((wavelengths_orders_ir,wavelengths_orders_opt))
    else:
        wavelengths_orders = wavelengths_orders_opt
        
    #plotting order ranges
    
    if plotting == True:
        
        plt.figure()
        if len(order_list_opt) != 0:
            for i in range(len(order_list_opt)):
                y = np.ones(len(wavelengths_orders_opt[i,:]))*order_list_opt[i]
                plt.plot(wavelengths_orders_opt[i,:],y,'-',label='Order %i'%order_list_opt[i])
        
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Order')
        plt.legend(loc='best',fontsize=11)
        if len(wavelengths_orders_opt) > 0:
            plt.text(np.min(wavelengths_orders_opt),np.min(order_list_opt),'Optical Arm',fontsize=11)
        #plt.title('Spread of orders across the optical wavelength ranges of the spectrum')
       
        plt.figure()
        
        if len(order_list_ir) != 0:
            for i in range(len(order_list_ir)):
                y = np.ones(len(wavelengths_orders_ir[i,:]))*order_list_ir[i]
                plt.plot(wavelengths_orders_ir[i,:],y,'-',label='Order %i'%order_list_ir[i])
        
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Order')
        plt.legend(loc='best',fontsize=11)
        if len(wavelengths_orders_ir) > 0:
            plt.text(np.min(wavelengths_orders_ir),np.min(order_list_ir),'Infrared Arm',fontsize=11)
        #plt.title('Spread of orders across the IR wavelength ranges of the spectrum')
    
    #calculating grating efficiency and plotting
    
    efficiencies = np.zeros_like(wavelengths_orders) #data array to store efficiencies 
    
    if plotting == True: #plotting code
        plt.figure()
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Efficiency')
        #plt.title('Grating efficiencies across each order using a %s model'%model)    
        
    for i in range(len(order_list)): #looping through orders to calculate each's efficiency
        
        if i < len(order_list_ir): #(order_list[i] == order_list_ir).any():
            alpha_eff = alpha_IR
            phi_eff = phi_IR
            refl_eff = reflected_angles_ir
        else:
            alpha_eff = alpha
            phi_eff = phi
            refl_eff = reflected_angles           
        
        
        if model == 'Casini':     #MAINLY USED
                
            
            
            sec_first = (order_list[i])*( np.cos(alpha_eff)/np.cos(alpha_eff-phi_eff) ) #The paper states there should be pi multiplied by the order o
            sec_second = (np.sin(alpha_eff-phi_eff) + np.sin(refl_eff-phi_eff)) #this is not done here because on python sinc(x) = sin(pi*x) / pi*x 
            
            eff_i = (np.sinc(sec_first*sec_second)**2)*0.84
                                
            #adding current orders results to array storing all order efficiencies
            for j in range(len(eff_i)):
                if eff_i[j] < 1e-20:
                    efficiencies[i,j] = 0.0
                else:
                    efficiencies[i,j] = eff_i[j]
            
            efficiencies_ir = efficiencies[:len(order_list_ir),:]
            efficiencies_opt = efficiencies[len(order_list_ir):,:] 
    
    
        if plotting == True:
            plt.plot(wavelengths_orders[i,:],eff_i,'-',label='Order %i'%order_list[i])
    
    if plotting == True:
        plt.legend(loc='best',fontsize=11)
    
    
    return order_list, wavelengths_orders,efficiencies,order_list_opt,wavelengths_orders_opt, \
        efficiencies_opt,order_list_ir,wavelengths_orders_ir,efficiencies_ir
    
    
    
    
    
def grating_binning(spec,w_o,all_w_o,order_list,eff,cutoff,IR=False,OPT=False,plotting=False): #note: w_o here is wavelengths_orders
    
    zeroed_wls = []
    
    mult_out = np.zeros_like(w_o)
    mult_wls = np.zeros_like(w_o)
    pixel_sums = np.zeros((n_pixels,len(order_list))) #storing photons each pixel sees
    
    #interpolation for model spectra, binstep altering is for the correct photon number after interpolation
    binstep_interpolation = ((spec[0][-1]-spec[0][0]) / 100000)
    binstep_factor = (binstep/1e-7) / binstep_interpolation
    f_int = interpolate.interp1d(spec[0], spec[1]/binstep_factor, bounds_error = False, fill_value = 0) #interpolating data
    
    spec_high_R = np.zeros((2,100000))
    spec_high_R[0] += np.linspace(spec[0][0],spec[0][-1],100000)
    spec_high_R[1] += f_int(spec_high_R[0])

    
    if IR == True:
        spec_high_R = spec_high_R[:,spec_high_R[0] >= cutoff+1]
    else:
        spec_high_R = spec_high_R[:,spec_high_R[0] <= cutoff+1]
    
    #this part takes the range of values around each wavelength in `order_wavelengths` as a bin
    w_o_bins = np.zeros((len(w_o[:]),n_pixels+1)) #was +1
    for i in range(len(w_o[:])):
        for j in range(len(w_o[i])):
            if j == 0:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j+1]-w_o[i,j])/2)
            elif j == len(w_o[i])-1:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j]-w_o[i,j-1])/2)
                w_o_bins[i][j+1] += w_o[i,j] + ((w_o[i,j]-w_o[i,j-1])/2)
            else:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j]-w_o[i,j-1])/2)
    
    #binning photons onto MKID order wavelength bins grid
    if len(spec_high_R[0]) > 0:
        print('\n > Beginning placement of incoming photons.')
        int_steps = np.ndarray.astype((np.linspace(0,len(spec_high_R[0]),10)),dtype='int') 
        prog = 0
        
        for i in range(len(spec_high_R[0])):
            
            coord_bins = []
            
            for j in range(len(w_o_bins[:])):
                coords_low = np.where(spec_high_R[0][i] > w_o_bins[j])[0]
                coords_high = np.where(spec_high_R[0][i] <= w_o_bins[j])[0]
                if len(coords_low) > 0 and len(coords_high) > 0:
                    coord_bins.append(np.array([j,coords_low[-1]]))
                
            if len(coord_bins) > 0:
                factor = len(coord_bins)
                tot_eff = 0
                for j in range(factor):
                    pixel_sums[coord_bins[j][1],coord_bins[j][0]] += (spec_high_R[1][i])*eff[coord_bins[j][0],coord_bins[j][1]]
                    tot_eff += eff[coord_bins[j][0],coord_bins[j][1]]
                for j in range(factor):
                    mult_wls[coord_bins[j][0],coord_bins[j][1]] += 1
                    mult_out[coord_bins[j][0],coord_bins[j][1]] += tot_eff
                
            
            if (i == int_steps).any():
                prog += 10
                print('\r%i%% of wavelengths complete.'%prog,end='',flush=True)
            
        #applying photon noise
        print('\n > Accounting for photon noise')
        for i in range(len(pixel_sums[0,:])):
            for j in range(len(pixel_sums[:,0])):
                try:
                    pixel_sums[j,i] = np.random.poisson(lam=pixel_sums[j,i])
                except:
                    pixel_sums[j,i] = np.random.normal(loc=pixel_sums[j,i],scale=np.sqrt(pixel_sums[j,i]))
                if eff[i,j] < 5e-10:
                    pixel_sums[j,i] = 0
                    
            pixel_sums[pixel_sums[:,i]<1.0,i] = 0
            zeroed_wls.append(w_o[i,:][pixel_sums[:,i]==0])
            
            
            if IR == True:
                pixel_sums[w_o[i,:] < cutoff,i] = 0
            else:
                pixel_sums[w_o[i,:] > cutoff,i] = 0
            
            extra_zero_wls = []
            for i in range(n_pixels):
                max_ph = np.max(pixel_sums[i])
                for j in range(len(pixel_sums[i])):
                    if (pixel_sums[i][j]/max_ph) < 5e-20: #was 5e-5
                        extra_zero_wls.append(w_o[j,i])
            zeroed_wls.append(np.asarray(extra_zero_wls))
        
        if plotting == True:
            plt.figure()
            for i in range(len(w_o[:,0])):
                plt.plot(w_o[i],pixel_sums[:,i],label='Order %i'%order_list[i])
            plt.xlabel('Wavelength / nm')
            plt.ylabel('Photon count')
            plt.legend(loc='best')
    
    return pixel_sums,zeroed_wls,w_o_bins

def grating_binning_high_enough_R(spec,w_o,all_w_o,order_list,eff,cutoff,IR=False,OPT=False,plotting=False): #note: w_o here is wavelengths_orders
    
    zeroed_wls = []
    
    mult_out = np.zeros_like(w_o)
    mult_wls = np.zeros_like(w_o)
    
    pixel_sums = np.zeros((n_pixels,len(order_list))) #storing photons each pixel sees
    pixel_sums_count = np.zeros_like(pixel_sums)
    spec_high_R = np.copy(spec)
    
    if dead_pixel_perc > 0.0:
        
        if IR == True:
            try:
                dead_pixels = np.load('DEAD_PIXEL_LISTS/DEAD_PIXELS_IR_ARM.npy')
            except:
                no_pixels_dead = int(n_pixels * (dead_pixel_perc/100))
                dead_pixels = np.random.randint(0,n_pixels,no_pixels_dead)
                np.save('DEAD_PIXEL_LISTS/DEAD_PIXELS_IR_ARM.npy',dead_pixels)
        else:
            try:
                dead_pixels = np.load('DEAD_PIXEL_LISTS/DEAD_PIXELS_OPT_ARM.npy')
            except:
                no_pixels_dead = int(n_pixels * (dead_pixel_perc/100))
                dead_pixels = np.random.randint(0,n_pixels,no_pixels_dead)
                np.save('DEAD_PIXEL_LISTS/DEAD_PIXELS_OPT_ARM.npy',dead_pixels)
                if IR_arm == False:
                    np.save('DEAD_PIXEL_LISTS/DEAD_PIXELS_IR_ARM.npy',np.zeros(no_pixels_dead))
                
    
    
    if IR == True:
        spec_high_R = spec_high_R[:,spec_high_R[0] >= cutoff+1]
    else:
        spec_high_R = spec_high_R[:,spec_high_R[0] <= cutoff+1]
    
    wave_no = len(spec_high_R[0])
    
    #this part takes the range of values around each wavelength in `order_wavelengths` as a bin
    w_o_bins = np.zeros((len(w_o[:]),n_pixels+1)) #was +1
    for i in range(len(w_o[:])):
        for j in range(len(w_o[i])):
            if j == 0:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j+1]-w_o[i,j])/2)
            elif j == len(w_o[i])-1:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j]-w_o[i,j-1])/2)
                w_o_bins[i][j+1] += w_o[i,j] + ((w_o[i,j]-w_o[i,j-1])/2)
            else:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j]-w_o[i,j-1])/2)
    
    #binning photons onto MKID order wavelength bins grid
    if len(spec_high_R[0]) > 0:
        print('\n > Beginning placement of incoming photons.')
        int_steps = np.ndarray.astype((np.linspace(0,len(spec_high_R[0]),10)),dtype='int') 
        prog = 0
        
        for i in range(len(spec_high_R[0])):
            
            coord_bins = []
            
            for j in range(len(w_o_bins[:])):
                coords_low = np.where(spec_high_R[0][i] > w_o_bins[j])[0]
                coords_high = np.where(spec_high_R[0][i] <= w_o_bins[j])[0]
                if len(coords_low) > 0 and len(coords_high) > 0:
                    coord_bins.append(np.array([j,coords_low[-1]]))
                
            if len(coord_bins) > 0:
                factor = len(coord_bins)
                tot_eff = 0
                for j in range(factor):
                    pixel_sums[coord_bins[j][1],coord_bins[j][0]] += (spec_high_R[1][i])*eff[coord_bins[j][0],coord_bins[j][1]]
                    pixel_sums_count[coord_bins[j][1],coord_bins[j][0]] += 1
                    tot_eff += eff[coord_bins[j][0],coord_bins[j][1]]
                for j in range(factor):
                    mult_wls[coord_bins[j][0],coord_bins[j][1]] += 1
                    mult_out[coord_bins[j][0],coord_bins[j][1]] += tot_eff
                

            if (i == int_steps).any():
                prog += 10
                print('\r%i%% of wavelengths complete.'%prog,end='',flush=True)

            
        #applying photon noise
        print('\n > Accounting for photon noise')
        for i in range(len(pixel_sums[0,:])):
            for j in range(len(pixel_sums[:,0])):
                if pixel_sums[j,i] < 1:
                    pixel_sums[j,i] = 0
                    zeroed_wls.append(w_o[i,j])
                try:
                    pixel_sums[j,i] = np.random.poisson(lam=pixel_sums[j,i])
                except:
                    pixel_sums[j,i] = np.random.normal(loc=pixel_sums[j,i],scale=np.sqrt(pixel_sums[j,i]))
                if eff[i,j] < 5e-10:
                    pixel_sums[j,i] = 0
                if dead_pixel_perc > 0.0 and (j == dead_pixels).any():
                    pixel_sums[j,i] = 0
                    
            
            
            if IR == True:
                pixel_sums[w_o[i,:] < cutoff,i] = 0
            else:
                pixel_sums[w_o[i,:] > cutoff,i] = 0
            
        
        if plotting == True:
            plt.figure()
            for i in range(len(w_o[:,0])):
                plt.plot(w_o[i],pixel_sums[:,i],label='Order %i'%order_list[i])
            plt.xlabel('Wavelength / nm')
            plt.ylabel('Photon count')
            plt.legend(loc='best')
        
        
    
    return pixel_sums,w_o_bins

def grating_binning_high_enough_R_sky(spec,w_o,all_w_o,order_list,eff,cutoff,IR=False,OPT=False,plotting=False): #note: w_o here is wavelengths_orders
    
    zeroed_wls = []
    
    mult_out = np.zeros_like(w_o)
    mult_wls = np.zeros_like(w_o)
    
    pixel_sums = np.zeros((n_pixels,len(order_list))) #storing photons each pixel sees

    spec_high_R = np.copy(spec)
    
    if IR == True:
        spec_high_R = spec_high_R[:,spec_high_R[0] >= cutoff+1]
    else:
        spec_high_R = spec_high_R[:,spec_high_R[0] <= cutoff+1]
    
    wave_no = len(spec_high_R[0])
    
    #this part takes the range of values around each wavelength in `order_wavelengths` as a bin
    w_o_bins = np.zeros((len(w_o[:]),n_pixels+1)) #was +1
    for i in range(len(w_o[:])):
        for j in range(len(w_o[i])):
            if j == 0:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j+1]-w_o[i,j])/2)
            elif j == len(w_o[i])-1:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j]-w_o[i,j-1])/2)
                w_o_bins[i][j+1] += w_o[i,j] + ((w_o[i,j]-w_o[i,j-1])/2)
            else:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j]-w_o[i,j-1])/2)
    
    #binning photons onto MKID order wavelength bins grid
    if len(spec_high_R[0]) > 0:
        print('\n > Beginning placement of incoming photons.')
        int_steps = np.ndarray.astype((np.linspace(0,len(spec_high_R[0]),10)),dtype='int') 
        prog = 0
        
        for i in range(len(spec_high_R[0])):
            
            coord_bins = []
            
            for j in range(len(w_o_bins[:])):
                coords_low = np.where(spec_high_R[0][i] > w_o_bins[j])[0]
                coords_high = np.where(spec_high_R[0][i] <= w_o_bins[j])[0]
                if len(coords_low) > 0 and len(coords_high) > 0:
                    coord_bins.append(np.array([j,coords_low[-1]]))
                
            if len(coord_bins) > 0:
                factor = len(coord_bins)
                tot_eff = 0
                for j in range(factor):
                    pixel_sums[coord_bins[j][1],coord_bins[j][0]] += (spec_high_R[1][i])*eff[coord_bins[j][0],coord_bins[j][1]]
                    tot_eff += eff[coord_bins[j][0],coord_bins[j][1]]
                for j in range(factor):
                    mult_wls[coord_bins[j][0],coord_bins[j][1]] += 1
                    mult_out[coord_bins[j][0],coord_bins[j][1]] += tot_eff
                

            if (i == int_steps).any():
                prog += 10
                print('\r%i%% of wavelengths complete.'%prog,end='',flush=True)
            
        #applying photon noise
        print('\n > Accounting for photon noise')
        for i in range(len(pixel_sums[0,:])):
            for j in range(len(pixel_sums[:,0])):
                if pixel_sums[j,i] < 1:
                    pixel_sums[j,i] = 0
                    zeroed_wls.append(w_o[i,j])
                try:
                    pixel_sums[j,i] = np.random.poisson(lam=pixel_sums[j,i])
                except:
                    pixel_sums[j,i] = np.random.normal(loc=pixel_sums[j,i],scale=np.sqrt(pixel_sums[j,i]))
                if eff[i,j] < 5e-10:
                    pixel_sums[j,i] = 0
                    
            pixel_sums[pixel_sums[:,i]<1.0,i] = 0
            zeroed_wls.append(w_o[i,:][pixel_sums[:,i]==0])
            
            
            if IR == True:
                pixel_sums[w_o[i,:] < cutoff,i] = 0
            else:
                pixel_sums[w_o[i,:] > cutoff,i] = 0

                      
        if plotting == True:
            plt.figure()
            for i in range(len(w_o[:,0])):
                plt.plot(w_o[i],pixel_sums[:,i],label='Sky Order %i'%order_list[i])
            plt.xlabel('Wavelength / nm')
            plt.ylabel('Photon count')
            plt.legend(loc='best')
    
    return pixel_sums,w_o_bins

def grating_binning_high_enough_R_lim_mag(spec,w_o,all_w_o,order_list,eff,cutoff,IR=False,OPT=False,plotting=False): #note: w_o here is wavelengths_orders
    
    zeroed_wls = []
    
    mult_out = np.zeros_like(w_o)
    mult_wls = np.zeros_like(w_o)
    
    pixel_sums = np.zeros((n_pixels,len(order_list))) #storing photons each pixel sees
    pixel_sums_count = np.zeros_like(pixel_sums)
    spec_high_R = np.copy(spec)
    
    if dead_pixel_perc > 0.0:
        no_pixels_dead = int(n_pixels * (dead_pixel_perc/100))
        dead_pixels = np.random.randint(0,n_pixels,no_pixels_dead)
        if IR == True:
            np.save('DEAD_PIXEL_LISTS/DEAD_PIXELS_IR_ARM.npy',dead_pixels)
        else:
            np.save('DEAD_PIXEL_LISTS/DEAD_PIXELS_OPT_ARM.npy',dead_pixels)
    
    
    if IR == True:
        spec_high_R = spec_high_R[:,spec_high_R[0] >= cutoff+1]
    else:
        spec_high_R = spec_high_R[:,spec_high_R[0] <= cutoff+1]
    
    wave_no = len(spec_high_R[0])
    
    #this part takes the range of values around each wavelength in `order_wavelengths` as a bin
    w_o_bins = np.zeros((len(w_o[:]),n_pixels+1)) #was +1
    for i in range(len(w_o[:])):
        for j in range(len(w_o[i])):
            if j == 0:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j+1]-w_o[i,j])/2)
            elif j == len(w_o[i])-1:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j]-w_o[i,j-1])/2)
                w_o_bins[i][j+1] += w_o[i,j] + ((w_o[i,j]-w_o[i,j-1])/2)
            else:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j]-w_o[i,j-1])/2)
    
    #binning photons onto MKID order wavelength bins grid
    if len(spec_high_R[0]) > 0:
        print('\n > Beginning placement of incoming photons.')
        int_steps = np.ndarray.astype((np.linspace(0,len(spec_high_R[0]),10)),dtype='int') 
        prog = 0
        
        for i in range(len(spec_high_R[0])):
            
            coord_bins = []
            
            for j in range(len(w_o_bins[:])):
                coords_low = np.where(spec_high_R[0][i] > w_o_bins[j])[0]
                coords_high = np.where(spec_high_R[0][i] <= w_o_bins[j])[0]
                if len(coords_low) > 0 and len(coords_high) > 0:
                    coord_bins.append(np.array([j,coords_low[-1]]))
                
            if len(coord_bins) > 0:
                factor = len(coord_bins)
                tot_eff = 0
                for j in range(factor):
                    pixel_sums[coord_bins[j][1],coord_bins[j][0]] += (spec_high_R[1][i])*eff[coord_bins[j][0],coord_bins[j][1]]
                    pixel_sums_count[coord_bins[j][1],coord_bins[j][0]] += 1
                    tot_eff += eff[coord_bins[j][0],coord_bins[j][1]]
                for j in range(factor):
                    mult_wls[coord_bins[j][0],coord_bins[j][1]] += 1
                    mult_out[coord_bins[j][0],coord_bins[j][1]] += tot_eff
                

            if (i == int_steps).any():
                prog += 10
                print('\r%i%% of wavelengths complete.'%prog,end='',flush=True)

            
        #applying photon noise
        #print('\n > Accounting for photon noise')
        for i in range(len(pixel_sums[0,:])):
            for j in range(len(pixel_sums[:,0])):
                if pixel_sums[j,i] < 1:
                    pixel_sums[j,i] = 0
                    zeroed_wls.append(w_o[i,j])
                if eff[i,j] < 5e-10:
                    pixel_sums[j,i] = 0
                if dead_pixel_perc > 0.0 and (j == dead_pixels).any():
                    pixel_sums[j,i] = 0
                    
            pixel_sums[pixel_sums[:,i]<1.0,i] = 0
            zeroed_wls.append(w_o[i,:][pixel_sums[:,i]==0])
            
            
            if IR == True:
                pixel_sums[w_o[i,:] < cutoff,i] = 0
            else:
                pixel_sums[w_o[i,:] > cutoff,i] = 0
            
        
        if plotting == True:
            plt.figure()
            for i in range(len(w_o[:,0])):
                plt.plot(w_o[i],pixel_sums[:,i],label='Order %i'%order_list[i])
            plt.xlabel('Wavelength / nm')
            plt.ylabel('Photon count')
            plt.legend(loc='best')
        
        
    
    return pixel_sums,w_o_bins

def grating_binning_high_enough_R_sky_lim_mag(spec,w_o,all_w_o,order_list,eff,cutoff,IR=False,OPT=False,plotting=False): #note: w_o here is wavelengths_orders
    
    zeroed_wls = []
    
    mult_out = np.zeros_like(w_o)
    mult_wls = np.zeros_like(w_o)
    
    pixel_sums = np.zeros((n_pixels,len(order_list))) #storing photons each pixel sees

    spec_high_R = np.copy(spec)
    
    if IR == True:
        spec_high_R = spec_high_R[:,spec_high_R[0] >= cutoff+1]
    else:
        spec_high_R = spec_high_R[:,spec_high_R[0] <= cutoff+1]
    
    wave_no = len(spec_high_R[0])
    
    #this part takes the range of values around each wavelength in `order_wavelengths` as a bin
    w_o_bins = np.zeros((len(w_o[:]),n_pixels+1)) #was +1
    for i in range(len(w_o[:])):
        for j in range(len(w_o[i])):
            if j == 0:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j+1]-w_o[i,j])/2)
            elif j == len(w_o[i])-1:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j]-w_o[i,j-1])/2)
                w_o_bins[i][j+1] += w_o[i,j] + ((w_o[i,j]-w_o[i,j-1])/2)
            else:
                w_o_bins[i][j] += w_o[i,j] - ((w_o[i,j]-w_o[i,j-1])/2)
    
    #binning photons onto MKID order wavelength bins grid
    if len(spec_high_R[0]) > 0:
        print('\n > Beginning placement of incoming photons.')
        int_steps = np.ndarray.astype((np.linspace(0,len(spec_high_R[0]),10)),dtype='int') 
        prog = 0
        
        for i in range(len(spec_high_R[0])):
            
            coord_bins = []
            
            for j in range(len(w_o_bins[:])):
                coords_low = np.where(spec_high_R[0][i] > w_o_bins[j])[0]
                coords_high = np.where(spec_high_R[0][i] <= w_o_bins[j])[0]
                if len(coords_low) > 0 and len(coords_high) > 0:
                    coord_bins.append(np.array([j,coords_low[-1]]))
                
            if len(coord_bins) > 0:
                factor = len(coord_bins)
                tot_eff = 0
                for j in range(factor):
                    pixel_sums[coord_bins[j][1],coord_bins[j][0]] += (spec_high_R[1][i])*eff[coord_bins[j][0],coord_bins[j][1]]
                    tot_eff += eff[coord_bins[j][0],coord_bins[j][1]]
                for j in range(factor):
                    mult_wls[coord_bins[j][0],coord_bins[j][1]] += 1
                    mult_out[coord_bins[j][0],coord_bins[j][1]] += tot_eff
                

            if (i == int_steps).any():
                prog += 10
                print('\r%i%% of wavelengths complete.'%prog,end='',flush=True)
            
        #applying photon noise
       # print('\n > Accounting for photon noise')
        for i in range(len(pixel_sums[0,:])):
            for j in range(len(pixel_sums[:,0])):
                if eff[i,j] < 5e-10:
                    pixel_sums[j,i] = 0
                    
            pixel_sums[pixel_sums[:,i]<1.0,i] = 0
            zeroed_wls.append(w_o[i,:][pixel_sums[:,i]==0])
            
            
            if IR == True:
                pixel_sums[w_o[i,:] < cutoff,i] = 0
            else:
                pixel_sums[w_o[i,:] > cutoff,i] = 0

                      
        if plotting == True:
            plt.figure()
            for i in range(len(w_o[:,0])):
                plt.plot(w_o[i],pixel_sums[:,i],label='Sky Order %i'%order_list[i])
            plt.xlabel('Wavelength / nm')
            plt.ylabel('Photon count')
            plt.legend(loc='best')
    
    return pixel_sums,w_o_bins


