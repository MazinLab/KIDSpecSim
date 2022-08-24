# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:47:14 2020

@author: BVH
"""

import numpy as np
from matplotlib import pyplot as plt

from parameters import object_file,lambda_low_val,lambda_high_val,alpha_val,phi_val,refl_start_val,refl_end_val,norders, \
                        OPT_a,eff_models,IR_a
from grating import grating_orders_2_arms
from useful_funcs import gaussian_eq,where_in_array,specific_wl_overlap,order_doubles,wavelength_array_maker,data_extractor

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------=------------------------
def grating_orders_2_arms_R_comp(model,n_pixels,plotting=False):
    
    #<800nm is OPT
    #>800nm is IR
    
    if model not in eff_models: #making sure the efficiency model exists in the list of supported models
        raise ValueError('Model must be one of ', eff_models)
    
    #getting variables from parameter file
    lambda_low = lambda_low_val #lambda = wavelength
    lambda_high = lambda_high_val #these two variables make up the wavelength range that will be covered
    alpha = alpha_val * (np.pi / 180) #angle of incidence
    phi = phi_val * (np.pi / 180) #grating blaze angle
    refl_start = refl_start_val * (np.pi / 180) #reflected angle (lowest)(end is highest)
    refl_end = refl_end_val * (np.pi / 180)  #converting to radians
    
    reflected_angles = np.linspace(refl_start,refl_end,n_pixels) #angles which will fall on each pixel
        
    order_list_opt = [] #orders which fall in the required range will be added to this list
    wavelengths_orders_opt = [] #their wavelengths will be added here
    
    order_list_ir = [] #orders which fall in the required range will be added to this list
    wavelengths_orders_ir = [] #their wavelengths will be added here
    

    
    for i in range(1,int(norders)):
        
        wavelengths_in_order_OPT = (((np.sin(reflected_angles) + np.sin(np.abs(alpha))) * OPT_a) / (i)) * 1000 #grating equation, 1000 is for groove conversion to nm
        wavelengths_in_order_IR = (((np.sin(reflected_angles) + np.sin(np.abs(alpha))) * IR_a) / (i)) * 1000 #grating equation, 1000 is for groove conversion to nm
        
        waves_in_order_OPT = len(list(x for x in wavelengths_in_order_OPT if lambda_low < x < 800 and x < lambda_high)) #checking if any wavelengths fall in the range setout in parameters
        waves_in_order_IR = len(list(x for x in wavelengths_in_order_IR if 800 < x < lambda_high and x > lambda_low)) #checking if any wavelengths fall in the range setout in parameters
        
        if waves_in_order_OPT != 0:
            order_list_opt.append(i)
            wavelengths_orders_opt.append(wavelengths_in_order_OPT) #adding relevant orders and their wavelengths to lists
        
        if waves_in_order_IR != 0:
            order_list_ir.append(i)
            wavelengths_orders_ir.append(wavelengths_in_order_IR) #adding relevant orders and their wavelengths to lists
    
    wavelengths_orders_opt = np.round(np.asarray(wavelengths_orders_opt),decimals=2) #number of decimals here depends on how much can be resolved
    order_list_opt = np.asarray(order_list_opt)
    
    wavelengths_orders_ir = np.round(np.asarray(wavelengths_orders_ir),decimals=2) #number of decimals here depends on how much can be resolved
    order_list_ir = np.asarray(order_list_ir)
        
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
    
    if len(order_list_ir) == 0:
        order_list_ir = np.array([1])
        wavelengths_orders = wavelengths_orders_opt
        order_list = order_list_opt
    elif len(order_list_opt) == 0:
        order_list_opt = np.array([1])
        wavelengths_orders = wavelengths_orders_ir
        order_list = order_list_ir
    elif order_list_ir[-1] == order_list_opt[0]:
        order_list = np.concatenate((order_list_ir,order_list_opt[1:]))
        wavelengths_orders = np.concatenate((wavelengths_orders_ir,wavelengths_orders_opt[1:]))
        order_list_opt = order_list_opt[1:]
        wavelengths_orders_opt = wavelengths_orders_opt[1:] 
    elif order_list_ir[-2] == order_list_opt[0]:
        order_list = np.concatenate((order_list_ir,order_list_opt[2:]))
        wavelengths_orders = np.concatenate((wavelengths_orders_ir,wavelengths_orders_opt[2:]))
        order_list_opt = order_list_opt[2:]
        wavelengths_orders_opt = wavelengths_orders_opt[2:]
    else:
        order_list = np.concatenate((order_list_ir,order_list_opt))
        wavelengths_orders = np.concatenate((wavelengths_orders_ir,wavelengths_orders_opt))
    
    detec_spec = wavelength_array_maker(wavelengths_orders) #forms a 1D array based on the wavelengths each order sees
    
    wavelength_overlaps,overlap_amounts,raw_overlap =  specific_wl_overlap(wavelengths_orders,order_list,detec_spec) #finds what orders overlap and at what array coordinates

    twin_wl = order_doubles(wavelengths_orders) #finds any wavelengths which appear twice 
    
    #calculating grating efficiency and plotting
    
    efficiencies = np.zeros_like(wavelengths_orders) #data array to store efficiencies 
    
    unaltered_eff = np.zeros_like(wavelengths_orders)
    
    w_o = wavelengths_orders[:,:-1]    
    
    if plotting == True: #plotting code
        plt.figure()
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Efficiency')
        #plt.title('Grating efficiencies across each order using a %s model'%model)    
        
    for i in range(len(order_list)): #looping through orders to calculate each's efficiency
        
        
        
        if model == 'Gaussian':   #REDUNDANT
        
            mu = (w_o[i,0] + w_o[i,-1]) / 2 #mid point of wavelength range
            sig = abs(w_o[i,0] - w_o[i,-2]) / 5 #width
            
            eff_i = gaussian_eq(w_o[i,:-1],mu,sig)
            eff_i = eff_i / np.max(eff_i) #setting to be maximum 1
            
            efficiencies[i,:] = eff_i
        
        
        
        
        if model == 'Sinc':      #NO LONGER USED but works
    
            sec_first = (order_list[i] * np.cos(alpha)) / np.cos(alpha-phi) #sec here corresponds to section of the equation for efficiency
            sec_second = np.sin(alpha-phi) + np.sin(reflected_angles[0:-1] - phi)
            sec_third = np.sin(alpha) + np.sin(reflected_angles[0:-1])
            eff_i = (np.sinc( sec_first*sec_second / sec_third )**2) * 0.82
            
            efficiencies[i,:] = eff_i
            
        
        
        
        
        if model == 'Schroeder':  #NOT APPLICABLE
            
            av_refl_angle = np.mean(reflected_angles)
            
            reflected_angles = reflected_angles
            
            sec_first = ((2*np.pi)/(w_o[i,:])) * (((OPT_a+IR_a)/2)*1000) * np.cos(phi)
            sec_second = np.sin((reflected_angles[:-1] - av_refl_angle)/2 )
            sec_third = np.cos(0 - ((reflected_angles[:-1]-av_refl_angle)/2))
            
            eff_i = ((np.sinc(sec_first*sec_second*sec_third))**2) * 0.82 #0.82 factor is for scaled difference between predicted and measured effciencies
            
            efficiencies[i,:] = eff_i
        
        
        
        if model == 'Casini':     #MAINLY USED
                
            
            sec_first = (order_list[i])*( np.cos(alpha)/np.cos(alpha-phi) ) #The paper states there should be pi multiplied by the order o
            sec_second = (np.sin(alpha-phi) + np.sin(reflected_angles-phi)) #this is not done here because on python sinc(x) = sin(pi*x) / pi*x 
            
            eff_i = (np.sinc(sec_first*sec_second)**2)*0.84
            
            unaltered_eff[i] = eff_i
            
            
            for A in range(len(w_o[i])):
                
                if (w_o[i,A] == wavelength_overlaps).any() and (w_o[i,A] == twin_wl).any(): #if the wavelength appears three times
                    
                    eff_i[A] /=  3
                            
                elif (w_o[i,A] == twin_wl).any(): #if the wavelength appears twice across orders
                    
                    eff_i[A] /=  2
                    
                    
                elif (w_o[i,A] == wavelength_overlaps).any(): #if the wavelength appears twice in a single order
                    
                    eff_i[A] /=  2
                    
                    
                else:
                                                   #if the wavelength appears only once
                    eff_i[A] /=  1
                                
                                
            efficiencies[i,:] = eff_i   #adding current orders results to array storing all order efficiencies
            
            efficiencies_ir = efficiencies[:len(order_list_ir),:]
            efficiencies_opt = efficiencies[len(order_list_ir):,:] 
    
    
        if plotting == True:
            plt.plot(wavelengths_orders[i,:],unaltered_eff[i],'-',label='Order %i'%order_list[i])
    
    if plotting == True:
        plt.legend(loc='best',fontsize=11)
    
    np.save('unaltered_eff',unaltered_eff)
    return order_list, wavelengths_orders,efficiencies,order_list_opt,wavelengths_orders_opt,efficiencies_opt,order_list_ir,wavelengths_orders_ir,efficiencies_ir


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


print('> Initialising...\n')

data_file_spec = data_extractor(object_file,XShooter=False,plotting=False) 


slit_width = 0.63
pixel_fov = 0.3

n_spaxels = {
        '1x1' : 1*1,
        '1x3' : 1*3,
        '1x6' : 1*6,
        '1x9' : 1*9,
        '2x1' : 2*1,
        '2x3' : 2*3,
        '2x6' : 2*6,
        '2x9' : 2*9,
        '3x1' : 3*1,
        '3x3' : 3*3,
        '3x6' : 3*6,
        '3x9' : 3*9,
        '4x1' : 4*1,
        '4x3' : 4*3,
        '4x6' : 4*6,
        '4x9' : 4*9,
        '5x1' : 5*1,
        '5x3' : 5*3,
        '5x6' : 5*6,
        '5x9' : 5*9,
        '6x1' : 6*1,
        '6x3' : 6*3,
        '6x6' : 6*6,
        '6x9' : 6*9,
        }

max_pixels = 20000

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
spec_R_1 = []
spec_R_2 = []
xs = []
prog = 0
perc = 0

markers = ['o','v','s','D','p','P','1']

print('> Beginning calculations...\n')

for i in n_spaxels:
    
    prog += 1
    if prog == 5:
        prog = 1
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1, 1, 1)
        spec_R_1 = []
        spec_R_2 = []
        xs = []
    
    spec_pix = int(max_pixels / n_spaxels[i])
    
    orders,order_wavelengths,grating_efficiency,orders_opt,order_wavelengths_opt,efficiencies_opt,orders_ir,order_wavelengths_ir,efficiencies_ir = grating_orders_2_arms_R_comp('Casini',spec_pix,plotting=False)
    
    wavelengths = wavelength_array_maker(order_wavelengths)[0]
    
    spec_R_low_lam = wavelengths[0] / abs(wavelengths[0]-wavelengths[1])
    spec_R_high_lam = wavelengths[-1] / abs(wavelengths[-2]-wavelengths[-1])
    
    spec_R_1.append(spec_R_low_lam)
    spec_R_2.append(spec_R_high_lam)
    xs.append(n_spaxels[i])
    
    x = np.array([n_spaxels[i],n_spaxels[i]])
    y = np.array([spec_R_low_lam,spec_R_high_lam])
    
    ax1.plot(x,y,'k%s'%(markers[int(prog-1)]),label='%s - %i'%(i,spec_pix))
    ax2.plot(x,y,'k-',lw=1,alpha=0.4)
    ax2.plot(x,y,'v')
    
    if prog == 4:
        spec_R_1 = np.array(spec_R_1)
        spec_R_2 = np.array(spec_R_2)
        xs = np.array(xs)
        spec_Rs = (spec_R_1 + spec_R_2) / 2
        
        ax1.fill_between(xs,spec_R_1,spec_R_2,color='aqua',alpha=0.5)
        ax1.plot(xs,spec_Rs,'k-',lw=3)
        ax1.legend(loc='best')
        ax1.set_xlabel('Spaxel number')
        ax1.set_ylabel('Spectral resolution')
    
    perc += 1
    print(int(perc/len(n_spaxels) *100),'% complete \n')
    

ax2.set_xlabel('Spaxel number')
ax2.set_ylabel('Spectral resolution')

































