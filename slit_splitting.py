# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 09:29:57 2020

@author: BVH
"""

'''
This code calculates the FWHM of the object intensity for all the wavelengths in a given data spectrum.
Done using a given value for seeing and airmass. The rest of the code then loops through each of these 'sigma'
values and finds the object intensity transmission for all the slitlets in a defined set up.
'''

import numpy as np
from matplotlib import pyplot as plt
from useful_funcs import gaussian_eq,sigma_calc,nearest
import datetime
plt.rcParams.update({'font.size': 13})


time_start = datetime.datetime.now()

slit_width = 0.63                       #
slit_length = 2.7#slit_width*(2./0.6)   #
pix_fov = 0.3                           #setting up parameters for slits and atmosphere
                                        #
airmass = 1.5                           #
pixels_array = 1050                     #
slits = 1                               #

#0.4, 0.8, 1.2, and 2
seeing = 0.8
mult_sig = False                                                    #setting whether all sigma values calculated
saving_list = False                                                 #will be used or just one (for testing), and
file_name = 'slitlet_transmissions_seeing_1_2_slit_length_2_7_slits_1'  #whether to save the results of the run



pix_rows = int(((slit_length)*slits / pix_fov)) #calculating how many rows of the pixels_array there will be

pixel_number = pix_rows * pixels_array
print('Pixel number is', pixel_number)

slits_range_x = np.linspace(-slit_width*slits/2,slit_width*slits/2,int(slits)+1) #coordinates of each slit and slitlet
slits_range_y = np.linspace(-slit_length/2,slit_length/2,int(pix_rows/(slits))+1)

spec = np.load('slit_spectrum.npy') #data spectrum in photons loaded in, could be raw data one too

x_bounds = slit_length * 2/3
y_bounds = slit_length * 2/3#setting the range of the x and y gaussians

x_data = np.linspace(-x_bounds,x_bounds,3000) #getting the numbers to pass to the gaussian equation
y_data = np.linspace(-y_bounds,y_bounds,3000)

sig = sigma_calc(seeing,spec[0],airmass) #calculating values of FWHM depending on wavelength
mean_x = 0.0
mean_y = 0.0

transmissions = [] #list store for the overall transmission for each slit
slitlet_transmissions = np.zeros((len(sig),int(pix_rows/slits),int(slits))) #numpy array store for each slitlet across all the wavelengths

runs = 0 #keeping track of how many loops have been done, and used to plot only once rather than 24000 times

progress = 0 #used to give updates on the progress of this code, since it takes ~1 hour to ~1.5 hours

if mult_sig == True:
    
    for j in range(len(sig)): #looping through all sigma values for each wavelength
        
        gaussian_x = gaussian_eq(x_data,mu=mean_x,sig=sig[j]) #setting up gaussian in x direction
        gaussian_y = gaussian_eq(y_data,mu=mean_y,sig=sig[j]) #setting up gaussian in y direction
        
        gaussian_x = gaussian_x / np.sum(gaussian_x) #normalising gaussians 
        gaussian_y = gaussian_y / np.sum(gaussian_y)
        
        field = np.zeros((3000,3000)) #2d array to store the values across the 2d plane of the object intensity
        
        for i in range(len(gaussian_x)): #looping through x gaussian values
            
            field[:,i] = gaussian_x[i] * gaussian_y #multiplying all y vaues by one x value to get a column of the 2d plane
            
        slit_int_trans = [] #storing overall transmission for slits
        
        if runs == 0: #plotting gaussian field
            plt.figure()
            plt.contourf(x_data,y_data,field,levels=30)
            
        for i in range(slits): #looping through the number of slits
            
            xs = np.array([slits_range_x[i],slits_range_x[i+1]]) #slit coordinates in x field
            
            coords_low_x = nearest(x_data,xs[0],'coord')            #matching up the slit edges with the gaussian points
            coords_high_x = nearest(x_data,xs[1],'coord')           #    
                                                                    #
            coords_low_y = nearest(y_data,-slit_length/2,'coord')   #
            coords_high_y = nearest(y_data,slit_length/2,'coord')   #
            
            slit_int = field[coords_low_y:coords_high_y+1,coords_low_x:coords_high_x+1] #sectioning area of 2D gaussian
            
            slit_int_trans.append(np.sum(slit_int))                 #summing the area of the gaussian the slit contains
                                                                    #for transmission
            if runs == 0: #plotting
                slit_x = np.array([slits_range_x[i],slits_range_x[i],slits_range_x[i+1],slits_range_x[i+1],slits_range_x[i] ])    
                slit_y = np.array([-slit_length/2,slit_length/2,slit_length/2,-slit_length/2,-slit_length/2])
                plt.plot(slit_x,slit_y,'-',label='Slit %i'%(i+1))
            
            for z in range(len(slits_range_y)-1): #looping through the slitlets contained within the current slit
                
                ys = np.array([slits_range_y[z],slits_range_y[z+1]]) #slitlet coordinates in y field
                
                coord_low_x = nearest(x_data,xs[0],'coord') #matching up the slit edges with the gaussian points
                coord_high_x = nearest(x_data,xs[1],'coord')#
                                                            #
                coord_low_y = nearest(y_data,ys[0],'coord') #
                coord_high_y = nearest(y_data,ys[1],'coord')#
                
                slitlet_int = field[coord_low_y:coord_high_y+1,coord_low_x:coord_high_x+1]#sectioning area of 2D gaussian
                
                slitlet_transmissions[j,z,i] = np.sum(slitlet_int)#summing the area of the gaussian the slitlet contains
                                                                    #for transmission
                
                if runs == 0: #plotting
                    row_x = np.array([slits_range_x[i],slits_range_x[i+1],slits_range_x[i+1],slits_range_x[i],slits_range_x[i]])
                    row_y = np.array([slits_range_y[z],slits_range_y[z],slits_range_y[z+1],slits_range_y[z+1],slits_range_y[z]])
                    plt.plot(row_x,row_y,'w--',linewidth=0.5)                
                
    
        if runs == 0: #plotting
            plt.colorbar(label='Object Intensity/ arbritary units')
            plt.xlabel('Width / arcseconds')
            plt.ylabel('Length / arcseconds')
            plt.legend(loc='upper left')
            
        transmissions.append(slit_int_trans) #recording overall transmissions for slits
        runs += 1
        
        progress += 1
        
        if progress == 500: #updating on progress of code
            print('\n',j+1,'runs complete.',np.round((j/len(sig))*100,decimals=2),'% complete.')
            timing = (datetime.datetime.now() - time_start).seconds
            time_estimation = (timing * (1/((j/len(sig))))) - timing
            print('Estimated time remaining for calcualtion completion: ~',int(time_estimation/60),'minutes')
            progress = 0
            
    if saving_list == True: #saving results
        with open('%s.txt'%file_name,'w') as f:
            f.writelines(['%s\n'% items for items in transmissions])
            f.close()
        np.save('%s'%file_name,slitlet_transmissions)
                
                
else:

    gaussian_x = gaussian_eq(x_data,mu=mean_x,sig=sig[0])
    gaussian_y = gaussian_eq(y_data,mu=mean_y,sig=sig[0])
    
    gaussian_x = gaussian_x / np.sum(gaussian_x)
    gaussian_y = gaussian_y / np.sum(gaussian_y)
    
    field = np.zeros((3000,3000))
    
    for i in range(len(gaussian_x)):

        field[:,i] = gaussian_x[i] * gaussian_y

    slit_int_trans = []
    slitlet_int_trans = []

    plt.figure()
    plt.contourf(x_data,y_data,field,levels=30)
    
    for i in range(slits):
        slits_range = slits_range_x
        slit_x = np.array([slits_range[i],slits_range[i],slits_range[i+1],slits_range[i+1],slits_range[i] ])    
        slit_y = np.array([-slit_length/2,slit_length/2,slit_length/2,-slit_length/2,-slit_length/2])
        plt.plot(slit_x,slit_y,'-',label='Slit %i'%(i+1))
    
    for i in range(slits):
        
        xs = np.array([slits_range_x[i],slits_range_x[i+1]])
        
        coords_low_x = nearest(x_data,xs[0],'coord') #matching up the slit edges with the gaussian points
        coords_high_x = nearest(x_data,xs[1],'coord')
        
        coords_low_y = nearest(y_data,-slit_length/2,'coord')
        coords_high_y = nearest(y_data,slit_length/2,'coord')
    
        slit_int = field[coords_low_y:coords_high_y+1,coords_low_x:coords_high_x+1]
    
        slit_int_trans.append(np.sum(slit_int))                
        
        print(np.sum(slit_int))   
    
        slitlets_int_trans = []
        
        for j in range(len(slits_range_y)-1):
            
            ys = np.array([slits_range_y[j],slits_range_y[j+1]])
        
            coord_low_x = nearest(x_data,xs[0],'coord') #matching up the slit edges with the gaussian points
            coord_high_x = nearest(x_data,xs[1],'coord')
                
            coord_low_y = nearest(y_data,ys[0],'coord')
            coord_high_y = nearest(y_data,ys[1],'coord')
            
            slitlet_int = field[coord_low_y:coord_high_y+1,coord_low_x:coord_high_x+1]
            
            slitlets_int_trans.append(np.sum(slitlet_int))
            
            row_x = np.array([slits_range_x[i],slits_range_x[i+1],slits_range_x[i+1],slits_range_x[i],slits_range_x[i]])
            row_y = np.array([slits_range_y[j],slits_range_y[j],slits_range_y[j+1],slits_range_y[j+1],slits_range_y[j]])
            plt.plot(row_x,row_y,'w--',linewidth=0.5)
        
        slitlet_int_trans.append(slitlets_int_trans)
        
    plt.colorbar(label='Object Intensity/ arbritary units')
    plt.xlabel('Width / arcseconds')
    plt.ylabel('Length / arcseconds')
    plt.legend(loc='upper left')
            



print('\n Calculations took', datetime.datetime.now() - time_start,'(hours:minutes:seconds)')









