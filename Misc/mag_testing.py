# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:59:45 2020

@author: BVH
"""


import numpy as np
from astropy.io import fits
from scipy import interpolate
from matplotlib import pyplot as plt

coord = 3

objects = ['HD6229','HD212442','HD18769','HD17072','HD31373','EPIC 201553007']
B_mags = np.array([9.35,6.703,6.054,7.22,5.701,9.809])
V_mags = np.array([8.58,6.785,5.909,6.59,5.776,9.239])
J_mags = np.array([7.088,6.833,5.601,5.244,5.861,9.137])

obj = objects[coord]
Loop = True

def mag_calc(spec,plotting=False):
    
    with open('Passbands/zero_fluxes_Vega_system.txt','r') as f:
        data = f.readlines()
        f.close()
    zero_fluxes = np.zeros(len(data))
    mags = []
    
    for i in range(len(data)):
        row = data[i].split()
        zero_fluxes[i] = float(row[2])
        mags.append(row[0])
    
    bandpass_wls = []
    bandpass_trans = []
    for i in range(len(zero_fluxes)):
        with open('Passbands/%s_bandpass.txt'%mags[i],'r') as f:
            data = f.readlines()
            f.close()
        current_wls = []
        current_trans = []
        
        for j in range(len(data)):
            current_wls.append(data[j].split()[0])
            current_trans.append(data[j].split()[1])
        if i >= 5:
            bandpass_wls.append(np.array(current_wls,dtype='float32')*1000)
            bandpass_trans.append(np.array(current_trans,dtype='float32'))
        else:
            bandpass_wls.append(np.array(current_wls,dtype='float32'))
            bandpass_trans.append(np.array(current_trans,dtype='float32'))
        
    bandpass_wls[6] /= 1000  
    
    obj_mags = []
    
    for i in range(len(mags)):
        
        trans = bandpass_trans[i]
        wls = bandpass_wls[i]
    
        if np.sum( (wls > np.min(spec[0])) ) >= 1 and np.sum( (np.max(spec[0]) > wls) ) >= 1:
            
            diff_high_lam = spec[0][-1] - wls[-1]
            diff_low_lam = wls[0] - spec[0][0]
            
            if diff_high_lam > 0 and diff_low_lam > 0:
                
                seg_1 = np.linspace(spec[0][0],wls[0],1000)
                seg_2 = wls
                seg_3 = np.linspace(wls[-1],spec[0][-1],1000)
                
                seg_12 = np.concatenate((seg_1,seg_2))
                final_wls = np.concatenate((seg_12,seg_3))
                
                final_bandpass = np.zeros(len(final_wls))
                
                final_bandpass[len(seg_1):len(seg_12)] += trans
                
            if diff_high_lam > 0 and diff_low_lam < 0:
        
                seg_1 = wls
                seg_2 = np.linspace(wls[-1],spec[0][-1],3000)
                
                final_wls = np.concatenate((seg_1,seg_2))
                
                final_bandpass = np.zeros(len(final_wls))
                
                final_bandpass[0:len(seg_1)] += trans
                
            if diff_high_lam < 0 and diff_low_lam > 0:
        
                seg_1 = np.linspace(spec[0][0],wls[0],3000)
                seg_2 = wls
                
                final_wls = np.concatenate((seg_1,seg_2))
                
                final_bandpass = np.zeros(len(final_wls))
                
                final_bandpass[len(seg_1):] += trans
            
            if diff_high_lam < 0 and diff_low_lam < 0:
                
                final_wls = wls
                final_bandpass = trans
            
            mag_f = interpolate.interp1d(final_wls,final_bandpass)
            filter_for_spec_1 = mag_f(spec[0])
            filter_for_spec_2 = filter_for_spec_1/np.max(filter_for_spec_1)
            filtered_spec = spec[1]*filter_for_spec_2
            
            f_mag_res = np.trapz(filtered_spec,spec[0])#integrate.quad(lambda x: f_mag_spec(x),spec[0][0],spec[0][-1])[0]
            
            filtered_spec_zero = filter_for_spec_2*zero_fluxes[i]
            
            f_mag_zero = np.trapz(filtered_spec_zero,spec[0])
            
            final_mag = (-2.5*np.log10(f_mag_res/f_mag_zero)) #- (-2.5*np.log10(f_mag_res/f_mag_zero))
            
            #final_mag = -2.5*np.log10(np.sum(filtered_spec)/zero_fluxes[i])
            
            obj_mags.append(final_mag)
            
            if plotting == True:
                
                fig1 = plt.figure()
                plt.plot(spec[0],filtered_spec,'b-')
                plt.xlabel('Wavelength /nm')
                plt.ylabel('Flux / $ergcm^{-2}s^{-1}\AA^{-1}$')
                fig1.text(0.15,0.82,'%s mag = %.2f'%(mags[i],final_mag))
    
        else:
            
            obj_mags.append('N/A')
        
    return obj_mags,mags#,zero_fluxes


if obj == 'HD6229':

    file_path = 'HD6229/HD6229_389582_55110_UVB+VIS.fits'
    
    fits_file = fits.open(file_path)
    
    flux = abs(fits_file[0].data)
    wave = np.linspace(300,1020,len(flux))
    
    fits_file.close()
    
    spec = (wave,flux)
    
    mags_HD6229 = mag_calc(spec)
    
    spec_2 = np.zeros((2,len(spec[0])))
    spec_2[0] += spec[0]
    spec_2[1] += spec[1]*1.3
    
    mags_HD6229_slit_corrected = mag_calc(spec_2)
    
    print(obj)
    print('B magnitude is 9.35')
    print('V magnitude is 8.58')
    print('J magnitude is 7.088')
    print('\nB --> %.2f --> error --> %.2f'%(mags_HD6229[0][1],abs(9.35-mags_HD6229[0][1])))
    print('V --> %.2f --> error --> %.2f'%(mags_HD6229[0][2],abs(8.58-mags_HD6229[0][2])))
    print('J --> %.3f --> error --> %.3f'%(mags_HD6229[0][7],abs(7.09-mags_HD6229[0][7])))

elif obj == 'HD212442':
    
    file_path = 'HD212442/ADP.2019-11-05T13_53_49.533.fits'
    fits_file = fits.open(file_path)
    
    data_fits = fits_file[1].data
    
    flux = abs(data_fits.field(1)[0])
    wave = data_fits.field(0)[0]
    
    fits_file.close()
    
    spec = (wave,flux)
    
    mags_HD212442 = mag_calc(spec)
    
    spec_2 = np.zeros_like(spec)
    spec_2[0] += spec[0]
    spec_2[1] += spec[1]*1.3
    
    mags_HD212442_slit_corrected = mag_calc(spec_2)
    
    print(obj)
    print('B magnitude is 6.703')
    print('V magnitude is 6.785')
    print('J magnitude is 6.833')
    print('\nB --> %.3f --> error --> %.3f'%(mags_HD212442_slit_corrected[0][1],abs(6.703-mags_HD212442_slit_corrected[0][1])))
    print('V --> %.3f --> error --> %.3f'%(mags_HD212442_slit_corrected[0][2],abs(6.785-mags_HD212442_slit_corrected[0][2])))
    print('J --> %.3f --> error --> %.3f'%(mags_HD212442_slit_corrected[0][7],abs(6.833-mags_HD212442_slit_corrected[0][7])))

elif obj == 'HD18769':

    file_path = 'HD18769/HD18769_389652_55162_UVB+VIS.fits'
    
    fits_file = fits.open(file_path)
    
    flux = abs(fits_file[0].data)
    wave = np.linspace(300,1020,len(flux))
    
    fits_file.close()
    
    spec = (wave,flux)
    
    mags = mag_calc(spec)
    
    spec_2 = np.zeros((2,len(spec[0])))
    spec_2[0] += spec[0]
    spec_2[1] += spec[1]*1.3
    
    mags_slit_corrected = mag_calc(spec_2)
    
    print(obj)
    print('B magnitude is 6.054')
    print('V magnitude is 5.909')
    print('J magnitude is 5.601')
    print('\nB --> %.3f --> error --> %.3f'%(mags[0][1],abs(6.054-mags[0][1])))
    print('V --> %.3f --> error --> %.3f'%(mags[0][2],abs(5.909-mags[0][2])))
    print('J --> %.3f --> error --> %.3f'%(mags[0][7],abs(5.601-mags[0][7])))


elif obj == 'HD17072':
    
    file_path = 'HD17072/HD17072_389656_55119_UVB+VIS.fits'
    
    fits_file = fits.open(file_path)
    
    flux = abs(fits_file[0].data)
    wave = np.linspace(300,1020,len(flux))
        
    fits_file.close()
    
    spec = (wave,flux)
    
    mags = mag_calc(spec)
    
    spec_2 = np.zeros((2,len(spec[0])))
    spec_2[0] += spec[0]
    spec_2[1] += spec[1]*1.3
    
    mags_slit_corrected = mag_calc(spec_2)
    
    print(obj)
    print('B magnitude is 7.22')
    print('V magnitude is 6.59')
    print('J magnitude is 5.244')
    print('\nB --> %.2f --> error --> %.2f'%(mags[0][1],abs(7.22-mags[0][1])))
    print('V --> %.2f --> error --> %.2f'%(mags[0][2],abs(6.59-mags[0][2])))
    print('J --> %.3f --> error --> %.3f'%(mags[0][7],abs(5.244-mags[0][7])))

elif obj == 'HD31373':
    
    file_path = 'HD31373/ADP.2020-02-04T15_18_49.475.fits'
    fits_file = fits.open(file_path)
    
    data_fits = fits_file[1].data
    
    flux = abs(data_fits.field(1)[0])
    wave = data_fits.field(0)[0]
    
    fits_file.close()
    
    spec = (wave,flux)
    
    mags = mag_calc(spec)
    
    spec_2 = np.zeros_like(spec)
    spec_2[0] += spec[0]
    spec_2[1] += spec[1]*1.3
    
    mags_slit_corrected = mag_calc(spec_2)
    
    print(obj)
    print('B magnitude is 5.701')
    print('V magnitude is 5.776')
    print('J magnitude is 5.861')
    print('\nB --> %.3f --> error --> %.3f'%(mags_slit_corrected[0][1],abs(5.701-mags_slit_corrected[0][1])))
    print('V --> %.3f --> error --> %.3f'%(mags_slit_corrected[0][2],abs(5.776-mags_slit_corrected[0][2])))
    print('J --> %.3f --> error --> %.3f'%(mags_slit_corrected[0][7],abs(5.861-mags_slit_corrected[0][7])))

elif obj == 'EPIC 201553007':
    
    file_path = 'EPIC_201553007/ADP.2019-04-03T12_45_36.491.fits'
    fits_file = fits.open(file_path)
    
    data_fits = fits_file[1].data
    
    flux = abs(data_fits.field(1)[0])
    wave = data_fits.field(0)[0]
    
    fits_file.close()
    
    spec = (wave,flux)
    
    mags = mag_calc(spec)
    
    spec_2 = np.zeros_like(spec)
    spec_2[0] += spec[0]
    spec_2[1] += spec[1]*1.3
    
    mags_slit_corrected = mag_calc(spec_2)
    
    print(obj)
    print('J magnitude is 9.809')
    print('H magnitude is 9.239')
    print('K magnitude is 9.137')
    print('\nJ --> %.3f --> error --> %.3f'%(mags_slit_corrected[0][7],abs(9.809-mags_slit_corrected[0][7])))
    print('H --> %.3f --> error --> %.3f'%(mags_slit_corrected[0][8],abs(9.239-mags_slit_corrected[0][8])))
    print('K --> %.3f --> error --> %.3f'%(mags_slit_corrected[0][9],abs(9.137-mags_slit_corrected[0][9])))


if Loop == True:
    
    fig1 = plt.figure(1)
    plt.xlabel('Factor to account for losses')
    plt.ylabel('Error in calculated magnitude')
    fig1.text(0.15,0.15,'B mag')
    
    fig2 = plt.figure(2)
    plt.xlabel('Factor to account for losses')
    plt.ylabel('Error in calculated magnitude')
    fig2.text(0.15,0.15,'V mag')
    
    fig3 = plt.figure(3)
    plt.xlabel('Factor to account for losses')
    plt.ylabel('Error in calculated magnitude')
    fig3.text(0.15,0.15,'J mag')
    
    factors = np.linspace(0.01,10,1000)
    
    for i in range(len(factors)):
        spec_2 = np.zeros_like(spec)
        spec_2[0] += spec[0]
        spec_2[1] += spec[1]*factors[i]
        mags_slit_corrected = mag_calc(spec_2)  
        B_err = abs(B_mags[coord]-mags_slit_corrected[0][1])
        V_err = abs(V_mags[coord]-mags_slit_corrected[0][2])
        J_err = abs(J_mags[coord]-mags_slit_corrected[0][7])
        plt.figure(1)
        plt.plot(factors[i],B_err,'ro',markersize=2)
        plt.figure(2)
        plt.plot(factors[i],V_err,'bo',markersize=2)
        plt.figure(3)
        plt.plot(factors[i],J_err,'go',markersize=2)
    












