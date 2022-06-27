# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:19:32 2020

@author: BVH
"""

from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
from parameters import h,c,mirr_diam,exposure_t,fov,area,binstep,slit_width,lambda_high_val,lambda_low_val #c is in cm/s and mirr_diam is in cm


#UVES------------------------------------------------------------------------------------------------------------------------------------------------------------------

def UVES_sky_data():

    file_path_uves_1 = 'UVES_SKY_DATA/fluxed_sky_346.fits'     #units:  1e-16 erg / (s*A*cm^2*arcs^2)
    f = fits.open(file_path_uves_1)
    uves_3140 = f[0].data
    uves_3140_3760 = np.zeros((len(uves_3140),2))
    uves_3140_3760[:,1] = abs(uves_3140)
    uves_3140_3760[:,0] = np.linspace(3140,3760,len(uves_3140)) 
    f.close()
    
    binstep = (3760 - 3140) / len(uves_3140)
    flux_conversion = (exposure_t*binstep*area*fov) / ( (1e-16) * ((h*c)/(uves_3140_3760[:,0]*1e-8)))
    uves_3140_3760[:,1] *= flux_conversion
    
    
    
    
    file_path_uves_2 = 'UVES_SKY_DATA/fluxed_sky_437.fits'
    f = fits.open(file_path_uves_2)
    uves_3740 = f[0].data
    uves_3740_4940 = np.zeros((len(uves_3740),2))
    uves_3740_4940[:,1] = abs(uves_3740)
    uves_3740_4940[:,0] = np.linspace(3740,4940,len(uves_3740)) 
    f.close()
    
    binstep = (4940 - 3740) / len(uves_3740)
    flux_conversion = (exposure_t*binstep*area*fov) / ( (1e-16)* ((h*c)/(uves_3740_4940[:,0]*1e-8)))
    uves_3740_4940[:,1] *= flux_conversion
    
    
    
    
    file_path_uves_3 = 'UVES_SKY_DATA/fluxed_sky_564U.fits'
    f = fits.open(file_path_uves_3)
    uves_4800 = f[0].data
    uves_4800_5800 = np.zeros((len(uves_4800),2))
    uves_4800_5800[:,1] = abs(uves_4800)
    uves_4800_5800[:,0] = np.linspace(4800,5800,len(uves_4800)) 
    f.close()
    
    binstep = (5800 - 4800) / len(uves_4800)
    flux_conversion = (exposure_t*binstep*area*fov) / ( (1e-16)* ((h*c)/(uves_4800_5800[:,0]*1e-8)))
    uves_4800_5800[:,1] *= flux_conversion
    
    
    
    
    file_path_uves_4 = 'UVES_SKY_DATA/fluxed_sky_580L.fits'
    f = fits.open(file_path_uves_4)
    uves_5800 = f[0].data
    uves_5800_6760 = np.zeros((len(uves_5800),2))
    uves_5800_6760[:,1] = abs(uves_5800)
    uves_5800_6760[:,0] = np.linspace(5800,6760,len(uves_5800)) 
    f.close()
    
    binstep = (6760 - 5800) / len(uves_5800)
    flux_conversion = (exposure_t*binstep*area*fov) / ( (1e-16)* ((h*c)/(uves_5800_6760[:,0]*1e-8)))
    uves_5800_6760[:,1] *= flux_conversion
    



    file_path_uves_5 = 'UVES_SKY_DATA/fluxed_sky_580U.fits'
    f = fits.open(file_path_uves_5)
    uves_5700 = f[0].data
    uves_5700_5900 = np.zeros((len(uves_5700),2))
    uves_5700_5900[:,1] = abs(uves_5700)
    uves_5700_5900[:,0] = np.linspace(5700,5900,len(uves_5700)) 
    f.close()
    
    binstep = (5700 - 5900) / len(uves_5700)
    flux_conversion = (exposure_t*binstep*area*fov) / ( (1e-16)* ((h*c)/(uves_5700_5900[:,0]*1e-8)))
    uves_5700_5900[:,1] *= flux_conversion
    
    
    
    
    file_path_uves_6 = 'UVES_SKY_DATA/fluxed_sky_800U.fits'
    f = fits.open(file_path_uves_6)
    uves_6700 = f[0].data
    uves_6700_8560 = np.zeros((len(uves_6700),2))
    uves_6700_8560[:,1] = abs(uves_6700)
    uves_6700_8560[:,0] = np.linspace(6700,8560,len(uves_6700)) 
    f.close()

    binstep = (8560 - 6700) / len(uves_6700)
    flux_conversion = (exposure_t*binstep*area*fov) / ( (1e-16)* ((h*c)/(uves_6700_8560[:,0]*1e-8)))
    uves_6700_8560[:,1] *= flux_conversion
    
    
    
    
    file_path_uves_7 = 'UVES_SKY_DATA/fluxed_sky_860L.fits'
    f = fits.open(file_path_uves_7)
    uves_8600 = f[0].data
    uves_8600_10430 = np.zeros((len(uves_8600),2))
    uves_8600_10430[:,1] = abs(uves_8600)
    uves_8600_10430[:,0] = np.linspace(8600,10430,len(uves_8600)) 
    f.close()
    
    binstep = (10430 - 8600) / len(uves_8600)
    flux_conversion = (exposure_t*binstep*area*fov) / ( (1e-16)* ((h*c)/(uves_8600_10430[:,0]*1e-8)))
    uves_8600_10430[:,1] *= flux_conversion
    
    
    
    
    file_path_uves_8 = 'UVES_SKY_DATA/fluxed_sky_860U.fits'
    f = fits.open(file_path_uves_8)
    uves_8530 = f[0].data
    uves_8530_8630 = np.zeros((len(uves_8530),2))
    uves_8530_8630[:,1] = abs(uves_8530)
    uves_8530_8630[:,0] = np.linspace(8530,8630,len(uves_8530))
    f.close()
        
    binstep = (8630 - 8530) / len(uves_8530)
    flux_conversion = (exposure_t*binstep*area*fov) / ( (1e-16)* ((h*c)/(uves_8530_8630[:,0]*1e-8)))
    uves_8530_8630[:,1] *= flux_conversion

    
    final_arr = np.concatenate((uves_3140_3760,uves_3740_4940,uves_4800_5800,uves_5700_5900,uves_5800_6760,uves_6700_8560,uves_8530_8630,uves_8600_10430),axis=0)
    
    return final_arr
    
    

#GEMINI--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        
def flux_conv_GEM(wls):
    binstep = (wls[-1] - wls[0]) / len(wls)
    return  exposure_t*binstep*fov*((0.5*(mirr_diam/100))**2 *np.pi)



def GEMINI_sky_data():
    
    file_path_gemini_opt = 'GEMINI_SKY_DATA/skybg_50_10_OPTICAL' #units: 
    f = open(file_path_gemini_opt)
    gemini_opt_str = f.readlines()[14:]
    gemini_opt = np.zeros((len(gemini_opt_str),2))                      # units are ph/sec/arcsec^2/nm/m^2
    for i in range(len(gemini_opt_str)):
        gemini_opt[i,0] = gemini_opt_str[i].split(' ')[0]
        gemini_opt[i,1] = gemini_opt_str[i].split(' ')[-1]
    f.close()
    
    file_path_gemini_ir = 'GEMINI_SKY_DATA/mk_skybg_zm_50_15_ph_IR.dat'
    with open(file_path_gemini_ir,'r') as f:                             
        gemini_ir_str = f.readlines()                                           
        f.close()    
    gemini_ir = np.zeros((len(gemini_ir_str),2))
    for i in range(len(gemini_ir_str)):
        gemini_ir[i,0] = gemini_ir_str[i].split('      ')[0]                #units are ph/sec/arcsec^2/nm/m^2
        gemini_ir[i,1] = gemini_ir_str[i].split('      ')[1]


    gemini_opt[:,1] *= flux_conv_GEM(gemini_opt[:,0])
    gemini_ir[:,1] *= flux_conv_GEM(gemini_ir[:,0])
    
    return np.concatenate((gemini_opt,gemini_ir),axis=0)



#ESO SKY MODEL---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def ESO_sky_model(file):
    
    file_path_sky_mod = 'ESO_SKY_MODEL/%s'%file
    f = fits.open(file_path_sky_mod)
    sky_mod = f[1].data
    f.close()
    
    with open('ESO_SKY_MODEL/mag_per_arcsec_sq.txt') as f:
        data_mags = f.readlines()
        f.close()
    mags = []
    for i in range(len(data_mags)):
        mags.append(float(data_mags[i].split()[1]))
    mags = np.asarray(mags)
    
    sky_mod_lam = sky_mod.field(0)                                          #units are ph/s/m2/micron/arcsec2
    sky_mod_flux = sky_mod.field(1)
    sky_mod = np.zeros((2,len(sky_mod_lam)))
    sky_mod[0] += sky_mod_lam
    sky_mod[1] += sky_mod_flux
    bin_width = (sky_mod_lam[-1] - sky_mod_lam[0]) / len(sky_mod_lam)
    
    units_conv = exposure_t*(np.pi*(0.5*(mirr_diam/100))**2)*(bin_width/1000)*(slit_width*2.7)   #*fov
    
    sky_mod[1] *= units_conv
    low_coord = np.where(sky_mod[0]>lambda_low_val)[0][0]
    high_coord = np.where(sky_mod[0]<lambda_high_val)[0][-1]
    sky_mod = sky_mod[:,low_coord:high_coord]
    
    for i in range(len(sky_mod[1])):
        if sky_mod[1][i] < 1:        
            sky_mod[1][i] += np.random.poisson(lam=1.0)
    
    return sky_mod


#PLOTTING-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def sky_plotter():

    uves = UVES_sky_data()
    gemini = GEMINI_sky_data()
    ESO = ESO_sky_model()
    
    plt.figure()

    plt.subplot(311)
    plt.plot(uves[:,0],uves[:,1],'b,',label='UVES Sky Atlas')
    plt.xlabel('Wavelength / A')
    plt.ylabel('Photons')
    plt.legend(loc='best')
    
    plt.subplot(312)
    plt.plot(gemini[:,0],gemini[:,1],'r,',label='GEMINI sky data')
    plt.xlabel('Wavelength / nm')
    plt.ylabel('Photons')
    plt.legend(loc='best')
    
    
    plt.subplot(313)
    plt.plot(ESO[:,0],ESO[:,1],'g-',label='ESO Sky Model')
    plt.xlabel('Wavelength / nm')
    plt.ylabel('Photons')
    plt.legend(loc='best')
    





