# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:38:40 2019

@author: BVH
"""

'''
This script extracts the object spectra data from the saved fits file
Also contains other useful funcs
'''


#filename is ADP.2019-11-05T13_53_49.533.fits
#filename of 2DSPECTRUM data is ADP.2019-11-05T13_53_49.534.fits
#form of fits file from ESO is (in order)
#Wavelength
#Flux
#Error
#Quality (?)
#SNR
#Flux Reduced (?)
#Error reduced (?)
import os
from astropy.io import fits
from parameters import *
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit
import scipy.stats
import multiprocessing as mp
from sky_background_data import ESO_sky_model
import h5py
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import L_sun, c








#############################################################################################################################################################
#extracts flux and wavelengths from FITS files
#################################################################################################################################################################################

def data_extractor(file_name,XShooter=False,II_D=False,fits_f=False,SDSS=False,Seyfert=False,
                   stand_star=False,JWST=False,quasar=False,plotting=False): 
    
    if XShooter == True:
        file_path = folder + '/' + file_name
        fits_file = fits.open(file_path)
        
        flux = abs(fits_file[0].data)
        wave = np.linspace(300,1020,len(flux))
        
        low_coord = nearest(wave,lambda_low_val,'coord')
        
        flux = flux[low_coord:]
        wave = wave[low_coord:]
        
        
        fits_file.close()
        
    elif II_D == True:
        
        if fits_f == True:

            fits_file = fits.open('%s/%s'%(folder,file_name))
            flux = fits_file[0].data
            fits_file.close
            
            z = float(object_file[12]+'.'+object_file[14])
            Hawave = (1+z)*656.28
            dl = 0.1e4
            lam_s = Hawave - dl
            lam_e = Hawave + dl
            
            wave = np.linspace(lam_s,lam_e,len(flux[:,0,0]))
        
        else:
            flux = np.load('Generated_Galaxy_5638463/generated_galaxy_01_05_2020_5638463.npy')
            wave = np.load('Generated_Galaxy_5638463/gal_wavelengths_01_05_2020_5638463.npy')/10
    
    elif SDSS == True:
        fits_file = fits.open('%s/%s'%(folder,file_name))
        wave = (10**fits_file[1].data['loglam'])/10
        flux = fits_file[1].data['flux'] *10e-17
        fits_file.close()
        
    elif Seyfert == True:
        fits_file = fits.open('%s/%s'%(folder,file_name))
        flux = fits_file[0].data
        hk_or_zj = file_name.split('_')[1]
        if hk_or_zj == 'hk':
            start_wl = 1388.996248
            del_wl = 0.950505
            flux = flux[0,:,:][0]
            wave = np.arange(start_wl,start_wl+(del_wl*len(flux)),del_wl)
        elif hk_or_zj == 'zj':
            start_wl = 883.697206
            del_wl = 0.596855
            wave = np.arange(start_wl,start_wl+(del_wl*len(flux)),del_wl)
        fits_file.close()
    
    elif stand_star == True:
        data = np.loadtxt('%s/%s'%(folder,file_name))
        wave = data[:,0]/10
        flux = data[:,1]
    
    elif JWST == True:
        file_fits = fits.open('%s/%s'%(folder,file_name))
        redshift_found = False
        redshift_coord = 0
        while redshift_found == False:
            if file_fits[3].data[redshift_coord][1] >= redshift:
                redshift_found = True
            else:
                redshift_coord += 1
        wave = file_fits[2].data
        flux = file_fits[1].data[redshift_coord]
        redshift_check = file_fits[3].data[redshift_coord][1]
        print('\nRedshift of data spectrum is',redshift_check)
        file_fits.close()
    
    elif quasar == True:
        file_fits = fits.open('%s/%s'%(folder,file_name))
        wave = file_fits[1].data['WAVE']/10
        flux = file_fits[1].data['FLUX']*1e-19
        
    
    else:
        file_path = folder + '/' + file_name
        fits_file = fits.open(file_path)
            
        data_fits = fits_file[1].data
        
        flux = abs(data_fits.field(1)[0])
        wave = data_fits.field(0)[0]
        
        fits_file.close()
    
    
    
    if plotting == True:   
        plt.figure()
        plt.plot(wave,flux,'r-',markersize=1,label='Data')
        plt.xlabel(object_x)
        plt.ylabel(object_y)
        plt.title('%s spectrum'%file_name[:-5])
        plt.legend(loc='best')
        
    return wave,flux

def data_extractor_TLUSTY(file_name,row,plotting=False):
    in_spec = np.load('%s/%s.NPY'%(folder,file_name))
    out_spec = np.zeros((2,len(in_spec[:,0])))
    
    out_spec[0,:] += in_spec[:,row]
    out_spec[1,:] += in_spec[:,-1]
    
    if plotting == True:
        plt.figure()
        plt.plot(out_spec[0],out_spec[1],'r-',label='%s input spectrum'%file_name)
        plt.xlabel(object_x)
        plt.ylabel(object_y)
        plt.legend(loc='best')
    
    return out_spec


def data_extractor_TLUSTY_joint_spec(file_name1,file_name2,row,plotting=False):
    in_spec = np.load('%s/%s.NPY'%(folder,file_name1))
    out_spec = np.zeros((2,len(in_spec[:,0])))
    
    out_spec[0,:] += in_spec[:,row]
    out_spec[1,:] += in_spec[:,-1]
    
    in_spec2 = np.load('%s/%s.NPY'%(folder,file_name2))
    out_spec2 = np.zeros((2,len(in_spec2[:,0])))
    
    out_spec2[0,:] += in_spec2[:,row]
    out_spec2[1,:] += in_spec2[:,-1]
    
    coord = nearest(out_spec2[0],out_spec[0][0],'coord')
    
    if coord == 0:
        out_spec[1] += out_spec2[1]
    else:
        out_spec[1,:-coord] += out_spec2[1,coord:]
    
    
    if plotting == True:
        plt.figure()
        plt.plot(out_spec[0],out_spec[1],'r-',label='%s input spectrum'%(file_name1+file_name2))
        plt.xlabel(object_x)
        plt.ylabel(object_y)
        plt.legend(loc='best')
    
    return out_spec




##########################################################################################################################################
#Loading the incoming sky spectrum
###############################################################################################################################################

def sky_spectrum_load(plotting=False):
    
    eso_sky = ESO_sky_model(raw_sky_file) #loading in sky file 

    if plotting == True:
        plt.figure()
        plt.plot(eso_sky[0],eso_sky[1],label='Sky Spectrum')
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Photon count')

    return eso_sky



##########################################################################################################################################
#Atmosphere file loading and application to spectra
############################################################################################################################################

def atmospheric_effects_data(spec):

    with open('Atmosphere_Data/optical_extinction_curve_gemini.txt','r') as f:             
        opt_ex_curve = f.readlines()                                        #
        f.close()                                                           #    
                                                                            #
    opt_ex_curve_numpy_array = np.zeros((len(opt_ex_curve),2))              #
                                                                            #
    for i in range(len(opt_ex_curve)):                                      #OPTICAL TRANSMISSIONS, CONVERTING FROM EXCTINCTION CURVE TO TRANSMISSION
        opt_ex_curve_numpy_array[i,:] = opt_ex_curve[i].split(' ')          #extracting model from data file
                                                                            #
    trans_opt = np.zeros((len(opt_ex_curve),2))                             #
    trans_opt[:,0] = opt_ex_curve_numpy_array[:,0]                          #
    trans_opt[:,1] = np.exp(-(opt_ex_curve_numpy_array[:,1]*airmass)/2.5)             
    
    
    
    
    with open('Atmosphere_Data/cptrans_zm_43_15.dat','r') as f:                             
        nir_trans = f.readlines()                                           #
        f.close()                                                           #
                                                                            #
    nir_trans_numpy_array = np.zeros((len(nir_trans),2))                    #
                                                                            #
    start = 0                                                               #
                                                                            #
    for i in range(len(nir_trans)):                                         #
                                                                            #
        line = nir_trans[i].split(' ')                                      #            
                                                                            #NIR TRANSMISSIONS 
        if line[start] == '':                                               #extracting model from data filw    
            start += 1                                                      #
                                                                            #
        nir_trans_numpy_array[i,:] = line[start],line[-1]                   #
                                                                            #
    nir_trans_numpy_array[:,0] *= 1000                                      #
    
    cutoff = where_in_array(nir_trans_numpy_array[:,0],lambda_high_val)[0]+1#cutoff is because there may be uneeded data, file goes to 6um
    
    tot_trans = np.concatenate((trans_opt,nir_trans_numpy_array[:cutoff])) #combining optical and NIR
    
    x_fit = tot_trans[:,0]
    y_fit = tot_trans[:,1] #setting up the x and y data to use for the interpolation.
    
    x_data = np.zeros_like(spec[0]) + spec[0]
    x_data[x_data > lambda_high_val] = lambda_high_val
    x_data[x_data < lambda_low_val] = lambda_low_val
    
    transmissions = interpolate.interp1d(x_fit,y_fit) #getting the transmissions across the spectrum data wavelengths
    transmissions = transmissions(x_data)
    return transmissions,x_data,tot_trans
    
def atmospheric_effects(spec,plotting=False,return_trans=False):

    transmissions,x_data,tot_trans = atmospheric_effects_data(spec)
    
    new_fluxes = spec[1] * transmissions
    
    if plotting == True:
        
        plt.figure()
        plt.plot(spec[0],spec[1],'r-',markersize='1',label='Spectrum from file')
        plt.plot(x_data,new_fluxes,'b-',markersize='1',alpha=0.6,label='Spectrum post atmosphere')
        plt.xlabel(object_x)
        plt.ylabel(object_y)
        plt.legend(loc='best')
        
        plt.figure()
        plt.plot(tot_trans[:,0],tot_trans[:,1],'b-',label='Transmission data')
        plt.xlabel(object_x)
        plt.ylabel('Transmission')
        plt.title('Atmsopheric transmission from Gemini South Cerra Pachon data')
        plt.legend(loc='best')
        
    if return_trans == True:
        return (x_data,new_fluxes),transmissions
    else:
        return x_data,new_fluxes


###################################################################################################################################################################################
#Slit effects calculation
########################################################################################################################################################################################


def sigma_calc(s,lambdas,airmass): #the data used (sky, spec, atmos, etc) have been done with an airmass ~1.5
    
    D = mirr_diam / 100 #converting mirror diameter to metres
    L0 = 46 #in metres, this is the wavefront outer length-scale of the atmospheric turbulence above Paranal
            #if L0 is infinity then this corresponds to pure Kolomogorov turbulence
    
    F_Kolb = (1. / (1. + (300.*D/L0))) - 1  #Kolb factor

    r0 = (0.976 * 500e-9 * (180/3.14) * 3600) * (1/s) * ((lambdas/500)**1.2) * (airmass**-0.6) #Fried parameter at the requested wavelength and airmass
    
    FWHM_atm = s * (airmass**0.6) * ((lambdas/500)**-0.2) * np.sqrt( 1 + F_Kolb * 2.183 * ((r0/L0)**0.356) ) #FWHM as a result of atmosphere with requested seeing, wavelengths, and airmass
    
    FWHM_tel = 1.028*((lambdas*1e-9)/D)*(180/np.pi)*(3600) #Telescope transfer function
    
    FWHM_ins = 0.45 #the transfer function value of XShooter
    
    FWHM = FWHM = np.sqrt( FWHM_atm**2 + FWHM_tel**2 + FWHM_ins**2 )
    
    sigmas = FWHM / (2*np.sqrt(2*np.log(2))) #calculating sigma using eqn from GIGA-Z paper
    
    return sigmas

def spec_seeing(spec,plotting=False): #for single spaxel in slit
    
    object_x = np.linspace(-seeing*2,seeing*2,10000) + (off_centre * seeing) #x coordinates of intensity gaussian with off centre factor
    
    mean = (np.max(object_x) + np.min(object_x)) / 2       #finding central point of gaussian
    
    sigmas = sigma_calc(seeing,spec[0],airmass) #finding the sigma value for the intensity gaussians depending on wavelength
    
    slit_trans = []
    
    for i in range(len(sigmas)):
        
        object_y = gaussian_eq(np.linspace(-seeing*2,seeing*2,10000),mu=mean,sig=sigmas[i]) #y values for intensity (arbitrary units)
        
        norm_obj_y = object_y / np.sum(object_y) #normalisation
        
        coords_high_y = nearest(object_x,pix_fov,'coord') #matching up the slit edges with the gaussian points
        coords_low_y = nearest(object_x,-(pix_fov),'coord')
        
        slit_y = norm_obj_y[coords_low_y:coords_high_y+1]    #Finding the extent of the slit along the length of the slit
        
        slit_trans.append(np.sum(slit_y))  #summing whats left of normalised gaussian
        
        perc_done = ((i+1)/len(sigmas))*100
        print('\r%.2f %% of wavelengths complete'%(perc_done),end='',flush=True)
    
    slit_trans = np.copy(np.asarray(slit_trans))
    spec_out = np.zeros_like(spec)
    spec_out[0] += spec[0]
    spec_out[1] += spec[1]*slit_trans
    
    if plotting == True:
        plt.figure()
        plt.plot(spec_out[0],slit_trans,label='Slit transmission')
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Transmission')
        plt.legend(loc='best')
    
    return spec_out,(spec[0],slit_trans)




##########################################################################################################################################
#finds the value or its position in an array that is closest to a given value
#########################################################################################################################################

def nearest(x,value,val_or_coord):
    coord = np.abs(x-value).argmin()
    val = x[coord]
    if val_or_coord == 'coord':
        return coord
    if val_or_coord == 'val':
        return val



##########################################################################################################################################
#Telescope, optics transmission
#########################################################################################################################################

def telescope_effects(spec,thorlabs=False,plotting=False):
    
    if thorlabs == True:
        transmissions = telescope_effects_data(spec)
        post_tele = spec[1] * (transmissions/100) #applying interpolation to data
    else:
        gemini_data = np.loadtxt('Mirror_Materials/gemini_mirrors_Ag_coating_data_nm.txt')
        #print(gemini_data)
        #print(np.shape(gemini_data))
        #print(spec[0])
        transmissions = np.interp(spec[0],gemini_data[:,0][::-1],gemini_data[:,1][::-1])
        post_tele = spec[1] * (transmissions/100) #applying interpolation to data
        #print(transmissions)

    if plotting == True:
        
        plt.figure()
        plt.plot(spec[0],spec[1],'r-',markersize='1',label='Photon spectrum before telescope')
        plt.plot(spec[0],post_tele,'b-',markersize='1',alpha=0.6,label='Spectrum post telescope')
        plt.xlabel(object_x)        
        plt.ylabel('Photon number')
        #plt.title('Photon spectrum before and after telescope')
        plt.legend(loc='best')
    
    return spec[0],post_tele


def optics_transmission(spec,number_surfaces):
    gemini_data = np.loadtxt('Mirror_Materials/gemini_mirrors_Ag_coating_data_nm.txt')
    transmissions = (np.interp(spec[0],gemini_data[:,0][::-1],gemini_data[:,1][::-1])/100)**number_surfaces
    post_optics = spec[1] * transmissions
    return spec[0],post_optics



##########################################################################################################################################
#Generate 1D spectrum from orders and order wavelengths seen by MKIDs
#########################################################################################################################################


def wavelength_array_maker(w_o):
    
    wavelengths = np.zeros( ((len(w_o[:,0])*len(w_o[0,:]))) ) #1d array which will hold all the wavelengths includng doubles and overlaps
    
    for i in range(len(w_o[:,0])):
        
        coord_start = i * len(w_o[0,:])
        coord_end = coord_start + len(w_o[0,:])             #for loop adds each orders wavelength to the array
        
        wavelengths[coord_start:coord_end] = w_o[i,:]
    
    wavelengths = np.unique(np.sort(wavelengths)) #sorts wavelengths in ascending order and then removes doubles
    
    mkid_spec = np.zeros((2,len(wavelengths)))
    
    mkid_spec[0] += wavelengths #returning to having [0] being the wavelengths and [1] will have photon counts/flux
    
    return mkid_spec


##########################################################################################################################################
#SNR calculations
#########################################################################################################################################


def SNR_calc(spec,sky,plotting=False): #spectra in photons ideally
    
    spec_lam = np.zeros(len(spec[0])) #data stores
    spec_flux = np.zeros(len(spec[0]))
    sky_lam = sky[0]
    sky_flux = sky[1]
    
    spec_lam += spec[0] #applying data to new stores
    spec_flux += spec[1]
    
    SNRs = np.zeros((2,len(spec_flux)))
    
    SNRs[0] += spec_lam
    SNRs[1] = nan_replacer(spec_flux / np.sqrt(spec_flux + sky_flux),single_val=False) #calculation of SNR for each wavelength
    
    
    SNR = np.sqrt(np.sum(SNRs[1]**2)) #summing SNRs together for overall SNR
    
    if plotting == True:
        plt.figure()
        plt.plot(SNRs[0],SNRs[1],'b-',label='SNRs across bins')
        plt.xlabel('Wavelength / nm')
        plt.ylabel('SNR')
        plt.title('Total SNR = %.2f'%SNR)
        plt.legend(loc='best')
        
    return SNR,SNRs

def SNR_calc_grid(spec,sky,plotting=False): #spectra in photons ideally
    
    SNRs = np.zeros_like(spec)
    
    SNRs += spec / np.sqrt(spec + sky) #calculation of SNR for each wavelength
    
    for i in range(len(SNRs)):
        SNRs[i] = np.nan_to_num(SNRs[i])
        SNRs[i][SNRs[i] < -1e300] = 0
        SNRs[i][SNRs[i] > 1e300] = 0
    
    SNR = np.sqrt(np.sum(SNRs**2)) #summing SNRs together for overall SNR
    
    if plotting == True:
        plt.figure()
        plt.imshow(SNRs, interpolation='nearest', aspect='auto',cmap='plasma')
        plt.title('Total SNR = %.2f'%SNR)
        plt.colorbar()
        
        plt.figure()
        plt.imshow(spec, interpolation='nearest', aspect='auto',cmap='plasma')
        plt.title('Raw output spectrum')
        plt.colorbar()
        
        plt.figure()
        plt.imshow(sky, interpolation='nearest', aspect='auto',cmap='plasma')
        plt.title('Raw sky output spectrum')
        plt.colorbar()
        
    return SNR,SNRs
    

def SNR_calc_pred(obj,sky,SOX=False,plotting=False): #spectra in photons ideally
    
    SNRs = np.zeros((2,len(obj[0])))
    
    SNRs[0] += obj[0]
    SNRs[1] = obj[1] / np.sqrt(obj[1] + sky[1]) #calculation of SNR for each wavelength
    
    #obj_interp = np.zeros((2,1000000))
    #obj_interp[0] += np.linspace(obj[0][0],obj[0][-1],1000000)
    #obj_interp[1] += np.interp(obj_interp[0],obj[0],obj[-1])
    #bin_size = obj_interp[0][-1] / 5600
    #obj_res = rebinner(obj_interp,bin_size,obj_interp[0][0],obj_interp[0][-1])
    
    #sky_interp = np.zeros((2,1000000))
    #sky_interp[0] += np.linspace(sky[0][0],sky[0][-1],1000000)
    #sky_interp[1] += np.interp(sky_interp[0],sky[0],sky[-1])
    #bin_size = sky_interp[0][-1] / 5600
    #sky_res = rebinner(sky_interp,bin_size,sky_interp[0][0],sky_interp[0][-1])
    #SNR = np.sqrt(np.sum(SNRs[1]**2)) #summing SNRs together for overall SNR
    if SOX == True:
        SNRs_inst= obj[1] / np.sqrt(obj[1] + sky[1] + (6*(3**2)) + (6*0.0*exposure_t))
        inst = 'SOXS'
        #VIS arm readout noise 3 e-
        #VIS arm dark current negligible e-/s
        #VIS arm pixels over which the sky signal is integrated  6 pixels
        #NIR arm readout noise 7 e-
        #NIR arm dark current 0.1 e-/s
        #NIR arm pixels over which the sky signal is integrated  4 pixels
        
    else:
        SNRs_inst = obj[1] / np.sqrt(obj[1] + sky[1] + (15*0.000722222*exposure_t) + (15*(3.4**2)))
        inst = 'X-Shooter'
        #VIS arm readout noise 3.4 e-
        #VIS arm dark current 0.000722222 e-/s
        #VIS arm pixels over which the sky signal is integrated  15 pixels
        #NIR arm readout noise 9.9 e-
        #NIR arm dark current 0.0125 e-/s
        #NIR arm pixels over which the sky signal is integrated  8 pixels
    
    
    if plotting == True:
        plt.figure()
        plt.plot(SNRs[0],SNRs[1],'b-',label='KIDSpec')
        plt.plot(obj[0],SNRs_inst,'r-',alpha=0.6,label=inst)
        plt.xlabel('Wavelength / nm')
        plt.ylabel('SNR')
        #plt.title('Total SNR = %.2f'%SNR)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('KIDSpec_%s_%s_pred_SNR.pdf'%(inst,object_name))

        
    return SNRs_inst


def SNR_calc_pred_grid(obj,sky,SOX=False,plotting=False): #spectra in photons ideally
    
    SNRs = np.zeros_like(obj)
    
    SNRs += obj / np.sqrt(obj + sky) #calculation of SNR for each wavelength
    
    #obj_interp = np.zeros((2,1000000))
    #obj_interp[0] += np.linspace(obj[0][0],obj[0][-1],1000000)
    #obj_interp[1] += np.interp(obj_interp[0],obj[0],obj[-1])
    #bin_size = obj_interp[0][-1] / 5600
    #obj_res = rebinner(obj_interp,bin_size,obj_interp[0][0],obj_interp[0][-1])
    
    #sky_interp = np.zeros((2,1000000))
    #sky_interp[0] += np.linspace(sky[0][0],sky[0][-1],1000000)
    #sky_interp[1] += np.interp(sky_interp[0],sky[0],sky[-1])
    #bin_size = sky_interp[0][-1] / 5600
    #sky_res = rebinner(sky_interp,bin_size,sky_interp[0][0],sky_interp[0][-1])
    #SNR = np.sqrt(np.sum(SNRs[1]**2)) #summing SNRs together for overall SNR
    
    if SOX == True:
        #SNRs_inst = obj / np.sqrt(obj + sky + (4*(7**2)) + (4*0.1*exposure_t))
        SNRs_inst = obj / np.sqrt(obj + sky + (6*(3**2)) + (6*0.0*exposure_t))
        inst = 'SOXS'
        #VIS arm readout noise 3 e-
        #VIS arm dark current negligible e-/s
        #VIS arm pixels over which the sky signal is integrated  6 pixels
        #NIR arm readout noise 7 e-
        #NIR arm dark current 0.1 e-/s
        #NIR arm pixels over which the sky signal is integrated  4 pixels
        
    else:
        #SNRs_inst = obj / np.sqrt(obj + sky + (8*0.0125*exposure_t) + (8*(9.9**2)))
        SNRs_inst = obj / np.sqrt(obj + sky + (15*0.000722222*exposure_t) + (15*(3.4**2)))
        inst = 'X-Shooter'
        #VIS arm readout noise 3.4 e-
        #VIS arm dark current 0.000722222 e-/s
        #VIS arm pixels over which the sky signal is integrated  15 pixels
        #NIR arm readout noise 9.9 e-
        #NIR arm dark current 0.0125 e-/s
        #NIR arm pixels over which the sky signal is integrated  8 pixels
    
    
    if plotting == True:
        plt.figure()
        plt.imshow(SNRs_inst, interpolation='nearest', aspect='auto',cmap='plasma')
        plt.title('Predicted SNRs for %s'%inst)
        plt.colorbar()
        #plt.savefig('KIDSpec_%s_%s_pred_SNR.pdf'%(inst,object_name))
        
    return SNRs_inst

##########################################################################################################################################
#Vega system magnitude calculations
#########################################################################################################################################


def mag_calc(spec,wls_check=False,return_flux=False,plotting=False):
    
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
    flux_incoming = []
    
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
            
            final_mag = (-2.5*np.log10(f_mag_res/f_mag_zero))#(-2.5*np.log10(f_mag_res/f_mag_zero))
            
            #final_mag = -2.5*np.log10(np.sum(filtered_spec)/zero_fluxes[i])
            
            obj_mags.append(final_mag)
            flux_incoming.append(f_mag_res)
            
            if plotting == True:
                
                fig1 = plt.figure()
                plt.plot(spec[0],filtered_spec,'b-')
                plt.xlabel('Wavelength /nm')
                plt.ylabel('Flux / $ergcm^{-2}s^{-1}\AA^{-1}$')
                fig1.text(0.15,0.82,'%s mag = %.2f'%(mags[i],final_mag))
    
        else:
            
            obj_mags.append('N/A')
            flux_incoming.append(0)
        
    if wls_check == True:
        return obj_mags,mags,bandpass_wls,bandpass_trans
    elif return_flux == True:
        central_wls = []
        for i in range(len(bandpass_wls)):
            central_wls.append((bandpass_wls[i][0]+bandpass_wls[i][-1])/2)
        return obj_mags,mags,np.asarray(central_wls),flux_incoming
    else:
        return obj_mags,mags




##########################################################################################################################################
#Rebinning with set included bins
#########################################################################################################################################

def rebinner_with_bins(spec,bins):
    new_spec = np.zeros((2,len(bins)))
    new_spec[0] += bins
    for i in range(len(spec[0])):
        coord_low = np.where(spec[0][i] >= bins)[0]
        #print(coord_low)
        new_spec[1][coord_low[-1]] += spec[1][i]
        
    return new_spec   


##########################################################################################################################################
#Rebinning of 2D array, with predefined number of bins
#########################################################################################################################################

def rebinner_2d(spec,wls,no_bins):
    
    new_spec = np.zeros((len(spec),no_bins+1))
    new_wls = np.zeros_like(new_spec)
    
    for i in range(len(spec)):
        new_wls[i] += np.linspace(wls[i][0],wls[i][-1],no_bins+1)
        for j in range(len(wls[i])):
            coord_low = np.where(wls[i,j] >= new_wls[i])[0][-1]
            
            new_spec[i,coord_low] += spec[i,j]
    
    return new_spec,new_wls
        



##########################################################################################################################################
#R value statistic
#########################################################################################################################################


def R_value(sim_spec,data_spec,plotting=False):
    
    x = sim_spec[1][~np.isnan(sim_spec[1])]
    x_1 = x
    y = data_spec[1][~np.isnan(sim_spec[1])]
    x = x[x < 1e308]
    x_2 = x
    x = x[x > -1e308]
    y = y[x_1 < 1e308]
    y = y[x_2 > -1e308]
    N = len(sim_spec[1])
    
    numerator = (N*np.sum(x*y)) - (np.sum(x)*np.sum(y))
    denominator = np.sqrt( (N*np.sum(x**2) - (np.sum(x)**2)) * (N*np.sum(y**2) - (np.sum(y)**2)) )
    
    if plotting == True:
        plt.figure()
        fac = 1
        #plt.plot(sim_spec[1],data_spec[1],'ko',markersize=3)
        plt.scatter(sim_spec[1]/fac,data_spec[1]/fac,s=2,c=sim_spec[0],cmap='plasma')
        plt.plot(np.array([min(sim_spec[1]),max(sim_spec[1])])/fac,np.array([min(sim_spec[1]),max(sim_spec[1])])/fac,'k--',linewidth=2)
        #plt.xlabel(r'Order Gaussian method flux / %s$\times 10^{-12}$'%object_y[7:])
        #plt.ylabel(r'PTS method flux / %s$\times 10^{-12}$'%object_y[7:])
        plt.xlabel('KSIM flux / %s'%object_y)
        plt.ylabel('Data flux / %s'%object_y)
        plt.colorbar(label='Wavelength / nm')
        plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
        plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
                          
    return numerator / denominator




##########################################################################################################################################
#Redshift applier
#########################################################################################################################################

def redshifter(spec,curr_z,desired_z):
    curr_wls = spec[0]*10 #in Angstrom
    curr_flux = spec[1]
    d_L_pre_rshift = cosmo.luminosity_distance(curr_z).to("cm").value
    curr_L = 4*np.pi*curr_flux*(d_L_pre_rshift**2)
    
    # redshift the wavelengths.
    a = 1 + desired_z
    rshift_wls = (curr_wls/(1+curr_z)) * a
    
    d_L = cosmo.luminosity_distance(desired_z).to("cm").value
    #rshift_flux = curr_L / (4 * a * np.pi * d_L**2)
    rshift_flux = curr_L / (4 * np.pi * d_L**2)
    
    redshifted_spec = np.zeros_like(spec)
    redshifted_spec[0] += rshift_wls/10
    redshifted_spec[1] += rshift_flux
    
    return redshifted_spec



#########################################################################################################################################
#finds where a value exists in an array
#########################################################################################################################################

def where_in_array(arr,value):
    coords = np.where(arr == value)[0]
    return coords


#########################################################################################################################################
#gaussian equation
#########################################################################################################################################

def gaussian_eq(x,mu=0.,sig=1.):
    return (1 / sig*np.sqrt(2*np.pi)) * np.exp( (x-mu)**2 / (-2. * sig**2) )


#########################################################################################################################################
#sorts through an array and replaces nans with numbers
#########################################################################################################################################
    
def nan_replacer(arr,single_val=False):
    
    if single_val == True:
        
        if np.isnan(arr):
            new_val = np.nan_to_num(arr)
            return new_val
        else:
            return arr
    
    
    if (np.isnan(arr)).any():
        new_arr = np.nan_to_num(arr)
        return new_arr
    else:
        return arr



#########################################################################################################################################
#order merger
#########################################################################################################################################


def order_merge_reg_grid(wls,resps,del_lam=None):
    
    if del_lam == None:
        del_lams = np.zeros(len(wls))
        for i in range(len(wls)):
            del_lams[i] += (wls[i,-1]-wls[i,0])/len(wls[i])
        min_del_lam = np.min(del_lams)
        del_lam = min_del_lam/reg_grid_factor
    
    out_spec_wls = np.arange(start=lambda_low_val,stop=lambda_high_val,step=del_lam)
    out_spec_counts = np.zeros_like(out_spec_wls)
    
    for order in range(len(wls)):
        for lam in range(len(wls[order])):
            
            if resps[order,lam] != 0:
                
                if lam == len(wls[order])-1:
                    curr_del_lam = wls[order,lam]-wls[order,lam-1]
                else:
                    curr_del_lam = wls[order,lam+1]-wls[order,lam]
                
                low_range = nearest(out_spec_wls,wls[order,lam]-curr_del_lam,'coord')
                high_range = nearest(out_spec_wls,wls[order,lam]+curr_del_lam,'coord')
                diff_range = (curr_del_lam*2) / del_lam
                
                out_spec_counts[low_range:high_range+1] += resps[order,lam]/diff_range
            
    out_spec = np.zeros((2,len(out_spec_wls)))
    out_spec[0] += out_spec_wls
    out_spec[1] += out_spec_counts
    
    return out_spec




#########################################################################################################################################
#plotting a grid (2D array) by row
#########################################################################################################################################


def grid_plotter(gridx,gridy):
    
    plt.figure()
    for i in range(len(gridx)):
        plt.plot(gridx[i],gridy[i])
    plt.xlabel('Wavelength / nm')
    plt.ylabel('Photon count')
    
    return

def grid_plotter_opp(gridx,gridy):
    
    plt.figure()
    for i in range(len(gridx)):
        plt.plot(gridx[i],gridy[:,i])
    plt.xlabel('Wavelength / nm')
    plt.ylabel('Photon count')
    
    return




#########################################################################################################################################
#interpolating data spectra to simulation format
#########################################################################################################################################

def model_interpolator(spec,no_points):
    
    binstep_interpolation = ((spec[0][-1]-spec[0][0]) / no_points)
    binstep_factor = (binstep/1e-7) / binstep_interpolation
    f_int = interpolate.interp1d(spec[0], spec[1]/binstep_factor, bounds_error = False, fill_value = 0) #interpolating data
    
    spec_out = np.zeros((2,no_points))
    spec_out[0] += np.linspace(spec[0][0],spec[0][-1],no_points)
    spec_out[1] += f_int(spec_out[0])
    
    spec_out[1] /= np.sum(spec_out[1])/np.sum(spec[1])

    return spec_out

def model_interpolator_sky(spec,no_points):
    
    binstep_interpolation = ((spec[0][-1]-spec[0][0]) / no_points)
    binstep_sky = ((spec[0][-1]-spec[0][0]) / len(spec[0]))
    binstep_factor = binstep_sky / binstep_interpolation
    f_int = interpolate.interp1d(spec[0], spec[1]/binstep_factor, bounds_error = False, fill_value = 0) #interpolating data
    
    spec_out = np.zeros((2,no_points))
    spec_out[0] += np.linspace(spec[0][0],spec[0][-1],no_points)
    spec_out[1] += f_int(spec_out[0])
    
    spec_out[1] /= np.sum(spec_out[1])/np.sum(spec[1])

    return spec_out


#########################################################################################################################################
#KSIM express function
#########################################################################################################################################


def KIDSpec_Express(pix_sum,pix_sky,ord_waves,pixels,bad_pix=[False,1.0],R_E_shift=[False,0],IR=False):
    
    if IR == True:
        ARM = 'IR'
    else:
        ARM = 'OPT'
    
    
    print('\n Beginning %s MKIDs pixel response estimation:\n'%ARM)
    
    file_path = '%s/Resample/'%(folder)
    
    R_E_data = np.array([[16.404,13.411],[4.238,2.478],[4.083,2.297],[4.016,2.249],[4.065,2.306],[4.059,2.274]])
    R_E_pix = np.array([1500,3000])
    R_E_vals = np.array([10,20,30,40,50,60])
    close_pixels = nearest(R_E_pix,pixels,'coord')
    R_E_trend = interpolate.interp1d(R_E_vals,R_E_data[:,close_pixels])
    
    R_E_store = np.zeros_like(pix_sum)
    
    int_steps = np.ndarray.astype((np.linspace(0,n_pixels,100)),dtype='int') 
    prog = 0
    
    for pix in range(pixels):
        
        pix_order_ph = np.zeros((2,len(pix_sum[pix,:])))
        pix_order_mis = np.zeros((2,len(pix_sum[pix,:])))
    
        
        for i in range(len(pix_sum[0,:])):
            diff_poiss_sky = pix_sky[pix,i]
            if pix_sum[pix,i]-pix_sky[pix,i] < 0:
                diff_poiss_obj = 0
            else:
                diff_poiss_obj = pix_sum[pix,i]-pix_sky[pix,i]
            
            pix_sky[pix,i] = diff_poiss_sky
            #print('SKY',pix_sky[pix,i])
            pix_sum[pix,i] = diff_poiss_sky + diff_poiss_obj
            #print('OBJ',pix_sum[pix,i])
            
            if pix_sum[pix,i] < 0.0:
                pix_sum[pix,i] = 0
            
            
        if bad_pix[0] == True:
            pix_sum[pix,:] *= bad_pix[1]
            pix_sky[pix,:] *= bad_pix[1]
            
        
        
        PIX_Rs =  ER_band_low / np.sqrt(ord_waves[:,pix]/400)
        
        
        if R_E_shift[0] == True:
            for i in range(len(PIX_Rs)):
                PIX_Rs[i] += ((-1*R_E_shift[1]) + (np.random.random()*(R_E_shift[1]*2)))
                if PIX_Rs[i] < min(PIX_Rs[i]):
                    PIX_Rs[i] = min(PIX_Rs[i])
                R_E_store[pix,i] += PIX_Rs[i]
        else:
            R_E_store[pix,:] += PIX_Rs
            
            
        #sky and object portion
        
        
        pix_order_ph[0,:] += ord_waves[:,pix]
        pix_order_mis[0,:] += ord_waves[:,pix]
        
        for i in range(len(PIX_Rs)):
            id_photons = 0
            mis_photons = 0
            if PIX_Rs[i] < min(R_E_vals):
                PIX_Rs[i] = min(R_E_vals)
            id_photons += pix_sum[pix,i] * (1-(R_E_trend(PIX_Rs[i])/100))
            mis_photons += pix_sum[pix,i] - id_photons
            
            pix_order_mis[1,i] += mis_photons
            pix_order_ph[1,i] += id_photons
            
            up_and_down_mis = np.random.random()
            up_mis = up_and_down_mis*mis_photons
            down_mis = mis_photons - up_mis
            
            if i == 0:
                pix_order_ph[1,i] += down_mis
                pix_order_ph[1,i+1] += up_mis
            elif i == (len(PIX_Rs)-1):
                pix_order_ph[1,i] += up_mis
                pix_order_ph[1,i-1] += down_mis
            else:
                pix_order_ph[1,i-1] += down_mis
                pix_order_ph[1,i+1] += up_mis
        
        for i in range(len(pix_order_ph[1])):
            if pix_sum[pix,i] == 0:
                pix_order_ph[1,i] = 0
        
        if IR == True:
            np.save('%s/spectrum_order_pixel_IR_%i.npy'%(file_path,pix),pix_order_ph)
            np.save('%s/spectrum_misident_pixel_IR_%i.npy'%(file_path,pix),pix_order_mis)
        else:
            #print('OBJ',pix_order_ph)
            np.save('%s/spectrum_order_pixel_OPT_%i.npy'%(file_path,pix),pix_order_ph)
            np.save('%s/spectrum_misident_pixel_OPT_%i.npy'%(file_path,pix),pix_order_mis)
    
        
            
            
            
        #Sky portion
        
        
        pix_order_ph = np.zeros((2,len(pix_sum[pix,:])))
        pix_order_mis = np.zeros((2,len(pix_sum[pix,:])))
        
        pix_order_ph[0,:] += ord_waves[:,pix]
        pix_order_mis[0,:] += ord_waves[:,pix]
        for i in range(len(PIX_Rs)):
            id_photons = 0
            mis_photons = 0
            id_photons += pix_sky[pix,i] * (1-(R_E_trend(PIX_Rs[i])/100))
            mis_photons += pix_sky[pix,i] - id_photons
            
            pix_order_mis[1,i] += mis_photons
            pix_order_ph[1,i] += id_photons
            
            up_and_down_mis = np.random.random()
            up_mis = up_and_down_mis*mis_photons
            down_mis = mis_photons - up_mis
            
            if i == 0:
                pix_order_ph[1,i] += down_mis
                pix_order_ph[1,i+1] += up_mis
            elif i == (len(PIX_Rs)-1):
                pix_order_ph[1,i] += up_mis
                pix_order_ph[1,i-1] += down_mis
            else:
                pix_order_ph[1,i-1] += down_mis
                pix_order_ph[1,i+1] += up_mis
        
        for i in range(len(pix_order_ph[1])):
            if pix_sky[pix,i] == 0:
                pix_order_ph[1,i] = 0
        
        
        
        if IR == True:
            np.save('%s/spectrum_order_pixel_IR_sky_%i.npy'%(file_path,pix),pix_order_ph)
            np.save('%s/spectrum_misident_pixel_IR_sky_%i.npy'%(file_path,pix),pix_order_mis)
        else:
            #print('SKY',pix_order_ph)
            np.save('%s/spectrum_order_pixel_OPT_sky_%i.npy'%(file_path,pix),pix_order_ph)
            np.save('%s/spectrum_misident_pixel_OPT_sky_%i.npy'%(file_path,pix),pix_order_mis)
    
            
        if (pix == int_steps).any():
            prog += 1
            print('\r%i%% of pixels complete.'%prog,end='',flush=True)
            
            
            
    print('\r100% of pixels complete.',end='',flush=True)
    return 
    

#########################################################################################################################################
#Photon Timestream Simulation (PTS)
#########################################################################################################################################

#simulating photon stream--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def photon_stream_SIM(pixel_sum,w_o,orders,R,plotting=False):
    
    data = np.zeros((len(w_o),3)) #setting up array of info including wavelengths, orders, and pixel counts
    data[:,0] = orders
    data[:,1] = w_o
    data[:,2] = pixel_sum
    
    len_ord = int(len(data[:,0]))

    time_res = 1/1e6 #1 microsecond time res assumed
    
    time_steps = (exposure_t / time_res) / 100
    
    photon_stream = np.zeros((2+len_ord,int(time_steps))) #setting up arrays to store time photon stream
    photon_stream[0] = np.linspace(0,time_steps,int(time_steps))
    
    count_stream = np.zeros_like(photon_stream) #array to store count
    count_stream[0] = photon_stream[0]
    
    gauss_features = np.zeros((len_ord,2)) #storing mean and 1-sigma of gaussians for each order
    
    #Starting photon stream simulation
    for k in range(len_ord):
        
        mu = data[k,1] #mean of gaussian
        tot_count = data[k,2] #expected count of photon arrivals over entire exposure time
        
        expect_count = tot_count/time_steps #expected count of photon arrivals over one time bin
        
        #MKID gaussian
        KID_R =  R / np.sqrt(mu/400) #energy resolution of the current pixel and the wavelengths it observes
        
        sig = mu / (KID_R * (2*np.sqrt( 2*np.log(2) )))  #1-sigma calculation using an equation from 'GIGA-Z: A 100,000 OBJECT SUPERCONDUCTING SPECTROPHOTOMETER FOR LSST FOLLOW-UP' 
        
        gauss_features[k,:] = np.array([mu,sig])
        #print('POINT TRACK 1')
        photon_arrival_array = random_photon_arrival_generator(expect_count,tot_count,int(time_steps))
        gauss_arrival_array_1s = np.random.normal(loc=mu,scale=sig,size=int(np.sum(photon_arrival_array[photon_arrival_array == 1])))
        gauss_arrival_array_2s = ( 
            np.random.normal(loc=mu,scale=sig,size=int(len(photon_arrival_array[photon_arrival_array == 2]))) + 
            np.random.normal(loc=mu,scale=sig,size=int(len(photon_arrival_array[photon_arrival_array == 2])))
            )
        
        #print('POINT TRACK 2')
        photon_stream[k+1][photon_arrival_array == 1] += gauss_arrival_array_1s 
        photon_stream[k+1][photon_arrival_array == 2] += gauss_arrival_array_2s 
        count_stream[k+1] += photon_arrival_array

    #print('POINT TRACK 3')
    for i in range(int(time_steps)):
        if len(np.nonzero(photon_stream[:,i])[0]) > 0:
            photon_stream[-1,i] =  energy_totaler(photon_stream[1:-1,i],gauss_features[:,0]) #summing the photons for each time step
    count_stream[-1] += np.sum(count_stream[1:]) #summing the counts for each time step
    
    #print('POINT TRACK 4')
    count_sum = np.zeros((len(count_stream[1:-1,0]),2)) #seperate array for the count sum
    count_sum[:,0] = gauss_features[:,0]
    for i in range(len(count_stream[1:-1,0])):          #summing 
        count_sum[i,1] = np.sum(count_stream[i+1])
    
    return gauss_features,photon_stream,count_stream,data,count_sum

def random_photon_arrival_generator(expected_per_bin,total_count,bins):
    time_array = np.zeros(int(bins))

    while np.sum(time_array) < total_count:
        random_coord = int(len(time_array)*np.random.random())
        time_array[random_coord] += 1 #+np.random.poisson(lam=expected_per_bin)

    return time_array

#energy conversion and back to the lambda the total energy corresponds to------------------------------------------------------------------------------------$

def energy_totaler(array,orig_lam):
    non_zero_arr = []
    orig_lam_non_zero = []
    for i in range(len(array)):
        if array[i] != 0.0:
            non_zero_arr.append(array[i])
            orig_lam_non_zero.append(orig_lam[i])

    if len(non_zero_arr) == 0.0:
        return 0

    orig_lam = np.asarray(orig_lam_non_zero)
    array = np.asarray(non_zero_arr)
    indiv_lam = array #(orig_lam**2)/array
    E = np.zeros(len(indiv_lam))
    for i in range(len(E)):
        E[i] = ((6.63e-34)*(3e8)) / indiv_lam[i]
    total_lam = ((6.63e-34)*(3e8)) / np.sum(E)
    return total_lam



#processing the generated photon stream------------------------------------------------------------------------------------------------------------------------------------------------------------------

def photon_assigner(gauss_features,photon_stream):
    
    photon_result = np.zeros((len(gauss_features[:,0])+1,len(photon_stream[0]))) #array to store the resulting photon stream after MKID
    photon_result[0] = photon_stream[0]
    
    gaussian_objects = []
    for i in range(len(gauss_features[:,0])):
        gaussian_objects.append(scipy.stats.norm(gauss_features[i,0],gauss_features[i,1])) #generates a gaussian object for the current order
    
    for i in range(len(photon_stream[0])): #looping through time steps
        
        if photon_stream[-1,i] != 0: #checking photon arrivals occurred
            probs = np.zeros(len(gauss_features[:,0])) #array to store probabilities of photon events belonging to particular orders
            for j in range(len(gauss_features[:,0])):
                probs[j] = gaussian_objects[j].pdf(photon_stream[-1,i]) #calculates probability
            
            max_prob = probs.argmax()
            photon_result[max_prob+1,i] = gauss_features[max_prob,0] #maximum probability order added to result
        else:
            photon_result[1:,i] = 0
    
    return photon_result



#reformatting the photon result for use in rest of simulation-------------------------------------------------------------------------------------------------------------------------------------------------------


def photon_result_processor(photon_result,count_stream,gauss_features):
    
    sat_photon = 0
    
    if (count_stream[-1] > 1).any():
        sat_photon += 1
    
    pix_order_ph = np.zeros((2,len(gauss_features[:,0])))
    pix_order_mis = np.zeros((2,len(gauss_features[:,0]))) #arrays generated in form accepted by subsequent functions in simulation
    
    pix_order_ph[0] = gauss_features[:,0] #adding wavelengths
    pix_order_mis[0] = gauss_features[:,0]
    
    for i in range(len(gauss_features[:,0])):
        count_ord = np.sum(photon_result[i+1])/gauss_features[i,0] #extracting count of photon result 
        count_diff = np.abs(np.sum(count_stream[i+1])-count_ord) #extracting misidentified photons by calculating difference in photon count before MKID and after
        
        pix_order_ph[1,i] = count_ord
        pix_order_mis[1,i] = count_diff #adding to resulting arrays
    
    return pix_order_ph,pix_order_mis,sat_photon

def photon_result_processor_with_orig(photon_result,count_stream,gauss_features):

    sat_photon = 0

    if (count_stream[-1] >= 2).any():
        sat_photon += 1

    pix_order_ph = np.zeros((2,len(gauss_features[:,0])))
    pix_order_mis = np.zeros((2,len(gauss_features[:,0]))) #arrays generated in form accepted by subsequent functions in simulation
    pix_order_orig = np.zeros((2,len(gauss_features[:,0])))

    pix_order_ph[0] = gauss_features[:,0] #adding wavelengths
    pix_order_mis[0] = gauss_features[:,0]
    pix_order_orig[0] = gauss_features[:,0]

    for i in range(len(gauss_features[:,0])):
        count_ord = np.sum(photon_result[i+1])/gauss_features[i,0] #extracting count of photon result
        count_diff = np.sum(count_stream[i+1])-count_ord #extracting misidentified photons by calculating difference in photon count before MKID and after
        count_orig = np.sum(count_stream[i+1])

        pix_order_ph[1,i] = count_ord
        pix_order_mis[1,i] = count_diff #adding to resulting arrays
        pix_order_orig[1,i] = count_orig

    return pix_order_ph,pix_order_mis,pix_order_orig,sat_photon


#multiprocessing function for photon stream----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def multiprocessing_PTS_IR(pixel_sum,w_o,orders,R,i,file_path_overlap):
    gauss_features,photon_stream,count_stream,data,count_sum = photon_stream_SIM(pixel_sum,w_o,orders,R,plotting=False)
    photon_result = photon_assigner(gauss_features,photon_stream)
    pix_order_ph,pix_order_mis,pix_order_orig,sat_ph = photon_result_processor_with_orig(photon_result,count_stream,gauss_features)
    np.save('%s/spectrum_order_pixel_IR_%i.npy'%(file_path_overlap,i),pix_order_ph)
    np.save('%s/spectrum_misident_pixel_IR_%i.npy'%(file_path_overlap,i),pix_order_mis)
    np.save('%s/orig_photons_pixel_IR_%i.npy'%(file_path_overlap,i),pix_order_orig)
    '''
    pixel_h5_data = wave_to_phase(photon_stream)
    with h5py.File('PIXEL_PHASE_DATA_SCIENCE.hdf5','a') as hf:
        hf.create_dataset('IR_pixel_%i'%i, data=pixel_h5_data, dtype='f8')
    '''
    f_count = open('PHOTON_PIX/Pixel_%i_IR_%i_counts.txt'%(i,np.sum(pix_order_ph[1])),'w+')
    f_count.close()
    if sat_ph > 0:
        f = open('SAT_PIX/Pixel_%i_IR_%i_time_bins.txt'%(i,sat_ph),'w+')
        f.close()
        #print('\nPHOTONS PIXEL %i:'%i,pix_order_ph)
        #print('\nMIS PIXEL %i:'%i,pix_order_mis)
    with open('%sIR.txt'%file_path_overlap, 'r+') as f:
        f.write('\n%i'%i)
        pix_done = len(f.readlines())
        f.close()
    perc_done = ((pix_done)/n_pixels)*100

    print('\r%.2f %% of pixels complete'%(perc_done),end='',flush=True)
    
    #if (int(perc_done) == np.array([10,20,30,40,50,60,70,80,90])).any():
    #    print('%.1f%% of pixels complete.'%perc_done)
    return


def multiprocessing_PTS_OPT(pixel_sum,w_o,orders,R,i,file_path_overlap):
    gauss_features,photon_stream,count_stream,data,count_sum = photon_stream_SIM(pixel_sum,w_o,orders,R,plotting=False)
    photon_result = photon_assigner(gauss_features,photon_stream)
    pix_order_ph,pix_order_mis,pix_order_orig,sat_ph = photon_result_processor_with_orig(photon_result,count_stream,gauss_features)
    np.save('%s/spectrum_order_pixel_OPT_%i.npy'%(file_path_overlap,i),pix_order_ph)
    np.save('%s/spectrum_misident_pixel_OPT_%i.npy'%(file_path_overlap,i),pix_order_mis)
    np.save('%s/orig_photons_pixel_OPT_%i.npy'%(file_path_overlap,i),pix_order_orig)
    '''
    pixel_h5_data = wave_to_phase(photon_stream)
    with h5py.File('PIXEL_PHASE_DATA_SCIENCE.hdf5','a') as hf:
        hf.create_dataset('OPT_pixel_%i'%i, data=pixel_h5_data, dtype='f8')
    '''
    f_count = open('PHOTON_PIX/Pixel_%i_OPT_%i_counts.txt'%(i,np.sum(pix_order_ph[1])),'w+')
    f_count.close()
    if sat_ph > 0:
        f = open('SAT_PIX/Pixel_%i_OPT_%i_time_bins.txt'%(i,sat_ph),'w+')
        f.close()

    #perc_done = ((len(os.listdir('%s/'%file_path_overlap)))/(n_pixels*3))*100
    #if (int(perc_done) == np.array([10,20,30,40,50,60,70,80,90])).any():
    #    print('%.1f%% of pixels complete.'%perc_done)

    with open('%sOPT.txt'%file_path_overlap, 'r+') as f:
        f.write('\n%i'%i)
        pix_done = len(f.readlines())
        f.close()
    perc_done = (pix_done/n_pixels)*100
    print('\r%.2f %% of pixels complete'%(perc_done),end='',flush=True)
    #if (int(perc_done) == np.array([10,20,30,40,50,60,70,80,90])).any():
    #    print('%.1f%% of pixels complete.'%perc_done)
    return



def multiprocessing_PTS_IR_sky(pixel_sum,w_o,orders,R,i,file_path_overlap):
    gauss_features,photon_stream,count_stream,data,count_sum = photon_stream_SIM(pixel_sum,w_o,orders,R,plotting=False)
    photon_result = photon_assigner(gauss_features,photon_stream)
    pix_order_ph,pix_order_mis,sat_ph = photon_result_processor(photon_result,count_stream,gauss_features)
    
    '''
    pixel_h5_data = wave_to_phase(photon_stream)
    with h5py.File('PIXEL_PHASE_DATA_SKY.hdf5','a') as hf:
        hf.create_dataset('IR_SKY_pixel_%i'%i, data=pixel_h5_data, dtype='f8')
    '''
    np.save('%s/spectrum_order_pixel_IR_sky_%i.npy'%(file_path_overlap,i),pix_order_ph)
    np.save('%s/spectrum_misident_pixel_IR_sky_%i.npy'%(file_path_overlap,i),pix_order_mis)
    
    with open('%sIR_sky.txt'%file_path_overlap, 'r+') as f:
        f.write('\n%i'%i)
        pix_done = len(f.readlines())
        f.close()
    perc_done = ((pix_done)/n_pixels)*100
    
    print('\r%.2f %% of pixels complete'%(perc_done),end='',flush=True)
    
    return 

def multiprocessing_PTS_OPT_sky(pixel_sum,w_o,orders,R,i,file_path_overlap):
    gauss_features,photon_stream,count_stream,data,count_sum = photon_stream_SIM(pixel_sum,w_o,orders,R,plotting=False)
    photon_result = photon_assigner(gauss_features,photon_stream)
    pix_order_ph,pix_order_mis,sat_ph = photon_result_processor(photon_result,count_stream,gauss_features)
    '''
    pixel_h5_data = wave_to_phase(photon_stream)
    with h5py.File('PIXEL_PHASE_DATA_SKY.hdf5','a') as hf:
        hf.create_dataset('OPT_SKY_pixel_%i'%i, data=pixel_h5_data, dtype='f8')
    '''
    np.save('%s/spectrum_order_pixel_OPT_sky_%i.npy'%(file_path_overlap,i),pix_order_ph)
    np.save('%s/spectrum_misident_pixel_OPT_sky_%i.npy'%(file_path_overlap,i),pix_order_mis)
    
    with open('%sOPT_sky.txt'%file_path_overlap, 'r+') as f:
        f.write('\n%i'%i)
        pix_done = len(f.readlines())
        f.close()
    perc_done = ((pix_done)/n_pixels)*100
    
    print('\r%.2f %% of pixels complete'%(perc_done),end='',flush=True)
    
    return 


#multiproccessing use-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def multiprocessing_use(jobs,cpu_count,IR=False,sky=False):
    pool = mp.Pool(cpu_count)
    if IR == True:
        if sky == True:
            pool.starmap(multiprocessing_PTS_IR_sky,jobs)
        else:
            pool.starmap(multiprocessing_PTS_IR,jobs)
    elif IR == False:
        if sky == True:
            pool.starmap(multiprocessing_PTS_OPT_sky,jobs)
        else:
            pool.starmap(multiprocessing_PTS_OPT,jobs)
    return


#########################################################################################################################################
#Photometry incoming flux 
#########################################################################################################################################


def spec_slicer_photometry(spec,return_dimensions=False,photon_noise=True):
    
    object_x = np.linspace(-seeing*3,seeing*3,10000) + (off_centre * seeing) #x coordinates of intensity gaussian with off centre factor
    
    mean = (np.max(object_x) + np.min(object_x)) / 2       #finding central point of gaussian
    
    sigmas = sigma_calc(seeing,spec[0],airmass) #finding the sigma value for the intensity gaussians depending on wavelength
    
    interval_factors_opt = [] #storing the transmissions
    interval_factors_ir = []
    interval_factors = []
    no_pixels_ir = []
    no_pixels_opt = []
    x_range_opt = []
    x_range_ir = []
    y_range_opt = []
    y_range_ir = []
    x_range = []
    y_range = []
    
    plot_amount = 0
    
    print('\nBeginning seeing transmission calculations for spectrum.')
    for i in range(len(sigmas)):
        
        if spec[0][i] > 1000:
            pix_width = 0.9 #arcsec
            pix_length = 0.2
        elif spec[0][i] < 1000:
            pix_width = 0.45
            pix_length = 0.15
        
        object_y_pre = gaussian_eq(np.linspace(-seeing*3,seeing*3,10000),mu=mean,sig=sigmas[i]) #y values for intensity (arbitrary units)
        object_y = object_y_pre / np.sum(object_y_pre) #normalisation
        
        object_y_min = np.max(object_y)*0.01
        
        object_y_range = (np.where(object_y_min > object_y[:int(len(object_y)/2)])[0][-1],int(len(object_y)/2)+np.where(object_y_min > object_y[int(len(object_y)/2):])[0][0])
        object_x_range = abs(object_x[object_y_range[0]] - object_x[object_y_range[1]])
        
        pixels_area = np.pi * np.square((0.5*object_x_range))
        #no_pixels.append(pixels_area/(pix_width*pix_length))
        
        x_range.append(object_x_range)
        y_range.append(object_y_range)
        interval_factors.append(object_y[object_y_range[0]:object_y_range[1]])
        
        if spec[0][i] > 1000:
            interval_factors_ir.append(object_y[object_y_range[0]:object_y_range[1]])
            interval_factors_opt.append(0)
            no_pixels_ir.append(np.sqrt(pixels_area/(pix_width*pix_length)))
            x_range_ir.append(object_x_range)
            y_range_ir.append(object_y_range)
        elif spec[0][i] < 1000:
            interval_factors_opt.append(object_y[object_y_range[0]:object_y_range[1]])
            interval_factors_ir.append(0)
            no_pixels_opt.append(np.sqrt(pixels_area/(pix_width*pix_length)))
            x_range_opt.append(object_x_range)
            y_range_opt.append(object_y_range)
        
        perc_done = ((i+1)/len(sigmas))*100
        print('\r%.2f %% of wavelengths complete'%(perc_done),end='',flush=True)
    
    opt_max_pix = int(max(no_pixels_opt))+1
    ir_max_pix = int(max(no_pixels_ir))+1
    opt_max_pix_range = object_x[y_range_opt[np.argmax(no_pixels_opt)][0]:y_range_opt[np.argmax(no_pixels_opt)][1]]
    ir_max_pix_range = object_x[y_range_ir[np.argmax(no_pixels_ir)][0]:y_range_ir[np.argmax(no_pixels_ir)][1]]
    
    bins_opt = np.linspace(opt_max_pix_range[0],opt_max_pix_range[-1],opt_max_pix)
    bins_ir = np.linspace(ir_max_pix_range[0],ir_max_pix_range[-1],ir_max_pix)
    
    transmission_cube_opt = np.zeros((opt_max_pix,opt_max_pix,len(sigmas)))
    transmission_cube_ir = np.zeros((ir_max_pix,ir_max_pix,len(sigmas)))
    
    print('\nBeginning specific pixel transmission calculations.')
    for i in range(len(sigmas)):
        
        y_sec = interval_factors[i]
        x_sec = object_x[y_range[i][0]:y_range[i][1]]
        
        if spec[0][i] > 1000:
            rebinned_interval = rebinner_with_bins((x_sec,y_sec),bins_ir)
            x, y = np.meshgrid(rebinned_interval[0],rebinned_interval[0])
            dst = np.sqrt(x*x+y*y)
            gauss = np.zeros_like(dst)
            for j in range(len(rebinned_interval[0])):
                gauss[j] += np.interp(dst[j],rebinned_interval[0],rebinned_interval[1])
            transmission_cube_ir[:,:,i] += (gauss/np.sum(gauss))
        elif spec[0][i] < 1000:
            rebinned_interval = rebinner_with_bins((x_sec,y_sec),bins_opt)
            x, y = np.meshgrid(rebinned_interval[0],rebinned_interval[0])
            dst = np.sqrt(x*x+y*y)
            gauss = np.zeros_like(dst)
            for j in range(len(rebinned_interval[0])):
                gauss[j] += np.interp(dst[j],rebinned_interval[0],rebinned_interval[1])
            transmission_cube_opt[:,:,i] += (gauss/np.sum(gauss))
        
        perc_done = ((i+1)/len(sigmas))*100
        print('\r%.2f %% of wavelengths complete'%(perc_done),end='',flush=True)
    
    print('\nApplying spectrum to pixels.')
    
    pixel_specs_ir = np.zeros_like(transmission_cube_ir)
    pixel_specs_opt = np.zeros_like(transmission_cube_opt)
    pixel_specs_ir += transmission_cube_ir
    pixel_specs_opt += transmission_cube_opt
    
    for i in range(len(spec[0])):
        if photon_noise == True:
            if spec[0][i] > 1000:
                pixel_specs_ir[:,:,i] *= (np.random.poisson(spec[1][i]))
            elif spec[0][i] < 1000:
                pixel_specs_opt[:,:,i] *= (np.random.poisson(spec[1][i]))
        else:
            if spec[0][i] > 1000:
                pixel_specs_ir[:,:,i] *= spec[1][i]
            elif spec[0][i] < 1000:
                pixel_specs_opt[:,:,i] *= spec[1][i]

    if return_dimensions == True:
        print('\nOptical incoming flux width:',len(pixel_specs_opt[:,0,0])*0.45,'x',len(pixel_specs_opt[:,0,0])*0.15,'arcseconds')
        print('\nInfrared incoming flux width:',len(pixel_specs_ir[:,0,0])*0.9,'x',len(pixel_specs_ir[:,0,0])*0.2,'arcseconds')
    return pixel_specs_opt,pixel_specs_ir,transmission_cube_opt,transmission_cube_ir



#########################################################################################################################################
#Bin array onto 3D array, for photometry
#########################################################################################################################################


def three_d_rebinner(spec,wls):
    
    new_spec = np.zeros((len(spec[:,0,0]),len(spec[0,:,0]),14))
    
    for i in range(len(spec[:,0,0])):
        
        for j in range(len(spec[0,:,0])):
            
            _,_,_,spec_sec = mag_calc((wls,spec[i,j,:]),plotting=False,return_flux=True)
            new_spec[i,j,:] += spec_sec
    
    return new_spec


#########################################################################################################################################
#FWHM calculator via fitting
#########################################################################################################################################

def gaussian_2(x,mu,sig,offset,amp):
    return (-amp * np.exp( (x-mu)**2 / (-2. * sig**2) )) + offset

def lorentzian(x,mu,amp,offset,fwhm):
    return ((-amp * ((fwhm/2)) / np.pi) / (  np.square(x-mu) + np.square((fwhm/2))  )) + offset

def exp_func(x,mu,amp,offset,fac):
    return (-amp * np.exp(-fac*abs(mu-x))) + offset

def straight_line_continuum_removal(x,m,c):
    return (m*x) + c

def polynomial_line_continuum_removal(x,a,b,c):
    return (a*np.square(x)) + (b*x) + c

def continuum_removal(spec,poly=False):
    continuum_removed_spec = np.copy(spec)
    if poly == True:
        popt,pcov = curve_fit(polynomial_line_continuum_removal,spec[0],spec[1])
        continuum_removed_spec[1] -= polynomial_line_continuum_removal(spec[0],*popt)
    else:
        popt,pcov = curve_fit(straight_line_continuum_removal,spec[0],spec[1])
        continuum_removed_spec[1] -= straight_line_continuum_removal(spec[0],*popt)
    return continuum_removed_spec


def fwhm_fitter_exp(data_x, data_y,mu,amp,offset,fac):
    popt,pcov = curve_fit(exp_func,data_x,data_y,p0=[mu,amp,offset,fac],maxfev=10000)
    fwhm = popt[3]
    fwhm_error = np.sqrt(pcov[3,3])
    print('\nFWHM:',fwhm,'+/-',fwhm_error)
    plt.figure()
    plt.plot(data_x,data_y,'rx',label='Data')
    plt.plot(data_x,exp_func(data_x,*popt),'b-',alpha=0.7,label='Fit')
    plt.xlabel('Wavelength / nm')
    plt.ylabel(object_y)
    plt.legend(loc='best')
    return fwhm,fwhm_error


def fwhm_fitter_lorentzian(data_x, data_y,mu,amp,offset,fwhm):
    
    popt,pcov = curve_fit(lorentzian,data_x,data_y,p0=[mu,amp,offset,fwhm],maxfev=10000)
    fwhm = popt[3]
    fwhm_error = np.sqrt(pcov[3,3])
    print('\nFWHM:',fwhm,'+/-',fwhm_error)
    
    plt.figure()
    plt.plot(data_x,data_y,'rx',label='Data')
    plt.plot(np.linspace(data_x[0],data_x[-1],100000),lorentzian(np.linspace(data_x[0],data_x[-1],100000),*popt),'b-',alpha=0.7,label='Fit')
    plt.xlabel('Wavelength / nm')
    plt.ylabel(object_y)
    plt.legend(loc='best')
    return fwhm,fwhm_error


def fwhm_fitter_gaussian(data_x, data_y,mu,amp,offset,sig):
    popt,pcov = curve_fit(gaussian_2,data_x,data_y,p0=[mu,sig,offset,amp],maxfev=10000)
    sigma = popt[1]
    sigma_error = np.sqrt(pcov[1,1])
    fwhm = 2*np.sqrt(2*np.log(2))*sigma
    fwhm_error = 2*np.sqrt(2*np.log(2))*sigma_error
    print('\nFWHM:',fwhm,'+/-',fwhm_error)
    plt.figure()
    plt.plot(data_x,data_y,'rx',label='Data')
    plt.plot(np.linspace(data_x[0],data_x[-1],100000),gaussian_2(np.linspace(data_x[0],data_x[-1],100000),*popt),'b-',alpha=0.7,label='Fit')
    plt.xlabel('Wavelength / nm')
    plt.ylabel(object_y)
    plt.legend(loc='best')
    return fwhm,fwhm_error


'''
plt.rcParams.update({'font.size': 20})
fig,ax = plt.subplots(1,3)


mu = cen_wl
amp = pts[1][coord_feature]
offset = pts[1][coord_feature]
fwhm = 2

data_x = pts[0,coord_feature-coord_range:coord_feature+coord_range]
data_y = pts[1,coord_feature-coord_range:coord_feature+coord_range]
popt,pcov = curve_fit(lorentzian,data_x,data_y,p0=[mu,amp,offset,fwhm],maxfev=10000)
fwhm = popt[3]
fwhm_error = np.sqrt(pcov[3,3])
print('\nFWHM:',fwhm,'+/-',fwhm_error)


ax[0].plot(data_x,data_y,'rx',markersize=3,label='PTS simulation')
ax[0].plot(data_x,lorentzian(data_x,*popt),'b-',alpha=0.7,label='Fit')
#plt.xlabel('Wavelength / nm')
#plt.ylabel(object_y)
#ax[0].legend(loc='best')
ax[0].set_ylim(-1.3e-12,0)
ax[0].set_ylabel(object_y)

amp = mod[1][coord_feature]
offset = mod[1][coord_feature]
data_x = mod[0,coord_feature-coord_range:coord_feature+coord_range]
data_y = mod[1,coord_feature-coord_range:coord_feature+coord_range]
popt,pcov = curve_fit(lorentzian,data_x,data_y,p0=[mu,amp,offset,fwhm],maxfev=10000)
fwhm = popt[3]
fwhm_error = np.sqrt(pcov[3,3])
print('\nFWHM:',fwhm,'+/-',fwhm_error)


ax[1].plot(data_x,data_y,'rx',markersize=3,label='HD212442 spectrum')
ax[1].plot(data_x,lorentzian(data_x,*popt),'b-',alpha=0.7,label='Fit')
#ax[1].legend(loc='best')
ax[1].set_ylim(-1.3e-12,0)
ax[1].tick_params(labelleft=False, left=False)
ax[1].set_xlabel('Wavelength / nm')

amp = ordg[1][coord_feature]
offset = ordg[1][coord_feature]
data_x = ordg[0,coord_feature-coord_range:coord_feature+coord_range]
data_y = ordg[1,coord_feature-coord_range:coord_feature+coord_range]
popt,pcov = curve_fit(lorentzian,data_x,data_y,p0=[mu,amp,offset,fwhm],maxfev=10000)
fwhm = popt[3]
fwhm_error = np.sqrt(pcov[3,3])
print('\nFWHM:',fwhm,'+/-',fwhm_error)


ax[2].plot(data_x,data_y,'rx',markersize=3,label='Order Gaussian simulation')
ax[2].plot(data_x,lorentzian(data_x,*popt),'b-',alpha=0.7,label='Fit')
#ax[2].legend(loc='best')
ax[2].set_ylim(-1.3e-12,0)
ax[2].tick_params(labelleft=False, left=False)
    
'''





