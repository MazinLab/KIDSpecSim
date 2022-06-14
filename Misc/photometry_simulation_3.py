# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:40:14 2022

@author: BVH
"""
import numpy as np
import datetime
import os
import scipy.stats
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import shutil

from scipy import interpolate

from useful_funcs import data_extractor, telescope_effects, optics_transmission, wavelength_array_maker, \
        sky_spectrum_load,SNR_calc,nearest,mag_calc, \
            data_extractor_TLUSTY, data_extractor_TLUSTY_joint_spec,spec_seeing,SNR_calc_pred, \
                atmospheric_effects,rebinner_with_bins, \
                    R_value,redshifter,order_merge_reg_grid,grid_plotter,SNR_calc_grid,grid_plotter_opp, \
                        model_interpolator,spec_slicer_photometry,three_d_rebinner
from parameters import *
from flux_to_photons_conversion import photons_conversion,flux_conversion,flux_conversion_3,flux_conversion_4
from apply_QE import QE

print('\n Importing spectrum from data file.')
if TLUSTY == True:
    #model_spec = data_extractor_TLUSTY_joint_spec(object_file_1,object_file_2,row,plotting=extra_plots)  
    #model_spec = model_spec[:,:150000]
    tlusty_spec = np.loadtxt('SHIFTED_SPECTRA/XS_T100T498_OBJ.txt')
    model_spec = np.zeros((2,len(tlusty_spec[:,0])))
    model_spec[0] += tlusty_spec[:,0]
    model_spec[1] += tlusty_spec[:,1]
else:
    #If data that will be used is from the DRs of XShooter then set XShooter to True, since their FITS files are setup differently to the ESO-XShooter archive
    #Spectrum is in units of Flux / $ergcm^{-2}s^{-1}\AA^{-1}$, the AA is angstrom, wavelength array is in nm
    model_spec = data_extractor(object_file,JWST=True,plotting=extra_plots)
original_spec = np.copy(model_spec)

print('\n Simulating observation of %s.'%object_name)

#sectioning the spectrum to chosen KIDSpec bandpass
model_spec = (model_spec[0],model_spec[1] / mag_reduce) 
low_c = nearest(model_spec[0],lambda_low_val,'coord')               
high_c = nearest(model_spec[0],lambda_high_val,'coord')
model_spec = (model_spec[0][low_c+1:high_c],model_spec[1][low_c+1:high_c])

#converting data spectrum to photons and loading incoming sky spectrum from ESO sky model
photon_spec_no_eff_original = photons_conversion(model_spec,model_spec,plotting=extra_plots)

#increasing number of points in model spectrum
photon_spec_no_eff = np.copy(photon_spec_no_eff_original) #model_interpolator(photon_spec_no_eff_original,100000)

#generating sky
photon_spec_of_sky = sky_spectrum_load(plotting=extra_plots)

#calculating magnitudes of model spectrum
SIM_obj_mags = mag_calc(model_spec,plotting=False,return_flux=True)


##############################################################################################################################################################################
#ATMOSPHERE TRANSMISSION, TELESCOPE TRANSMISSION, SLIT EFFECTS, QE
##############################################################################################################################################################################

print('\n Applying atmospheric, telescope, slit, and QE effects.')

photon_spec_post_atmos = atmospheric_effects(photon_spec_no_eff,plotting=extra_plots,return_trans=False)
photon_sky_post_atmos = atmospheric_effects(photon_spec_of_sky,plotting=False,return_trans=False)

photon_spec_pre_optics = telescope_effects(photon_spec_post_atmos,plotting=extra_plots) 
photon_sky_pre_optics = telescope_effects(photon_sky_post_atmos,plotting=False)

photon_spec_to_instr = optics_transmission(photon_spec_pre_optics,6)
photon_sky_to_instr = optics_transmission(photon_sky_pre_optics,6)

print('\nSimulating seeing.')
photon_spec_opt,photon_spec_ir,seeing_transmiss_opt_photo,seeing_transmiss_ir_photo = spec_slicer_photometry(photon_spec_to_instr,return_dimensions=True)
print('\nAgain for incoming sky.')
sky_spectrum_opt,sky_spectrum_ir,_,_ = spec_slicer_photometry(photon_sky_to_instr)

sky_spectrum_opt_sep,sky_spectrum_ir_sep,_,_ = spec_slicer_photometry(photon_sky_to_instr,photon_noise=False)

qe_vals = np.array([0.78,0.78,0.91,0.74,0.23,0.85,0.85])
qe_wls = np.array([400,550,700,900,1000,1001,3000])
qe = np.interp(photon_spec_to_instr[0],qe_wls,qe_vals)
qe_sky = np.interp(photon_sky_to_instr[0],qe_wls,qe_vals)

photon_spec_opt *= qe
sky_spectrum_opt *= qe_sky
sky_spectrum_opt_sep *= qe_sky
photon_spec_ir *= qe
sky_spectrum_ir *= qe_sky
sky_spectrum_ir_sep *= qe_sky

print('\nFinalising incoming photons and beginning post arrival processing.')

photon_spec_opt = three_d_rebinner(photon_spec_opt,model_spec[0])
photon_spec_ir = three_d_rebinner(photon_spec_ir,model_spec[0])
sky_spectrum_opt = three_d_rebinner(sky_spectrum_opt,photon_spec_of_sky[0])
sky_spectrum_ir = three_d_rebinner(sky_spectrum_ir,photon_spec_of_sky[0])
sky_spectrum_opt_sep = three_d_rebinner(sky_spectrum_opt_sep,photon_spec_of_sky[0])
sky_spectrum_ir_sep = three_d_rebinner(sky_spectrum_ir_sep,photon_spec_of_sky[0])

print('\nApplying read and dark current noise. ')

photon_spec_opt += np.random.poisson((3.4/0.64) + (0.0007222/0.64 * exposure_t)) 
sky_spectrum_opt += np.random.poisson((3.4/0.64) + (0.0007222/0.64 * exposure_t)) 
sky_spectrum_opt_sep += ((3.4/0.64) + (0.0007222/0.64 * exposure_t)) 
photon_spec_ir += np.random.poisson((8/2.29) + (0.0125/2.29 * exposure_t)) 
sky_spectrum_ir += np.random.poisson((8/2.29) + (0.0125/2.29 * exposure_t)) 
sky_spectrum_ir_sep += ((8/2.29) + (0.0125/2.29 * exposure_t)) 

incoming_total_photon_spec_opt = np.zeros_like(photon_spec_opt)
incoming_total_photon_spec_ir = np.zeros_like(photon_spec_ir)
incoming_total_photon_spec_opt += (photon_spec_opt+sky_spectrum_opt)
incoming_total_photon_spec_ir += (photon_spec_ir+sky_spectrum_ir)


#POST PROCESSING--------------------------------------------------------------------------------------------------
print('\nSubtracting average of sky spectrum and read noise.')

sky_spec_load_opt = np.copy(sky_spectrum_opt_sep)
sky_spec_load_ir = np.copy(sky_spectrum_ir_sep)

sky_subbed_spec_opt = np.zeros_like(incoming_total_photon_spec_opt)
sky_subbed_spec_opt = incoming_total_photon_spec_opt - sky_spec_load_opt - ((3.4/0.64) + (0.0007222/0.64 * exposure_t))
sky_subbed_spec_ir = np.zeros_like(incoming_total_photon_spec_ir)
sky_subbed_spec_ir = incoming_total_photon_spec_ir - sky_spec_load_ir - ((8/2.29) + (0.0125/2.29 * exposure_t))

corrected_spec_opt = np.zeros_like(sky_subbed_spec_opt)
corrected_spec_opt += sky_subbed_spec_opt
corrected_spec_ir = np.zeros_like(sky_subbed_spec_ir)
corrected_spec_ir += sky_subbed_spec_ir

for i in range(len(corrected_spec_opt[:,0,0])):
    for j in range(len(corrected_spec_opt[:,0,0])):
        corrected_spec_opt[i,j,:][corrected_spec_opt[i,j,:] > 1e308] = 0
        corrected_spec_opt[i,j,:][corrected_spec_opt[i,j,:] < -1e308] = 0
        corrected_spec_opt[i,j,:] = np.nan_to_num(corrected_spec_opt[i,j,:])

for i in range(len(corrected_spec_ir[:,0,0])):
    for j in range(len(corrected_spec_ir[:,0,0])):
        corrected_spec_ir[i,j,:][corrected_spec_ir[i,j,:] > 1e308] = 0
        corrected_spec_ir[i,j,:][corrected_spec_ir[i,j,:] < -1e308] = 0
        corrected_spec_ir[i,j,:] = np.nan_to_num(corrected_spec_ir[i,j,:])


print('\nFinalising.')

full_spec = np.zeros((2,len(SIM_obj_mags[2])))
full_spec[0] += SIM_obj_mags[2]
full_spec[1] += (np.sum(np.sum(corrected_spec_ir,axis=1),axis=0)+np.sum(np.sum(corrected_spec_opt,axis=1),axis=0))#*mag_bins

flux_full_spec = flux_conversion_3(full_spec)

SIM_photon_obj_mags = mag_calc(photon_spec_no_eff,return_flux=True)
data_full_spec = flux_conversion_3((SIM_photon_obj_mags[2],SIM_photon_obj_mags[3]))

factors = np.load('PHOTOMETRY_STAND_STAR_FACTORS.npy')
stand_star_spec = np.load('PHOTOMETRY_STAND_STAR_OUT_FLUXES.npy')

#flux_full_spec[1] *= factors
fac = SIM_obj_mags[3] / flux_full_spec[1]

plt.figure()
plt.plot(flux_full_spec[0],flux_full_spec[1],'r-',label='Simulated data')
plt.plot(flux_full_spec[0],flux_full_spec[1],'ro')
#plt.plot(SIM_obj_mags[2],SIM_obj_mags[3],'b-',label='Model data')
#plt.plot(SIM_obj_mags[2],SIM_obj_mags[3],'bo')
plt.plot(data_full_spec[0],data_full_spec[1],'g-',label='Model data')
plt.plot(data_full_spec[0],data_full_spec[1],'go')
plt.xlabel('Wavelength / nm')
plt.ylabel(object_y)
plt.legend(loc='best')





