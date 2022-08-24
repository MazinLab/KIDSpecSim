# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:52:48 2021

@author: BVH
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

from parameters import *
import sys
sys.path.insert(0,r"C:\Users\BVH\Documents\KineticInductanceDetectors\KIDSpec_Simulator\KSIM\V3_4")
from useful_funcs import data_extractor, telescope_effects, optics_transmission, specific_wl_overlap, wavelength_array_maker, \
        order_doubles, grat_eff_sum, sky_addition,SNR_calc,post_processing,nearest,mag_calc,sky_spec_generator, \
            data_extractor_TLUSTY, data_extractor_TLUSTY_joint_spec,wavelength_finder,cutoff_sorter,spec_slicer_photometry,XS_ETC_spec,SNR_calc_pred, cent_wl_finder, \
                rebinner,atmospheric_effects,cent_wl_overlaps,rebinner_with_bins,pixel_grid_resp,efficiency_interp, \
                    order_merger_max_intensity_alt,eff_reversal,order_merge_reg_grid,post_processing_stand_star,R_value,redshifter, \
                        atmospheric_effects,three_d_rebinner,gaussian_eq
from flux_to_photons_conversion import photons_conversion,flux_conversion_4,flux_conversion_3
sky_save = False


#SEPERATE SKY SPECTRUM---------------------------------------------------------------------------------------------------------

if sky_save == True:    
    print('\nSimulating seperate sky exposure.')    
    data_file_spec = data_extractor(object_file,stand_star=True,plotting=False) 
    original_spec = data_file_spec
    if redshift > 0:
        print('\nRedshifting.')
        data_file_spec = redshifter(data_file_spec,0.200344,redshift)
    low_c = nearest(data_file_spec[0],lambda_low_val,'coord')
    high_c = nearest(data_file_spec[0],lambda_high_val,'coord')
    data_file_spec = (data_file_spec[0][low_c+1:high_c],data_file_spec[1][low_c+1:high_c])
    SIM_obj_mags = mag_calc(data_file_spec,plotting=False,return_flux=True)
    #data_file_spec_photo = (SIM_obj_mags[2],SIM_obj_mags[3])
    photon_spec_no_eff_pre_atm = photons_conversion(data_file_spec,data_file_spec,plotting=False)
    photon_spec_of_sky_pre_atm,sky_spec = sky_addition(photon_spec_no_eff_pre_atm,sky_only=True,plotting=False)
    photon_spec_of_sky,_ = atmospheric_effects(photon_spec_of_sky_pre_atm,plotting=False,return_trans=True)
    photon_spec_numpy = np.zeros((2,len(photon_spec_of_sky[0])))
    photon_spec_numpy[0] += photon_spec_of_sky[0]
    photon_spec_numpy[1] += photon_spec_of_sky[1]
    photon_spec_of_sky = photon_spec_numpy
    sky_spectrum_opt,sky_spectrum_ir,_,_ = spec_slicer_photometry(photon_spec_of_sky,photon_noise=False)
    gemini_data = np.loadtxt('Mirror_Materials/gemini_mirrors_Ag_coating_data_nm.txt')
    surf_transmissions = (np.interp(photon_spec_no_eff_pre_atm[0],gemini_data[:,0][::-1],gemini_data[:,1][::-1])/100)**(7)
    qe_vals = np.array([0.78,0.78,0.91,0.74,0.23,0.85,0.85])
    qe_wls = np.array([400,550,700,900,1000,1001,3000])
    qe = np.interp(photon_spec_no_eff_pre_atm[0],qe_wls,qe_vals)
    sky_spectrum_opt *= (surf_transmissions*qe)
    sky_spectrum_ir *= (surf_transmissions*qe)
    sky_spectrum_opt = three_d_rebinner(sky_spectrum_opt,data_file_spec[0])
    sky_spectrum_ir = three_d_rebinner(sky_spectrum_ir,data_file_spec[0])
    sky_spectrum_opt += (3.4/0.64) + (0.0007222/0.64 * exposure_t)
    sky_spectrum_ir += (8/2.29) + (0.0125/2.29 * exposure_t)
    np.save('Misc/SKY_IR_PHOTOMETRY.npy',sky_spectrum_ir)
    np.save('Misc/SKY_OPT_PHOTOMETRY.npy',sky_spectrum_opt)

#INCOMING SPECTRUM-----------------------------------------------------------------------------------------------------------------

data_file_spec = data_extractor(object_file,stand_star=True,plotting=False) 

original_spec = data_file_spec

if redshift > 0:
    print('\nRedshifting.')
    data_file_spec = redshifter(data_file_spec,0.200344,redshift)


low_c = nearest(data_file_spec[0],lambda_low_val,'coord')
high_c = nearest(data_file_spec[0],lambda_high_val,'coord')
data_file_spec = (data_file_spec[0][low_c+1:high_c],data_file_spec[1][low_c+1:high_c])

bins = np.linspace(lambda_low_val,lambda_high_val,1000)
data_file_spec = rebinner_with_bins(data_file_spec,bins)


SIM_obj_mags = mag_calc(data_file_spec,plotting=False,return_flux=True,wls_check=False)
sigmas = np.array([9.906225017553158,15.178628605486852,
                   15.093651191154265,15.85741651446661,
                   23.00280820329854,25.20130781097441,
                   18.195471194903188,37.39353919950567,
                   22.47564837881215,28.399103109581212,
                   37.69093910494157,30.449462679980027,
                   208.460671738252,1239.98825277682])*2

mag_widths = np.array([93.0,236.0,273.0,560.0,514.0,399.0,200.0,500.05,400.02002,500.31982,1000.75024,998.5298,9000.0,20000.0])
mag_bins = mag_widths / (binstep*1e7)
#data_file_spec_photo = (SIM_obj_mags[2],SIM_obj_mags[3])

print('\nSimulating incoming object and sky photons.')
photon_spec_no_eff_pre_atm = photons_conversion(data_file_spec,data_file_spec,plotting=False)
photon_spec,atm_trans = atmospheric_effects(photon_spec_no_eff_pre_atm,plotting=False,return_trans=True)
photon_spec_numpy = np.zeros((2,len(photon_spec[0])))
photon_spec_numpy[0] += photon_spec[0]
photon_spec_numpy[1] += photon_spec[1]
photon_spec = photon_spec_numpy

photon_spec_of_sky_pre_atm,sky_spec = sky_addition(photon_spec_no_eff_pre_atm,sky_only=True,plotting=False)
photon_spec_of_sky,_ = atmospheric_effects(photon_spec_of_sky_pre_atm,plotting=False,return_trans=True)
photon_spec_numpy = np.zeros((2,len(photon_spec_of_sky[0])))
photon_spec_numpy[0] += photon_spec_of_sky[0]
photon_spec_numpy[1] += photon_spec_of_sky[1]
photon_spec_of_sky = photon_spec_numpy

print('\nSimulating seeing.')
photon_spec_opt,photon_spec_ir,seeing_transmiss_opt_photo,seeing_transmiss_ir_photo = spec_slicer_photometry(photon_spec,return_dimensions=True)
print('\nAgain for incoming sky.')
sky_spectrum_opt,sky_spectrum_ir,_,_ = spec_slicer_photometry(photon_spec_of_sky)

print('\nApplying telescope and optics effects to spectrum.')
gemini_data = np.loadtxt('Mirror_Materials/gemini_mirrors_Ag_coating_data_nm.txt')
surf_transmissions = (np.interp(photon_spec_no_eff_pre_atm[0],gemini_data[:,0][::-1],gemini_data[:,1][::-1])/100)**(7)

qe_vals = np.array([0.78,0.78,0.91,0.74,0.23,0.85,0.85])
qe_wls = np.array([400,550,700,900,1000,1001,3000])
qe = np.interp(photon_spec_no_eff_pre_atm[0],qe_wls,qe_vals)

photon_spec_opt *= (surf_transmissions*qe)
sky_spectrum_opt *= (surf_transmissions*qe)
photon_spec_ir *= (surf_transmissions*qe)
sky_spectrum_ir *= (surf_transmissions*qe)

print('\nFinalising incoming photons and beginning post arrival processing.')

photon_spec_opt = three_d_rebinner(photon_spec_opt,data_file_spec[0])
photon_spec_ir = three_d_rebinner(photon_spec_ir,data_file_spec[0])
sky_spectrum_opt = three_d_rebinner(sky_spectrum_opt,data_file_spec[0])
sky_spectrum_ir = three_d_rebinner(sky_spectrum_ir,data_file_spec[0])

print('\nApplying read and dark current noise. ')

photon_spec_opt += np.random.poisson((3.4/0.64) + (0.0007222/0.64 * exposure_t)) 
sky_spectrum_opt += np.random.poisson((3.4/0.64) + (0.0007222/0.64 * exposure_t)) 
photon_spec_ir += np.random.poisson((8/2.29) + (0.0125/2.29 * exposure_t)) 
sky_spectrum_ir += np.random.poisson((8/2.29) + (0.0125/2.29 * exposure_t)) 

incoming_total_photon_spec_opt = np.zeros_like(photon_spec_opt)
incoming_total_photon_spec_ir = np.zeros_like(photon_spec_ir)
incoming_total_photon_spec_opt += (photon_spec_opt+sky_spectrum_opt)
incoming_total_photon_spec_ir += (photon_spec_ir+sky_spectrum_ir)


#POST PROCESSING--------------------------------------------------------------------------------------------------


sky_spec_load_opt = np.load('Misc/SKY_OPT_PHOTOMETRY.npy')
sky_spec_load_ir = np.load('Misc/SKY_IR_PHOTOMETRY.npy')

sky_subbed_spec_opt = np.zeros_like(incoming_total_photon_spec_opt)
sky_subbed_spec_opt = incoming_total_photon_spec_opt - sky_spec_load_opt - ((3.4/0.64) + (0.0007222/0.64 * exposure_t))
sky_subbed_spec_ir = np.zeros_like(incoming_total_photon_spec_ir)
sky_subbed_spec_ir = incoming_total_photon_spec_ir - sky_spec_load_ir - ((8/2.29) + (0.0125/2.29 * exposure_t))

atm_trans[atm_trans < 0.05] = 0 
atm_trans_2 = np.interp(SIM_obj_mags[2],data_file_spec[0],atm_trans)
qe_2 = np.interp(SIM_obj_mags[2],qe_wls,qe_vals)
surf_transmissions_2 = np.interp(SIM_obj_mags[2],data_file_spec[0],surf_transmissions)

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


full_spec = np.zeros((2,len(SIM_obj_mags[2])))
full_spec[0] += SIM_obj_mags[2]
full_spec[1] += (np.sum(np.sum(corrected_spec_ir,axis=1),axis=0)+np.sum(np.sum(corrected_spec_opt,axis=1),axis=0))#*mag_bins

flux_full_spec = flux_conversion_4(full_spec,sigmas)
#flux_full_spec[1] /= test

SIM_out_obj_mags = mag_calc(flux_full_spec,plotting=False,return_flux=True)
SIM_photon_obj_mags = mag_calc(photon_spec_no_eff_pre_atm,return_flux=True)
data_full_spec = flux_conversion_4((SIM_photon_obj_mags[2],SIM_photon_obj_mags[3]),sigmas)

factors = SIM_obj_mags[3] / flux_full_spec[1]
np.save('PHOTOMETRY_STAND_STAR_FACTORS.npy',factors)
np.save('PHOTOMETRY_STAND_STAR_OUT_FLUXES.npy',flux_full_spec)
np.save('PHOTOMETRY_STAND_STAR_DATA_FLUXES.npy',(SIM_obj_mags[2],SIM_obj_mags[3]))



