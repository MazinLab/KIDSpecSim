# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 09:52:48 2021

@author: BVH
"""


import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from useful_funcs import data_extractor, telescope_effects, optics_transmission, specific_wl_overlap, wavelength_array_maker, \
        order_doubles, grat_eff_sum, sky_addition,SNR_calc,post_processing,nearest,mag_calc,sky_spec_generator, \
            data_extractor_TLUSTY, data_extractor_TLUSTY_joint_spec,wavelength_finder,cutoff_sorter,spec_slicer_photometry,XS_ETC_spec,SNR_calc_pred, cent_wl_finder, \
                rebinner,atmospheric_effects,cent_wl_overlaps,rebinner_with_bins,pixel_grid_resp,efficiency_interp, \
                    order_merger_max_intensity_alt,eff_reversal,order_merge_reg_grid,post_processing_stand_star,R_value,redshifter,atmospheric_effects,mag_calc_bandpass
from flux_to_photons_conversion import photons_conversion,flux_conversion,flux_conversion_3

sky_save = True
save_outputs = True

if sky_save == True:
    data_file_spec = data_extractor(object_file,JWST=True,plotting=False) 
    original_spec = data_file_spec
    #if redshift > 0:
    #    print('\nRedshifting.')
    #    data_file_spec = redshifter(data_file_spec,0.200344,redshift)
    low_c = nearest(data_file_spec[0],lambda_low_val,'coord')
    high_c = nearest(data_file_spec[0],lambda_high_val,'coord')
    data_file_spec = (data_file_spec[0][low_c+1:high_c],data_file_spec[1][low_c+1:high_c])
    #data_file_spec[1][data_file_spec[1]<0] = 0
    #data_file_spec = np.load('JWST_MOCK_SPECTRA/gaussian_spectrum_for_tests.npy')
    SIM_obj_mags = mag_calc(data_file_spec,plotting=False,return_flux=True)
    #data_file_spec_photo = (SIM_obj_mags[2],SIM_obj_mags[3])
    photon_spec_no_eff_pre_atm = photons_conversion(data_file_spec,data_file_spec,plotting=False)
    photon_spec,atm_trans = atmospheric_effects(photon_spec_no_eff_pre_atm,plotting=False,return_trans=True)
    photon_spec_of_sky_pre_atm,sky_spec = sky_addition(photon_spec_no_eff_pre_atm,sky_only=True,plotting=False)
    photon_spec_of_sky,_ = atmospheric_effects(photon_spec_of_sky_pre_atm,plotting=False,return_trans=True)
    sky_spectrum_opt,sky_spectrum_ir,seeing_transmiss_opt,seeing_transmiss_ir = spec_slicer_photometry(photon_spec_of_sky,photon_noise=False)
    ir_coord = int(len(seeing_transmiss_ir[:,0,0])/2)-1
    opt_coord = int(len(seeing_transmiss_opt[:,0,0])/2)-1
    seeing_transmiss_ir = seeing_transmiss_ir[:,ir_coord:ir_coord+2,:]
    seeing_transmiss_ir = np.sum(seeing_transmiss_ir,axis=0)
    seeing_transmiss_opt = seeing_transmiss_opt[:,opt_coord:opt_coord+2,:]
    seeing_transmiss_opt = np.sum(seeing_transmiss_opt,axis=0)
    sky_spectrum_ir = sky_spectrum_ir[:,ir_coord:ir_coord+2,:]
    sky_spectrum_ir = np.sum(sky_spectrum_ir,axis=0)
    sky_spectrum_opt = sky_spectrum_opt[:,opt_coord:opt_coord+2,:]
    sky_spectrum_opt = np.sum(sky_spectrum_opt,axis=0)
    gemini_data = np.loadtxt('Mirror_Materials/gemini_mirrors_Ag_coating_data_nm.txt')
    surf_transmissions = (np.interp(photon_spec_no_eff_pre_atm[0],gemini_data[:,0][::-1],gemini_data[:,1][::-1])/100)**(7)
    nir_QE = 0.85
    opt_QE_vals = np.array([0.78,0.78,0.91,0.74,0.23])
    opt_QE_wls = np.array([400,550,700,900,1000])
    opt_QE = np.interp(photon_spec_no_eff_pre_atm[0],opt_QE_wls,opt_QE_vals)
    sky_spectrum_opt *= (surf_transmissions*opt_QE)
    sky_spectrum_ir *= (surf_transmissions*nir_QE)
    sky_spectrum_opt += (3.4/0.64) + (0.0007222/0.64 * exposure_t)
    sky_spectrum_ir +=  (8/2.29) + (0.0125/2.29 * exposure_t)
    np.save('Misc/SKY_OPT_XS.npy',sky_spectrum_opt)
    np.save('Misc/SKY_IR_XS.npy',sky_spectrum_ir)

data_file_spec = data_extractor(object_file,JWST=True,plotting=False) 
original_spec = data_file_spec

#if redshift > 0:
#    print('\nRedshifting.')
#    data_file_spec = redshifter(data_file_spec,0.200344,redshift)


low_c = nearest(data_file_spec[0],lambda_low_val,'coord')
high_c = nearest(data_file_spec[0],lambda_high_val,'coord')
data_file_spec = (data_file_spec[0][low_c+1:high_c],data_file_spec[1][low_c+1:high_c])
#data_file_spec[1][data_file_spec[1]<0] = 0

#data_file_spec = np.load('JWST_MOCK_SPECTRA/gaussian_spectrum_for_tests.npy')

SIM_obj_mags = mag_calc(data_file_spec,plotting=False,return_flux=True)
#data_file_spec_photo = (SIM_obj_mags[2],SIM_obj_mags[3])

print('\nSimulating incoming object photons')
photon_spec_no_eff_pre_atm = photons_conversion(data_file_spec,data_file_spec,plotting=False)
photon_spec,atm_trans = atmospheric_effects(photon_spec_no_eff_pre_atm,plotting=False,return_trans=True)
photon_spec_opt,photon_spec_ir,seeing_transmiss_opt,seeing_transmiss_ir = spec_slicer_photometry(photon_spec,return_dimensions=True)

ir_coord = int(len(seeing_transmiss_ir[:,0,0])/2)-1
opt_coord = int(len(seeing_transmiss_opt[:,0,0])/2)-1
seeing_transmiss_ir = seeing_transmiss_ir[:,ir_coord:ir_coord+2,:]
seeing_transmiss_ir = np.sum(seeing_transmiss_ir,axis=0)
seeing_transmiss_opt = seeing_transmiss_opt[:,opt_coord:opt_coord+2,:]
seeing_transmiss_opt = np.sum(seeing_transmiss_opt,axis=0)

photon_spec_ir = photon_spec_ir[:,ir_coord:ir_coord+2,:]
photon_spec_ir = np.sum(photon_spec_ir,axis=0)
photon_spec_opt = photon_spec_opt[:,opt_coord:opt_coord+2,:]
photon_spec_opt = np.sum(photon_spec_opt,axis=0)


print('\nAgain for incoming sky.')
photon_spec_of_sky_pre_atm,sky_spec = sky_addition(photon_spec_no_eff_pre_atm,sky_only=True,plotting=False)
photon_spec_of_sky,_ = atmospheric_effects(photon_spec_of_sky_pre_atm,plotting=False,return_trans=True)
sky_spectrum_opt,sky_spectrum_ir,_,_ = spec_slicer_photometry(photon_spec_of_sky)

sky_spectrum_ir = sky_spectrum_ir[:,ir_coord:ir_coord+2,:]
sky_spectrum_ir = np.sum(sky_spectrum_ir,axis=0)
sky_spectrum_opt = sky_spectrum_opt[:,opt_coord:opt_coord+2,:]
sky_spectrum_opt = np.sum(sky_spectrum_opt,axis=0)

print('\nApplying telescope and optics effects to spectrum.')
gemini_data = np.loadtxt('Mirror_Materials/gemini_mirrors_Ag_coating_data_nm.txt')
surf_transmissions = (np.interp(photon_spec_no_eff_pre_atm[0],gemini_data[:,0][::-1],gemini_data[:,1][::-1])/100)**(7)

nir_QE = 0.85
opt_QE_vals = np.array([0.78,0.78,0.91,0.74,0.23])
opt_QE_wls = np.array([400,550,700,900,1000])
opt_QE = np.interp(photon_spec_no_eff_pre_atm[0],opt_QE_wls,opt_QE_vals)

photon_spec_opt *= (surf_transmissions*opt_QE)
sky_spectrum_opt *= (surf_transmissions*opt_QE)
photon_spec_ir *= (surf_transmissions*nir_QE)
sky_spectrum_ir *= (surf_transmissions*nir_QE)

print('\nApplying read and dark current noise. ')

photon_spec_opt += np.random.poisson((3.4/0.64) + (0.0007222/0.64 * exposure_t))
sky_spectrum_opt += np.random.poisson((3.4/0.64) + (0.0007222/0.64 * exposure_t)) 
photon_spec_ir +=  np.random.poisson((8/2.29) + (0.0125/2.29 * exposure_t)) 
sky_spectrum_ir +=  np.random.poisson((8/2.29) + (0.0125/2.29 * exposure_t)) 

print('\nFinalising incoming photons and beginning post arrival processing.')

incoming_total_photon_spec_opt = np.zeros_like(photon_spec_opt)
incoming_total_photon_spec_ir = np.zeros_like(photon_spec_ir)
incoming_total_photon_spec_opt += (photon_spec_opt+sky_spectrum_opt)
incoming_total_photon_spec_ir += (photon_spec_ir+sky_spectrum_ir)

#######################################################################################################################################

#POST PHOTON ARRIVAL

#######################################################################################################################################

sky_spec_load_opt = np.load('Misc/SKY_OPT_XS.npy')
sky_spec_load_ir = np.load('Misc/SKY_IR_XS.npy')

sky_subbed_spec_opt = np.zeros_like(incoming_total_photon_spec_opt)
sky_subbed_spec_opt = incoming_total_photon_spec_opt - sky_spec_load_opt - ((3.4/0.64) + (0.0007222/0.64 * exposure_t))
sky_subbed_spec_ir = np.zeros_like(incoming_total_photon_spec_ir)
sky_subbed_spec_ir = incoming_total_photon_spec_ir - sky_spec_load_ir - ((8/2.29) + (0.0125/2.29 * exposure_t))
sig_sq_opt = sky_subbed_spec_opt + sky_spec_load_opt + np.square(3.4/0.64) + (0.0007222/0.64 * exposure_t)


sig_sq_opt = np.sqrt(sky_subbed_spec_opt) + np.sqrt(sky_spec_load_opt) + (3.4/0.64) + np.sqrt(0.0007222/0.64 * exposure_t)
sig_sq_opt_sky = np.sqrt(sky_spec_load_opt) + (3.4/0.64) + np.sqrt(0.0007222/0.64 * exposure_t)
sig_sq_opt_tot = np.sqrt((sig_sq_opt) + (sig_sq_opt_sky))
sig_sq_ir = np.sqrt(sky_subbed_spec_ir) + np.sqrt(sky_spec_load_ir) + (8/2.29) + np.sqrt(0.0125/2.29 * exposure_t)
sig_sq_ir_sky = np.sqrt(sky_spec_load_ir) + (8/2.29) + np.sqrt(0.0125/2.29 * exposure_t)
sig_sq_ir_tot = np.sqrt((sig_sq_ir) + (sig_sq_ir_sky))

snrs_opt = np.zeros_like(sky_subbed_spec_opt)
snrs_ir = np.zeros_like(sky_subbed_spec_ir)
snrs_opt += (sky_subbed_spec_opt / np.sqrt( sky_subbed_spec_opt + sky_spec_load_opt + np.square(3.4/0.64) + (0.0007222/0.64 * exposure_t) ) )
snrs_opt = np.sqrt(np.sum(snrs_opt**2,axis=0))
snrs_opt[data_file_spec[0]>1000] = 0
snrs_ir += (sky_subbed_spec_ir / np.sqrt( sky_subbed_spec_ir + sky_spec_load_ir + np.square(8/2.29) + (0.0125/2.29 * exposure_t) ) )
snrs_ir = np.sqrt(np.sum(snrs_ir**2,axis=0))
snrs_ir[data_file_spec[0]<1000] = 0
snrs_full_spec = np.zeros((2,len(data_file_spec[0])))
snrs_full_spec[0] += photon_spec_no_eff_pre_atm[0]
snrs_full_spec[1] += abs(snrs_ir + snrs_opt)
if save_outputs == True:
    np.save('Misc/XSIM_SNRS_5_7.npy',snrs_full_spec)

atm_trans[atm_trans < 0.05] = 0 

corrected_spec_opt = np.zeros_like(sky_subbed_spec_opt)
corrected_spec_opt += ((sky_subbed_spec_opt/((surf_transmissions*opt_QE*atm_trans))))
corrected_spec_ir = np.zeros_like(sky_subbed_spec_ir)
corrected_spec_ir += ((sky_subbed_spec_ir/((surf_transmissions*nir_QE*atm_trans))))

corrected_sig_opt = np.zeros_like(sig_sq_opt_tot)
corrected_sig_opt += ((sig_sq_opt_tot/((surf_transmissions*opt_QE*atm_trans))))
corrected_sig_ir = np.zeros_like(sig_sq_ir_tot)
corrected_sig_ir += ((sig_sq_ir_tot/((surf_transmissions*nir_QE*atm_trans))))

for i in range(len(corrected_spec_opt[0,:])):
    corrected_sig_opt[:,i][corrected_spec_opt[:,i] > 1e308] = 0
    corrected_spec_opt[:,i][corrected_spec_opt[:,i] > 1e308] = 0
    corrected_sig_opt[:,i][corrected_spec_opt[:,i] < -1e308] = 0
    corrected_spec_opt[:,i][corrected_spec_opt[:,i] < -1e308] = 0
    corrected_sig_opt[:,i] = np.nan_to_num(corrected_spec_opt[:,i])
    corrected_spec_opt[:,i] = np.nan_to_num(corrected_spec_opt[:,i])

for i in range(len(corrected_spec_ir[0,:])):
    corrected_sig_ir[:,i][corrected_spec_ir[:,i] > 1e308] = 0
    corrected_spec_ir[:,i][corrected_spec_ir[:,i] > 1e308] = 0
    corrected_sig_ir[:,i][corrected_spec_ir[:,i] < -1e308] = 0
    corrected_spec_ir[:,i][corrected_spec_ir[:,i] < -1e308] = 0
    corrected_sig_ir[:,i] = np.nan_to_num(corrected_spec_ir[:,i])
    corrected_spec_ir[:,i] = np.nan_to_num(corrected_spec_ir[:,i])


full_spec = np.zeros_like(photon_spec_no_eff_pre_atm)
full_spec[0] += photon_spec_no_eff_pre_atm[0]
full_spec[1] += (np.sum(corrected_spec_ir,axis=0)+np.sum(corrected_spec_opt,axis=0))

sig_sq_full_spec = np.zeros((2,len(full_spec[0])))
sig_sq_full_spec[0] += full_spec[0]
sig_sq_full_spec[1] += (np.sum(np.sum(corrected_sig_ir,axis=1),axis=0)+np.sum(np.sum(corrected_sig_opt,axis=1),axis=0))

flux_full_spec = flux_conversion_3(full_spec)
sig_sq_full_flux = flux_conversion_3(sig_sq_full_spec)

SIM_out_obj_mags = mag_calc(flux_full_spec,plotting=False,return_flux=True)
SIM_out_obj_band = mag_calc_bandpass(flux_full_spec,plotting=False,return_flux=True)

print('\nPlotting results.')

plt.figure()
plt.plot(SIM_out_obj_mags[2],SIM_out_obj_mags[3],'b-',label='Simulated fluxes')
plt.plot(SIM_out_obj_mags[2],SIM_out_obj_mags[3],'bx')
plt.plot(SIM_obj_mags[2],SIM_obj_mags[3],'r-',alpha=0.6,label='Data fluxes')
plt.plot(SIM_obj_mags[2],SIM_obj_mags[3],'rx')
plt.xlabel('Wavelength / nm')
plt.ylabel(object_y)
plt.legend(loc='best')

plt.figure()
plt.plot(full_spec[0],full_spec[1],'b-',label='Simulated photon counts')
plt.plot(photon_spec_no_eff_pre_atm[0],photon_spec_no_eff_pre_atm[1],'r-',alpha=0.6,label='Data photon counts')
plt.xlabel('Wavelength / nm')
plt.ylabel('Photon count')
plt.legend(loc='best')

plt.figure()
plt.plot(flux_full_spec[0],flux_full_spec[1],'b-',label='Simulated fluxes')
plt.plot(data_file_spec[0],data_file_spec[1],'r-',alpha=0.6,label='Data fluxes')
plt.xlabel('Wavelength / nm')
plt.ylabel(object_y)
plt.legend(loc='best')

if save_outputs == True:
    np.save('Misc/JWST_MOCK_SPEC_5_7_XSIM.npy',flux_full_spec)
    #np.save('Misc/JWST_MOCK_SPEC_5_7_DATA.npy',data_file_spec)
    np.save('Misc/XSIM_SIGSQ_5_7.npy',sig_sq_full_flux)


