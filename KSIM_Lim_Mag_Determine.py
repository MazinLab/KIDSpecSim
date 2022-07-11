# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:33:38 2021

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
            data_extractor_TLUSTY, data_extractor_TLUSTY_joint_spec,spec_seeing,SNR_calc_pred_grid, \
                atmospheric_effects,rebinner_with_bins, \
                    R_value,redshifter,order_merge_reg_grid,grid_plotter,SNR_calc_grid,grid_plotter_opp, \
                        model_interpolator,model_interpolator_sky
from parameters import *
from flux_to_photons_conversion import photons_conversion,flux_conversion,flux_conversion_3
from apply_QE import QE
from grating import * #grating_orders_2_arms,grating_binning,grating_binning_high_enough_R,grating_binning_high_enough_R_sky
from MKID_gaussians import MKID_response_Express,recreator

plt.rcParams.update({'font.size': 28}) #sets the fontsize of any plots
time_start = datetime.datetime.now() #beginning timer

extra_plots = False

print('\n Simulating the grating orders and calculating efficiencies.')
orders,order_wavelengths,grating_efficiency,orders_opt,order_wavelengths_opt, \
    efficiencies_opt,orders_ir,order_wavelengths_ir,efficiencies_ir, \
         = grating_orders_2_arms('Casini',cutoff,plotting=extra_plots)
        
detec_spec = wavelength_array_maker(order_wavelengths) #forms a 1D array based on the wavelengths each order sees
    
print('Exposure time = %i /s'%exposure_t)

def KSIM_looper(mag_reduce_fac,blaze_coords):
    #####################################################################################################################################################
    #SPECTRUM DATA FILE IMPORT AND CONVERTING TO PHOTONS
    ######################################################################################################################################################
    
    print('\n Importing spectrum from data file.')
    if TLUSTY == True:
        #model_spec = data_extractor_TLUSTY_joint_spec(object_file_1,object_file_2,row,plotting=extra_plots)  
        #model_spec = model_spec[:,:150000]
        tlusty_spec = np.loadtxt('SHIFTED_SPECTRA/XS_T100T498_OBJ.txt')
        model_spec = np.zeros((2,len(tlusty_spec[:,0])))
        model_spec[0] += tlusty_spec[:,0]
        model_spec[1] += tlusty_spec[:,1]
    elif supported_file_extraction == True:
        #If data that will be used is from the DRs of XShooter then set XShooter to True, since their FITS files are setup differently to the ESO-XShooter archive
        #Spectrum is in units of Flux / $ergcm^{-2}s^{-1}\AA^{-1}$, the AA is angstrom, wavelength array is in nm
        model_spec = data_extractor(object_file,quasar=True,plotting=extra_plots)
    else:
        model_spec_initial = np.loadtxt('%s/%s'%(folder,object_file)) #previous text file containing two columns, wavelength in nm and flux in ergcm^{-2}s^{-1}\AA^{-1} 
        
        if np.shape(model_spec_initial) != (len(model_spec_initial[:,0]),2):
            raise Exception('Format for input file should be 2 columns, one for wavelength in nm and one for flux in ergcm^{-2}s^{-1}\AA^{-1}')
                                        
        
        model_spec = np.zeros((2,len(model_spec_initial)))
        model_spec[0] += model_spec_initial[:,0]
        model_spec[1] += model_spec_initial[:,1]
    
    
    original_spec = np.copy(model_spec)
    
    #if redshift > 0:
    #    print('\n Redshifting.')
    #    model_spec = redshifter(model_spec,0.200344,redshift)
    
    print('\n Simulating observation of %s.'%object_name)
    
    #sectioning the spectrum to chosen KIDSpec bandpass
    model_spec1 = (original_spec[0],original_spec[1] / mag_reduce_fac) 
    model_spec = np.zeros((2,len(model_spec[0])))
    model_spec[0] += model_spec1[0]
    model_spec[1] += np.max(model_spec1[1])
    
    low_c = nearest(model_spec[0],lambda_low_val,'coord')               
    high_c = nearest(model_spec[0],lambda_high_val,'coord')
    model_spec = (model_spec[0][low_c+1:high_c],model_spec[1][low_c+1:high_c])
    
    #converting data spectrum to photons and loading incoming sky spectrum from ESO sky model
    photon_spec_no_eff_original = photons_conversion(model_spec,model_spec,plotting=extra_plots)
    
    #increasing number of points in model spectrum
    photon_spec_no_eff = model_interpolator(photon_spec_no_eff_original,1000000)
    
    #generating sky
    photon_spec_of_sky_orig = sky_spectrum_load(plotting=extra_plots)
    photon_spec_of_sky = model_interpolator_sky(photon_spec_of_sky_orig,1000000)
     
    #calculating magnitudes of model spectrum
    SIM_obj_mags = mag_calc(model_spec,plotting=False,wls_check=True)
    
    
    ##############################################################################################################################################################################
    #ATMOSPHERE TRANSMISSION, TELESCOPE TRANSMISSION, SLIT EFFECTS, QE
    ##############################################################################################################################################################################
    
    print('\n Applying atmospheric, telescope, slit, and QE effects.')
    
    photon_spec_post_atmos = atmospheric_effects(photon_spec_no_eff,plotting=extra_plots,return_trans=False)
    photon_sky_post_atmos = atmospheric_effects(photon_spec_of_sky,plotting=False,return_trans=False)
    
    photon_spec_pre_optics = telescope_effects(photon_spec_post_atmos,plotting=extra_plots)
    photon_sky_pre_optics = telescope_effects(photon_sky_post_atmos,plotting=False)
    
    photon_spec_to_instr = optics_transmission(photon_spec_pre_optics,20)
    photon_sky_to_instr = optics_transmission(photon_sky_pre_optics,20)
    
    if gen_model_seeing_eff == True:
        photon_spec_post_slit,seeing_transmiss_model = spec_seeing(photon_spec_to_instr,plotting=extra_plots)
        np.save('Misc/%s.npy'%model_seeing_eff_file_save_or_load,seeing_transmiss_model)
    else:
        photon_spec_post_slit = np.copy(photon_spec_to_instr)
        seeing_transmiss_model = np.load('Misc/%s.npy'%model_seeing_eff_file_save_or_load)
        photon_spec_post_slit[1] *= seeing_transmiss_model[1]
        
    print('\nModel spectrum complete')
    
    if gen_sky_seeing_eff == True:
        photon_sky_post_slit,seeing_transmiss_sky = spec_seeing(photon_sky_to_instr)
        np.save('Misc/%s.npy'%sky_seeing_eff_file_save_or_load,seeing_transmiss_sky)
    else:
        photon_sky_post_slit = np.copy(photon_sky_to_instr)
        seeing_transmiss_sky = np.load('Misc/%s.npy'%sky_seeing_eff_file_save_or_load)
        photon_sky_post_slit[1] *= seeing_transmiss_sky[1]
    print('\nSky spectrum complete.')
    
    spec_QE = QE(photon_spec_post_slit,constant=False,plotting=False)
    sky_QE = QE(photon_sky_post_slit,constant=False,plotting=False)
    
    
    ############################################################################################################################################################################################
    #GRATING
    ##########################################################################################################################################################################################
    
    
    print('\n Simulating the grating orders and calculating efficiencies.')
    orders,order_wavelengths,grating_efficiency,orders_opt,order_wavelengths_opt, \
        efficiencies_opt,orders_ir,order_wavelengths_ir,efficiencies_ir, \
             = grating_orders_2_arms('Casini',cutoff,plotting=extra_plots)
            
    detec_spec = wavelength_array_maker(order_wavelengths) #forms a 1D array based on the wavelengths each order sees
    
    
    
    #############################################################################################################################################################################################
    #BINNING PHOTONS ONTO MKIDS AND THEIR ORDER WAVELENGTHS FOR >>OPTICAL<< ARM
    #################################################################################################################################################################################################
    
    
    if orders_opt[0] != 1:
        print('\n Binning photons for OPT arm (incoming object photons).')
        pixel_sums_opt,order_wavelength_bins_opt = grating_binning_high_enough_R_lim_mag(spec_QE,order_wavelengths_opt,order_wavelengths,
                                                                                                                          orders_opt,efficiencies_opt,cutoff,IR=False,OPT=True,plotting=extra_plots)
    
        print('\n OPT arm sky photons.')
        pixel_sums_opt_sky,_= grating_binning_high_enough_R_lim_mag(sky_QE,order_wavelengths_opt,order_wavelengths,
                                                      orders_opt,efficiencies_opt,cutoff,IR=False,OPT=True,plotting=extra_plots)
        
        #adding the object and sky grids together
        pixel_sums_opt_no_sky = np.zeros_like(pixel_sums_opt)
        pixel_sums_opt_no_sky += np.copy(pixel_sums_opt)
        pixel_sums_opt += np.copy(pixel_sums_opt_sky)
        
    
    
    
    
    #############################################################################################################################################################################################
    #BINNING PHOTONS ONTO MKIDS AND THEIR ORDER WAVELENGTHS FOR >>NIR<< ARM
    #################################################################################################################################################################################################
    
    if len(orders_ir) == 0:
        orders_ir = np.append(orders_ir,200)
    if orders_ir[0] != 1:
        #FIRST DOING IR / LOWER ORDERS 
        #bins the photons onto relevant MKIDs and orders
        print('\n  Binning photons for NIR arm (incoming object photons).')
        pixel_sums_ir,order_wavelength_bins_ir = grating_binning_high_enough_R_lim_mag(spec_QE,order_wavelengths_ir,order_wavelengths,
                                                                                                                    orders_ir,efficiencies_ir,cutoff,IR=True,OPT=False,
                                                                                                                        plotting=extra_plots)
        print('\n NIR arm sky photons.')
        pixel_sums_ir_sky,_ = grating_binning_high_enough_R_lim_mag(sky_QE,order_wavelengths_ir,
                                                                    order_wavelengths,orders_ir,efficiencies_ir,
                                                                    cutoff,IR=True,OPT=False,plotting=extra_plots)
        
        #adding the object and sky grids together
        pixel_sums_ir_no_sky = np.zeros_like(pixel_sums_ir)
        pixel_sums_ir_no_sky += pixel_sums_ir
        pixel_sums_ir += pixel_sums_ir_sky
        
    
    
    
    #############################################################################################################################################################################################
    #SIMULATING MKID RESPONSES USING GAUSSIAN METHOD
    #################################################################################################################################################################################################
    
    print('\n Beginning MKID response simulation for each arm and simultaneous sky exposure.')
    
    #OPT object mkid response
    kidspec_resp_opt,kidspec_mis_opt = MKID_response_Express(orders_opt,order_wavelengths,order_wavelengths_opt,n_pixels,pixel_sums_opt,
                      IR=False,sky=False,dual_arm_=True,make_folder=True)
    print('\nOPT object observation complete. 1/4')
    
    #OPT sky mkid response
    kidspec_sky_resp_opt,kidspec_sky_mis_opt = MKID_response_Express(orders_opt,order_wavelengths,order_wavelengths_opt,n_pixels,pixel_sums_opt_sky,
                      IR=False,sky=True,dual_arm_=True,make_folder=False)
    print('\nOPT sky observation complete. 2/4')
    
    
    #NIR object mkid response
    kidspec_resp_ir,kidspec_mis_ir = MKID_response_Express(orders_ir,order_wavelengths,order_wavelengths_ir,n_pixels,pixel_sums_ir,
                      IR=True,sky=False,dual_arm_=True,make_folder=False)
    print('\nNIR object observation complete. 3/4')
    
    #NIR sky mkid response
    kidspec_sky_resp_ir,kidspec_sky_mis_ir = MKID_response_Express(orders_ir,order_wavelengths,order_wavelengths_ir,n_pixels,pixel_sums_ir_sky,
                      IR=True,sky=True,dual_arm_=True,make_folder=False)
    print('\nNIR sky observation complete. 4/4')
    
    
    print('\nFinalising MKID response grids.')
    if IR_arm == True:
        kidspec_raw_output = np.zeros_like(order_wavelengths)
        kidspec_raw_output[:len(order_list_ir)] += kidspec_resp_ir
        kidspec_raw_output[len(order_list_ir):] += kidspec_resp_opt
        
        misidentified_spectrum = np.zeros_like(order_wavelengths)
        misidentified_spectrum[:len(order_list_ir)] += kidspec_mis_ir
        misidentified_spectrum[len(order_list_ir):] += kidspec_mis_opt
        
        kidspec_raw_output_sky = np.zeros_like(order_wavelengths)
        kidspec_raw_output_sky[:len(order_list_ir)] += kidspec_sky_resp_ir
        kidspec_raw_output_sky[len(order_list_ir):] += kidspec_sky_resp_opt
        
        misidentified_sky_spectrum = np.zeros_like(order_wavelengths)
        misidentified_sky_spectrum[:len(order_list_ir)] += kidspec_sky_mis_ir
        misidentified_sky_spectrum[len(order_list_ir):] += kidspec_sky_mis_opt
    else:
        kidspec_raw_output = np.zeros_like(order_wavelengths)
        misidentified_spectrum = np.zeros_like(order_wavelengths)
        kidspec_raw_output_sky = np.zeros_like(order_wavelengths)
        misidentified_sky_spectrum = np.zeros_like(order_wavelengths)
        
        kidspec_raw_output += kidspec_resp_opt
        misidentified_spectrum += kidspec_mis_opt
        kidspec_raw_output_sky += kidspec_sky_resp_opt
        misidentified_sky_spectrum += kidspec_sky_mis_opt
        
        
        
        
    #kidspec_raw_output,misidentified_spectrum,percentage_misidentified_pp, \
    #    percentage_misidentified_tot,no_misident = recreator(spec_QE,n_pixels,order_wavelengths,orders_ir,sky=False) #1D spectrum of pixel grid, ORDERS NOT MERGED
    
    #kidspec_raw_output_sky,misidentified_sky_spectrum,percentage_sky_misidentified_pp, \
    #    percentage_sky_misidentified_tot,no_misident_sky = recreator(spec_QE,n_pixels,order_wavelengths,orders_ir,sky=True)
    
    
    if delete_folders == True:
        shutil.rmtree('%s/Resample'%folder)
    
    
    #############################################################################################################################################################################################
    #SKY SUBTRACTION, ORDER MERGING AND SNR CALCULATION
    #################################################################################################################################################################################################
    
    print('\nSubtracting sky.')
    #subtracting sky
    raw_sky_subbed_spec_pre_ord_merge = np.zeros_like(kidspec_raw_output)
    raw_sky_subbed_spec_pre_ord_merge += (kidspec_raw_output - kidspec_raw_output_sky)
    
    #if extra_plots == True:
    #    grid_plotter(order_wavelengths,raw_sky_subbed_spec_pre_ord_merge)
    
    #SNR calculation
    SNR_total,SNRs = SNR_calc_grid(raw_sky_subbed_spec_pre_ord_merge,kidspec_raw_output_sky,plotting=False)
    #predicted_SNRs_x = SNR_calc_pred_grid(raw_sky_subbed_spec_pre_ord_merge,kidspec_raw_output_sky,SOX=False,plotting=False) #X-Shooter
    #predicted_SNRs_s = SNR_calc_pred_grid(raw_sky_subbed_spec_pre_ord_merge,kidspec_raw_output_sky,SOX=True,plotting=False) #SOXS
    
    SNRs_blaze = np.zeros((len(order_wavelengths[:,0]),2))
    SNRs_blaze[:,0] += order_wavelengths[:,blaze_coords]
    SNRs_blaze[:,1] += SNRs[:,blaze_coords]
    
    
    
    return SNRs_blaze,SIM_obj_mags,SNRs,pixel_sums_opt

#############################################################################################################################################################################################
#BEGINNING LOOP FOR LIMITING MAGNITUDE, WHERE SNR > 10
#################################################################################################################################################################################################


blaze_coord = int(np.round(n_pixels/2,decimals=0))

mag_central_wls = np.array([  358.5       ,   458.        ,   598.5       ,   820.        ,
         943.        ,   999.5       ,  1020.        ,  1250.12503051,
         1650.13000489,  2150.2199707 ,  3800.70495605,  4700.94506836,
         10500.        , 20000.        ])



blaze_wls_mag = np.zeros((len(order_wavelengths[:,0]),3))
blaze_wls_mag[:,0] += order_wavelengths[:,blaze_coord]

mag_coord_match_for_wls = np.zeros(len(order_wavelengths[:,0]))
for i in range(len(order_wavelengths[:,0])):
    mag_coord_match_for_wls[i] += int(nearest(mag_central_wls,order_wavelengths[i,blaze_coord],'coord'))


stop_condition = 0
blaze_wl_mag_found_check = np.zeros(len(order_wavelengths[:,0]))

mag_reduce_fac = mag_reduce #starting magnitude reduction

while stop_condition != len(blaze_wl_mag_found_check):
    
    print('\nCurrent magnitude reduction factor: %.3f'%mag_reduce_fac)
    print('\nCurrent wavelengths found: %i / %i'%(stop_condition,len(blaze_wl_mag_found_check)))
    print(blaze_wls_mag)
    print('\nCurrent SNRs found:')
    try:
        print(SNRs_blaze)
    except:
        print('\nWill begin printing SNRs found after initial run.')
    
    SNRs_blaze,current_mag,SNRs,pixel_sums_opt = KSIM_looper(mag_reduce_fac,blaze_coord)

    for i in range(len(SNRs_blaze[:,0])):
        if SNRs_blaze[i,1] < 10 and blaze_wl_mag_found_check[i] == 0:
            blaze_wls_mag[i,1] += current_mag[0][int(mag_coord_match_for_wls[i])]
            blaze_wls_mag[i,2] += SNRs_blaze[i,1]
            blaze_wl_mag_found_check[i] += 1
            stop_condition += 1
    
    mag_reduce_fac_prev = np.copy(mag_reduce_fac)
    if mag_reduce_fac >= 100:
        mag_reduce_fac += 20
    if mag_reduce_fac >= 500:
        mag_reduce_fac += 30
    if mag_reduce_fac >= 1000:
        mag_reduce_fac += 200
    if mag_reduce_fac >= 5000:
        mag_reduce_fac += 250
    if mag_reduce_fac >= 15000:
        mag_reduce_fac += 50
    if mag_reduce_fac >= 20000:
        mag_reduce_fac += 2000
    if mag_reduce_fac_prev == mag_reduce_fac:
        mag_reduce_fac += 10
            
    print('\nCurrent time taken:',datetime.datetime.now() - time_start)


time_took = datetime.datetime.now() - time_start

print('\n Simulation took', time_took,'(hours:minutes:seconds)')



#############################################################################################################################################################################################
#GENERATING OUTPUT METRICS AND FILE
#################################################################################################################################################################################################


time_str = ('DATE_'+
            str(datetime.datetime.now().year)+
            '-'+
            str(datetime.datetime.now().month)+
            '-'+
            str(datetime.datetime.now().day)+
            '_TIME_'+
            str(datetime.datetime.now().hour)+
            '-'+
            str(datetime.datetime.now().minute)+
            '-'+
            str(datetime.datetime.now().second)
            )

f = open('%s/%s_output_metrics_%s.txt'%(folder,object_name,time_str),'w+')

f.write('> Output metrics for KIDSpec Simulation of object %s\n'%(object_name))
f.write('\nDate run: %s \n\n'%datetime.datetime.now())
f.write('> Input Parameters for simulation and instrument: \n\n')
f.write('Telescope mirror diameter: %i cm \n'%mirr_diam)
f.write('Exposure time: %i s \n'%exposure_t)
f.write('Seeing: %.1f arcseconds \n'%seeing)
f.write('Airmass: %.1f \n\n'%airmass)

f.write('Slit width: %.2f arcseconds \n'%slit_width)
f.write('Slit length: %.2f arcseconds \n'%slit_length)
f.write('Pixel plate scale: %.2f arcseconds \n'%pix_fov)
f.write('Slicers: %i \n\n'%slicers)

if IR_arm == True:
    f.write('OPT arm incidence angle: %.1f deg \n'%alpha_val)
    f.write('OPT arm blaze angle: %.1f deg \n'%phi_val)
    f.write('OPT arm grooves: %.1f /mm \n'%OPT_grooves)
    
    f.write('IR arm incidence angle: %.1f deg \n'%IR_alpha)
    f.write('IR arm blaze angle: %.1f deg \n'%IR_phi)
    f.write('IR arm grooves: %.1f /mm \n\n'%IR_grooves)
    
else:
    f.write('KIDSpec grating incidence angle: %.1f deg \n'%alpha_val)
    f.write('KIDSpec grating blaze angle: %.1f deg \n'%phi_val)
    f.write('KIDSpec grating grooves: %.1f /mm \n\n'%OPT_grooves)

f.write('Number of spectral pixels in each arm: %i \n'%n_pixels)
f.write('Pixel plate scale: %.1f \n'%pix_fov)
f.write('Chosen MKID energy resolution at fiducial point: %i \n'%(ER_band_low))

#if IR_arm == True:
    #f.write('Number of dead spectal pixels in OPT arm: %i \n'%dead_pixel_opt_amount)
    #f.write('Number of dead spectal pixels in IR arm: %i \n'%dead_pixel_ir_amount)
    #f.write('MKID energy resolution at fiducial point standard deviation for OPT arm: %.2f \n'%(R_E_OPT_std))
    #f.write('MKID energy resolution at fiducial point standard deviation for IR arm: %.2f \n\n'%(R_E_IR_std))
#else:
    #f.write('Number of dead spectal pixels in KIDSpec: %i \n'%dead_pixel_opt_amount)
    #f.write('MKID energy resolution at fiducial point standard deviation for KIDSpec: %.2f \n\n'%(R_E_OPT_std))

f.write('Flux reduced in total by factor of %.1f\n'%mag_reduce)
f.write('Spectrum redshifted by %.3f\n'%redshift)

f.write('\n')

f.write('> Result parameters: \n\n')
f.write('OPT orders observed: %i \n'%len(orders_opt))
f.write('IR orders observed: %i \n'%len(orders_ir))
f.write('Wavelength range tested: %i - %i nm \n \n'%(lambda_low_val,lambda_high_val))

f.write('Simulation limiting magnitudes [Wavelength/nm, Vega band, Vega magnitude, SNR at limit]:\n')
for i in range(int(len(blaze_wls_mag[:,1]))):
    f.write('%.5f | %s | %.5f | %.5f \n'%(blaze_wls_mag[i,0],current_mag[1][int(mag_coord_match_for_wls[i])],
                                    blaze_wls_mag[i,1],blaze_wls_mag[i,2]))

f.write('Array coordinates for magnitudes:\n')
for i in range(len(mag_coord_match_for_wls)):
    f.write('%i '%mag_coord_match_for_wls[i])
f.write('\n')


f.write('Simulation run duration: %s (hours:minutes:seconds)\n'%time_took)
f.close()









