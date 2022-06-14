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
                        model_interpolator,KIDSpec_Express
from parameters import *
from flux_to_photons_conversion import photons_conversion,flux_conversion,flux_conversion_3
from apply_QE import QE
from grating import grating_orders_2_arms,grating_binning,grating_binning_high_enough_R,grating_binning_high_enough_R_sky
from MKID_gaussians import MKID_response,recreator

plt.rcParams.update({'font.size': 28}) #sets the fontsize of any plots
time_start = datetime.datetime.now() #beginning timer




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
model_spec = (model_spec[0],model_spec[1] / mag_reduce) 

low_c = nearest(model_spec[0],lambda_low_val,'coord')               
high_c = nearest(model_spec[0],lambda_high_val,'coord')
model_spec = (model_spec[0][low_c+1:high_c],model_spec[1][low_c+1:high_c])

#converting data spectrum to photons and loading incoming sky spectrum from ESO sky model
photon_spec_no_eff_original = photons_conversion(model_spec,model_spec,plotting=extra_plots)

#increasing number of points in model spectrum
photon_spec_no_eff = model_interpolator(photon_spec_no_eff_original,100000)

#generating sky
photon_spec_of_sky = sky_spectrum_load(plotting=extra_plots)
 
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

photon_spec_to_instr = optics_transmission(photon_spec_pre_optics,6)
photon_sky_to_instr = optics_transmission(photon_sky_pre_optics,6)

if gen_model_seeing_eff == True:
    photon_spec_post_slit,seeing_transmiss_model = spec_seeing(photon_spec_to_instr,plotting=extra_plots)
    np.save('Misc/%s.npy'%model_seeing_eff_file_save_or_load,seeing_transmiss_model)
else:
    photon_spec_post_slit = np.copy(photon_spec_to_instr)
    seeing_transmiss_model = np.load('Misc/%s.npy'%model_seeing_eff_file_save_or_load)
    photon_spec_post_slit[1] *= seeing_transmiss_model[1]
    
    plt.figure()
    plt.plot(photon_spec_to_instr[0],photon_spec_to_instr[1],'r-',label='Pre slit')
    plt.plot(photon_spec_post_slit[0],photon_spec_post_slit[1],'b-',alpha=0.7,label='Post slit')
    plt.xlabel('Wavelength / nm')
    plt.ylabel('Photon count')
    plt.legend(loc='best')
    
    plt.figure()
    plt.plot(seeing_transmiss_model[0],seeing_transmiss_model[1],label='Slit transmission')
    plt.xlabel('Wavelength / nm')
    plt.ylabel('Transmission') 
    plt.legend(loc='best')
    
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
    pixel_sums_opt,opt_zeroed_wls,order_wavelength_bins_opt = grating_binning_high_enough_R(spec_QE,order_wavelengths_opt,order_wavelengths,
                                                                                                                      orders_opt,efficiencies_opt,cutoff,IR=False,OPT=True,plotting=extra_plots)

    print('\n OPT arm sky photons.')
    pixel_sums_opt_sky,_,_= grating_binning_high_enough_R_sky(sky_QE,order_wavelengths_opt,order_wavelengths,
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
    pixel_sums_ir,ir_zeroed_wls,order_wavelength_bins_ir = grating_binning_high_enough_R(spec_QE,order_wavelengths_ir,order_wavelengths,
                                                                                                                orders_ir,efficiencies_ir,cutoff,IR=True,OPT=False,
                                                                                                                    plotting=extra_plots)
    print('\n NIR arm sky photons.')
    pixel_sums_ir_sky,_,_ = grating_binning_high_enough_R_sky(sky_QE,order_wavelengths_ir,
                                                                order_wavelengths,orders_ir,efficiencies_ir,
                                                                cutoff,IR=True,OPT=False,plotting=extra_plots)
    
    #adding the object and sky grids together
    pixel_sums_ir_no_sky = np.zeros_like(pixel_sums_ir)
    pixel_sums_ir_no_sky += pixel_sums_ir
    pixel_sums_ir += pixel_sums_ir_sky


#############################################################################################################################################################################################
#SIMULATING MKID RESPONSES USING GAUSSIAN METHOD
#################################################################################################################################################################################################


KIDSpec_Express(pixel_sums_ir,pixel_sums_ir_sky,order_wavelengths_ir,
                n_pixels,bad_pix=[False,1.0],R_E_shift=[False,0],IR=True)

KIDSpec_Express(pixel_sums_opt,pixel_sums_opt_sky,order_wavelengths_opt,
                n_pixels,bad_pix=[False,1.0],R_E_shift=[False,0],IR=False)


kidspec_raw_output,misidentified_spectrum,percentage_misidentified_pp, \
    percentage_misidentified_tot,no_misident = recreator(spec_QE,n_pixels,order_wavelengths,orders_ir,sky=False) #1D spectrum of pixel grid, ORDERS NOT MERGED

kidspec_raw_output_sky,misidentified_sky_spectrum,percentage_sky_misidentified_pp, \
    percentage_sky_misidentified_tot,no_misident_sky = recreator(spec_QE,n_pixels,order_wavelengths,orders_ir,sky=True)


if delete_folders == True:
    shutil.rmtree('%s/Resample'%folder)


#############################################################################################################################################################################################
#SKY SUBTRACTION, ORDER MERGING AND SNR CALCULATION
#################################################################################################################################################################################################

print('\nSubtracting sky.')
#subtracting sky
raw_sky_subbed_spec_pre_ord_merge = np.zeros_like(kidspec_raw_output)
raw_sky_subbed_spec_pre_ord_merge += (kidspec_raw_output - kidspec_raw_output_sky)

if extra_plots == True:
    grid_plotter(order_wavelengths,raw_sky_subbed_spec_pre_ord_merge)


#SNR calculation
SNR_total,SNRs = SNR_calc_grid(raw_sky_subbed_spec_pre_ord_merge,kidspec_raw_output_sky,plotting=True)
predicted_SNRs_x = SNR_calc_pred_grid(raw_sky_subbed_spec_pre_ord_merge,kidspec_raw_output_sky,SOX=False,plotting=False) #X-Shooter
predicted_SNRs_s = SNR_calc_pred_grid(raw_sky_subbed_spec_pre_ord_merge,kidspec_raw_output_sky,SOX=True,plotting=False) #SOXS

#calculating spectral resolution of KIDSpec setup
Rs = []
for i in range(len(order_wavelengths)):
    R1 = order_wavelengths[i][0] / (order_wavelengths[i][1]-order_wavelengths[i][0])
    Rs.append(R1)
    R2 = order_wavelengths[i][-1] / (order_wavelengths[i][-1]-order_wavelengths[i][-2])
    Rs.append(R2)
Rs = np.asarray(Rs)/slit_R_factor
R_high = np.max(Rs)
R_low = np.min(Rs)
spec_R_pre_cutoff = np.arange(lambda_low_val,cutoff,lambda_low_val/R_high)
spec_R_post_cutoff = np.arange(cutoff,lambda_high_val,lambda_high_val/R_low)
spec_R = np.append(spec_R_pre_cutoff,spec_R_post_cutoff)

print('\nMerging orders.')
#merging orders here onto grid with many bins
raw_sky_subbed_spec = order_merge_reg_grid(order_wavelengths,raw_sky_subbed_spec_pre_ord_merge) 

#rebinning to KIDSpec's current spectral resolution
#raw_sky_subbed_spec = rebinner_with_bins(raw_sky_subbed_spec_pre_bin,spec_R)

#############################################################################################################################################################################################
#STANDARD STAR FACTORS GENERATION OR APPLICATION
#################################################################################################################################################################################################

SIM_total_flux_spectrum = flux_conversion_3(raw_sky_subbed_spec)

if stand_star_factors_run == False: #loading standard star factors
    print('\nLoading standard star weights and applying them.')
    stand_star_spec = np.load('STANDARD_STAR_FACTORS_STORE/DATA_SPECTRUM_GD71_%spix%s.npy'%(n_pixels,stand_star_filename_detail))
    factors = np.zeros_like(stand_star_spec)
    factors[0] += stand_star_spec[0]
    factors[1] += np.load('STANDARD_STAR_FACTORS_STORE/FACTORS_STAND_STAR_GD71_%spix%s.npy'%(n_pixels,stand_star_filename_detail))[1]
    
    SIM_flux_pre_weights = np.copy(SIM_total_flux_spectrum)
    
    #applying standard star weights and flux conversion
    SIM_total_flux_spectrum[1] /= factors[1]
    
    corrected_KS_spec = np.copy(raw_sky_subbed_spec)
    corrected_KS_spec[1] /= factors[1]
    
elif stand_star_factors_run == True: #generating standard star factors
    print('\nGenerating standard star weights.')
    model_func = np.zeros_like(SIM_total_flux_spectrum)
    model_func[0] += SIM_total_flux_spectrum[0]
    model_func[1] += np.interp(SIM_total_flux_spectrum[0],model_spec[0],model_spec[1])
    
    factors = np.zeros_like(SIM_total_flux_spectrum)
    factors[0] += SIM_total_flux_spectrum[0]
    factors[1] += SIM_total_flux_spectrum[1] / model_func[1]
    
    np.save('STANDARD_STAR_FACTORS_STORE/FACTORS_STAND_STAR_GD71_%spix%s.npy'%(n_pixels,stand_star_filename_detail),factors)
    np.save('STANDARD_STAR_FACTORS_STORE/DATA_SPECTRUM_GD71_%spix%s.npy'%(n_pixels,stand_star_filename_detail),model_func)
    
    SIM_flux_pre_weights = np.copy(SIM_total_flux_spectrum)

    SIM_total_flux_spectrum[1] /= factors[1]
    
    corrected_KS_spec = np.copy(raw_sky_subbed_spec)
    corrected_KS_spec[1] /= factors[1]

else:
    raise Exception('STANDARD STAR FACTOR RUN OPTION NOT SELECTED: TRUE OR FALSE')


#plotting the effects of the standard star factors

#plotting the effects of the standard star factors
plt.figure()
plt.plot(SIM_flux_pre_weights[0],SIM_flux_pre_weights[1],'b-',label='Raw KSIM output')
plt.plot(model_spec[0],model_spec[1],'r-',label='Model spectrum')
plt.plot(SIM_total_flux_spectrum[0],SIM_total_flux_spectrum[1],'g-',alpha=0.6,label='Standard star weights applied')
plt.legend(loc='best')
plt.xlabel('Wavelength / nm')
plt.ylabel(object_y)


#############################################################################################################################################################################################
#MISCELLEANOUS ANALYSIS
#################################################################################################################################################################################################

coord_low = nearest(corrected_KS_spec[0],model_spec[0][0],'coord')
coord_high = nearest(corrected_KS_spec[0],model_spec[0][-1],'coord')

corrected_KS_spec = corrected_KS_spec[:,coord_low+1:coord_high]
SIM_total_flux_spectrum = SIM_total_flux_spectrum[:,coord_low+1:coord_high]

#rebinning back to model spectrum 
try:
    SIM_rebin_to_data = rebinner_with_bins(corrected_KS_spec,model_spec[0])

except:
    SIM_rebin_to_data = rebinner_with_bins(corrected_KS_spec[:,1:-1],model_spec[0])

#flux conversion
SIM_total_flux_spectrum_model_bins = flux_conversion_3(SIM_rebin_to_data)

#magnitude calculation from simulation result
SIM_out_mags = mag_calc(SIM_total_flux_spectrum,plotting=False,wls_check=False)

#R value statistic between model and simulation output
R_value_stat = R_value(SIM_total_flux_spectrum_model_bins,model_spec,plotting=True)


#############################################################################################################################################################################################
#PLOTTING
#################################################################################################################################################################################################

fig2 = plt.figure()
plt.plot(model_spec[0],model_spec[1],'r-',label='Model spectrum')
plt.plot(SIM_total_flux_spectrum[0],SIM_total_flux_spectrum[1],'b-',label='Spectrum from simulation',alpha = 0.6)
plt.xlabel(object_x)
plt.ylabel(object_y)
plt.legend(loc='best')
fig2.text(0.73,0.70,'%s '%object_name)


fig3 = plt.figure()
plt.plot(model_spec[0],model_spec[1],'r-',label='Model spectrum')
plt.plot(SIM_total_flux_spectrum_model_bins[0],SIM_total_flux_spectrum_model_bins[1],'b-',label='Spectrum from simulation rebinned',alpha = 0.6)
plt.xlabel(object_x)
plt.ylabel(object_y)
plt.legend(loc='best')
fig3.text(0.73,0.70,'%s '%object_name)

time_took = datetime.datetime.now() - time_start

print('\n Simulation took', time_took,'(hours:minutes:seconds)')



#############################################################################################################################################################################################
#GENERATING OUTPUT METRICS AND FILE
#################################################################################################################################################################################################

residuals = (SIM_total_flux_spectrum_model_bins[1] - model_spec[1]) / model_spec[1]
res1 = abs(residuals)
nans= np.isnan(res1)
res1[nans] = 0
infs = np.isinf(res1)
res1[infs] = 0

residuals_av = np.median(res1)*100
residuals_spread = scipy.stats.median_abs_deviation(res1)*100
SNR_av = np.median(np.nan_to_num(SNRs)[np.nonzero(np.nan_to_num(SNRs))])
SNR_spread = scipy.stats.median_abs_deviation(np.nan_to_num(SNRs[1])[np.nonzero(np.nan_to_num(SNRs[1]))])

if r_e_spread == True:
    R_E_pixels_IR = np.load('R_E_PIXELS/R_E_PIXELS_IR_ARM.npy')
    R_E_pixels_OPT = np.load('R_E_PIXELS/R_E_PIXELS_OPT_ARM.npy')
else:
    R_E_pixels_IR = np.load('R_E_PIXELS/R_E_PIXELS_IR.NPY')
    R_E_pixels_OPT = np.load('R_E_PIXELS/R_E_PIXELS_OPT.NPY')

R_E_IR_std = np.std(R_E_pixels_IR)
R_E_OPT_std = np.std(R_E_pixels_OPT)

av_R = np.median(Rs)
av_R_dist = scipy.stats.median_abs_deviation(Rs)

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

if IR_arm == True: 
    f.write('Number of dead spectal pixels in OPT arm: %i \n'%int(dead_pixel_perc*n_pixels))
    f.write('Number of dead spectal pixels in IR arm: %i \n'%int(dead_pixel_perc*n_pixels))
    f.write('MKID energy resolution at fiducial point standard deviation for OPT arm: %.2f \n'%(R_E_OPT_std))
    f.write('MKID energy resolution at fiducial point standard deviation for IR arm: %.2f \n\n'%(R_E_IR_std))
else:
    f.write('Number of dead spectal pixels in KIDSpec: %i \n'%int(dead_pixel_perc*n_pixels))
    f.write('MKID energy resolution at fiducial point standard deviation for KIDSpec: %.2f \n\n'%(R_E_OPT_std))

f.write('Simulation spectrum magnitudes:\n')
for i in range(int(len(SIM_obj_mags[0]))):
    f.write('%s --> %s \n'%(SIM_obj_mags[1][i],SIM_obj_mags[0][i]))

f.write('Flux reduced by factor of %.5f\n'%mag_reduce)
f.write('Spectrum redshifted by %.3f\n'%redshift)

f.write('\n')

f.write('> Result parameters: \n\n')
f.write('OPT orders observed: %i \n'%len(orders_opt))
f.write('IR orders observed: %i \n \n'%len(orders_ir))

f.write('Wavelength range tested: %i - %i nm \n'%(lambda_low_val,lambda_high_val))
f.write('Recreated spectrum resolution: %i - %i \n'%(R_low,R_high))
f.write('Recreated spectrum resolution average: %i +/- %i \n'%(av_R,av_R_dist))
f.write('Average residuals: (%.3f +/- %.3f)%% \n'%(residuals_av,residuals_spread))
f.write('R value: %.3f \n'%R_value_stat)
f.write('Average SNR: %.3f +/- %.3f \n'%(SNR_av,SNR_spread))
f.write('Percentage of photons which were misidentified (total): %.2f\n'%percentage_misidentified_tot)
f.write('Percentage of photons which were misidentified (average per pixel): %.2f\n\n'%percentage_misidentified_pp)


f.write('Simulation run duration: %s (hours:minutes:seconds)\n'%time_took)
f.close()








