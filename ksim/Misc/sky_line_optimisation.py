# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:43:50 2020

@author: BVH
"""


import numpy as np
import datetime
import os
import scipy.stats
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import shutil

from useful_funcs import data_extractor, telescope_effects, specific_wl_overlap, wavelength_array_maker, \
        order_doubles, grat_eff_sum, sky_addition,SNR_calc,post_processing,nearest,mag_calc,sky_spec_generator, \
            data_extractor_TLUSTY,wavelength_finder,cutoff_sorter,spec_slicer,XS_ETC_spec,SNR_calc_pred
from parameters import object_file,pix_mult,n_pixels,folder,object_name,object_x,object_y,slicers,slitlet_tranmissions, \
        mirr_diam,exposure_t,seeing,airmass,slit_width,slit_length,alpha_val,phi_val,OPT_grooves,IR_alpha,IR_phi,IR_grooves, \
        pix_fov,ER_band_low,lambda_low_val,lambda_high_val
from flux_to_photons_conversion import photons_conversion,flux_conversion,photons_conversion_2,flux_conversion_2
from grating import grating_orders_2_arms,grating_binning,grating_binning_lower_R
from apply_QE import QE


plt.rcParams.update({'font.size': 17})

time_start = datetime.datetime.now()

sky_bckgrd_frame = False

delete_folders = True

sky_spec_only = False

Sky_R = 17000

atmos_present = True

PTS_ = True

lower_R = False

IR_arm = True

z = 1

TLUSTY = False
row = 3

redshift = 0.0
mag_reduce = 1.5

photon_spec = sky_spec_generator(Sky_R,plotting=False)

orders,order_wavelengths,grating_efficiency,orders_opt,order_wavelengths_opt,efficiencies_opt,orders_ir,order_wavelengths_ir,efficiencies_ir,order_issues,order_issues_loc,cutoffs_ir,cutoffs_opt = grating_orders_2_arms('Casini',lower_R=lower_R,plotting=False)

detec_spec = wavelength_array_maker(order_wavelengths) #forms a 1D array based on the wavelengths each order sees
    
wavelength_overlaps,overlap_amounts,raw_overlap =  specific_wl_overlap(order_wavelengths,orders,detec_spec) #finds what orders overlap and at what array coordinates

twinned_wavelengths = order_doubles(order_wavelengths) #finds any wavelengths which appear twice 

repeated_wls = wavelength_finder(order_wavelengths,wavelength_overlaps)

cutoff_wls_ir = cutoff_sorter(cutoffs_opt,order_wavelengths_opt)
cutoffs_wls_opt = cutoff_sorter(cutoffs_ir,order_wavelengths_ir)


spec_to_instr = telescope_effects(photon_spec,plotting=False)

spec_QE = QE(spec_to_instr,constant=False,plotting=False)

pixel_sums_opt,opt_zeroed_wls = grating_binning(spec_QE,order_wavelengths_opt,orders_opt,efficiencies_opt,wavelength_overlaps,overlap_amounts,raw_overlap,twinned_wavelengths,repeated_wls,order_issues,order_issues_loc,cutoff_wls_ir,cutoffs_wls_opt,IR=False,OPT=True,plotting=False,PTS_=PTS_)
pixel_sums_ir,ir_zeroed_wls = grating_binning(spec_QE,order_wavelengths_ir,orders_ir,efficiencies_ir,wavelength_overlaps,overlap_amounts,raw_overlap,twinned_wavelengths,repeated_wls,order_issues,order_issues_loc,cutoff_wls_ir,cutoffs_wls_opt,IR=True,OPT=False,plotting=False,PTS_=PTS_)



def order_pixel_plotter(pixel_sums,order_list,IR=False):
    
    if IR == True:
        plt_label = 'IR photons'
    else:
        plt_label = 'OPT photons'
    
    pix_orien = np.rot90(pixel_sums)
    
    plt.figure()
    im = plt.imshow(pix_orien,cmap='plasma',extent=[0, n_pixels, min(order_list), max(order_list)],origin='lower',interpolation='none',aspect='auto')
    plt.colorbar(mappable=im,label=plt_label)
    plt.yticks(order_list)
    





