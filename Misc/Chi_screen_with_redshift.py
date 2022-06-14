# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:07:36 2021

@author: BVH
"""


import numpy as np 
import matplotlib.pyplot as plt
import scipy.stats as stats
from IPython import display
from useful_funcs import redshifter,R_value,mag_calc,rebinner_with_bins_with_if,mag_calc_bandpass,gaussian_eq,chi_sq_test_spectrum,t_test
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from flux_to_photons_conversion import photons_conversion_with_bin_calc

def gr_0_filter(rs,zs):
    x = zs[rs>0.5]
    y = rs[rs>0.5]
    out_spec = np.zeros((2,len(x)))
    out_spec[0] += x
    out_spec[1] += y*1000
    return out_spec

def z_entry_generator(spec):
    zs = np.array([])
    for i in range(len(spec[0])):
        this_z = np.ones(int(spec[1][i]))*spec[0][i]
        zs = np.append(zs,this_z)
    return zs

def stats_ret(arr):
    mean = np.mean(arr)
    std_dev = np.std(arr)
    std_err = std_dev / np.sqrt(len(arr))
    return np.array([mean,std_dev,std_err])

def weighted_mean(zs,weights):
    return np.sum(zs*weights) / np.sum(weights)    

def spec_shortener(long_spec,short_spec):
    wls_keep = []
    for i in range(len(long_spec[0])):
        if (long_spec[0][i] == short_spec[0]).any():
            wls_keep.append(i)
    new_spec = np.zeros_like(short_spec)
    new_spec[0] += short_spec[0]
    new_spec[1] += long_spec[1][np.asarray(wls_keep)]
    return new_spec

def chi_sq_sig(o,e,sig_sq):
    x = o[1][~np.isnan(o[1])]
    x_1 = x
    y = e[1][~np.isnan(o[1])]
    x = x[x < 1e308]
    x_2 = x
    x = x[x > 0]
    y = y[x_1 < 1e308]
    y = y[x_2 > 0]
    sig_sq2 = sig_sq[1][~np.isnan(o[1])]
    sig_sq2 = sig_sq2[x_1 < 1e308]
    sig_sq2 = sig_sq2[x_2 > 0]
    return np.sum( np.square(x-y) / np.square(sig_sq2) ) / (len(x)-1)

def data_interpolater(wls,data_spec):
    out_y = np.zeros_like(wls)
    data_f = interp1d(data_spec[0],data_spec[1])
    for i in range(len(wls)):
        if data_spec[0][-1] >= wls[i] >= data_spec[0][0]:
            out_y[i] += data_f(wls[i])
    return out_y

def nan_replacer_with_1(sigs,spec):
    sigs_new = np.zeros_like(spec)
    sigs_new[0] += spec[0]
    for i in range(len(sigs_new[1])):
        if (sigs[0] == spec[0][i]).any():
                coord = np.where(sigs[0] == spec[0][i])[0][0]
                sigs_new[1][i] += np.nan_to_num(sigs[1][coord],nan=1)
    return sigs_new

def SNR_prep(snrs,spec):
    new_snrs = np.zeros_like(spec)
    new_snrs[0] += spec[0]
    new_snrs[1] += data_interpolater(spec[0], snrs)
    new_snrs[1][new_snrs[1] == 0] = 1
    new_snrs[1] = np.mean(spec[1]) #/ new_snrs[1]
    #new_snrs[1] *= np.mean(spec[1])
    #new_snrs[1] **= 0.5
    return new_snrs

def photo_data_stats(no_sets,len_photo_data):
    no_sets += 1
    data_sets = np.zeros((no_sets,len_photo_data))
    for i in range(no_sets):
        #print(i)
        if i == 0:
            curr_data = np.load('Misc/JWST_MOCK_SPEC_5_7_PHOTO.npy')[:,:10]
            curr_data_ph = photons_conversion_with_bin_calc(curr_data)
            data_sets[i] += curr_data_ph[1]
        else:
            curr_data = np.load('Misc/JWST_MOCK_SPEC_5_7_PHOTO_%i.npy'%(i-1))[:,:10]
            curr_data_ph = photons_conversion_with_bin_calc(curr_data)
            data_sets[i] += curr_data_ph[1]
    means = np.zeros((2,len(curr_data_ph[1])))
    stds = np.zeros((2,len(curr_data_ph[1])))
    mean_sets = np.mean(data_sets,axis=0)
    std_sets = np.std(data_sets,axis=0)
    means[0] += curr_data_ph[0]
    means[1] += mean_sets
    stds[0] += curr_data_ph[0]
    stds[1] += std_sets
    return data_sets,means,stds
        

base_z = 5.7
file_z = '5_7'
plots = False

print('\nLoading in files.')
ksim_spec = np.load('Misc/JWST_MOCK_SPEC_%s_KSIM.npy'%file_z)
data_spec = np.load('Misc/JWST_MOCK_SPEC_%s_DATA.npy'%file_z)
xsim_spec = np.load('Misc/JWST_MOCK_SPEC_%s_XSIM.npy'%file_z)
psim_spec = np.load('Misc/JWST_MOCK_SPEC_%s_PHOTO.npy'%file_z)[:,:10]

#ksim_sigsq1 = np.load('Misc/KSIM_SIGSQ_5_7.npy')
ksim_snrs1 = np.load('Misc/KSIM_SNRS_5_7.npy')
psim_snrs1 = np.load('Misc/PSIM_SNRS_5_7.npy')[:,:10]
xsim_snrs1 = np.load('Misc/XSIM_SNRS_5_7.npy')

'''
ksim_sigsq = nan_replacer_with_1(ksim_sigsq1, ksim_spec)
psim_sigsq = nan_replacer_with_1(psim_sigsq1, psim_spec)
xsim_sigsq = nan_replacer_with_1(xsim_sigsq1, xsim_spec)
ksim_sigsq = photons_conversion_with_bin_calc(ksim_sigsq)
psim_sigsq = photons_conversion_with_bin_calc(psim_sigsq)
xsim_sigsq = photons_conversion_with_bin_calc(xsim_sigsq)


ksim_sigsq = np.zeros_like(ksim_spec)
ksim_sigsq[0] += ksim_spec[0]
for i in range(len(ksim_sigsq[1])):
    if (ksim_sigsq1[0] == ksim_spec[0][i]).any():
            coord = np.where(ksim_sigsq1[0] == ksim_spec[0][i])[0][0]
            #print(ksim_sigsq1[1][coord])
            ksim_sigsq[1][i] += np.nan_to_num(ksim_sigsq1[1][coord],nan=1)
ksim_sigsq = photons_conversion_with_bin_calc(ksim_sigsq)
'''


ksim_spec_filter1 = mag_calc(ksim_spec,return_flux=True)[2:]
ksim_spec_filter = np.zeros((2,10))
ksim_spec_filter[0] += ksim_spec_filter1[0][:10]
ksim_spec_filter[1] += ksim_spec_filter1[1][:10]
#ksim_sigsq_filter1 = mag_calc(ksim_sigsq,return_flux=True)[2:]
#ksim_sigsq_filter = np.zeros((2,10))
#ksim_sigsq_filter[0] += ksim_sigsq_filter1[0][:10]
#ksim_sigsq_filter[1] += ksim_sigsq_filter1[1][:10]

ksim_spec_bandpass = mag_calc_bandpass(ksim_spec,return_flux=True)[1:]
#ksim_sigsq_bandpass = mag_calc_bandpass(ksim_sigsq,return_flux=True)[1:]
xsim_spec_bandpass = mag_calc_bandpass(xsim_spec,return_flux=True)[1:]
#xsim_sigsq_bandpass = mag_calc_bandpass(xsim_sigsq,return_flux=True)[1:]

xsim_spec_filter1 = mag_calc(xsim_spec,return_flux=True)[2:]
xsim_spec_filter = np.zeros((2,10))
xsim_spec_filter[0] += xsim_spec_filter1[0][:10]
xsim_spec_filter[1] += xsim_spec_filter1[1][:10]
#xsim_sigsq_filter1 = mag_calc(xsim_sigsq,return_flux=True)[2:]
xsim_sigsq_filter = np.zeros((2,10))
#xsim_sigsq_filter[0] += xsim_sigsq_filter1[0][:10]
#xsim_sigsq_filter[1] += xsim_sigsq_filter1[1][:10]


test_zs = np.linspace(0.1,10,3000)
chis_ksim = np.zeros(len(test_zs))
chis_psim = np.zeros(len(test_zs))
chis_xsim = np.zeros(len(test_zs))
chis_k_filt = np.zeros(len(test_zs))
chis_k_bandpass = np.zeros(len(test_zs))
chis_x_filt = np.zeros(len(test_zs))
chis_x_bandpass = np.zeros(len(test_zs))

loop_psim_ph = photons_conversion_with_bin_calc(psim_spec)
loop_ksim_bandpass_ph = photons_conversion_with_bin_calc(ksim_spec_bandpass)
loop_ksim_filt_ph = photons_conversion_with_bin_calc(ksim_spec_filter)
loop_ksim_ph = photons_conversion_with_bin_calc(ksim_spec)
loop_xsim_ph = photons_conversion_with_bin_calc(xsim_spec)
loop_xsim_bandpass_ph = photons_conversion_with_bin_calc(xsim_spec_bandpass)
loop_xsim_filt_ph = photons_conversion_with_bin_calc(xsim_spec_filter)


psim_data_sets,psim_mean_sets,psim_std_sets = photo_data_stats(14,len(loop_psim_ph[1]))

ksim_snrs1 = nan_replacer_with_1(ksim_snrs1, ksim_spec)
ksim_snrs = SNR_prep(ksim_snrs1,loop_ksim_ph)
ksim_snrs_filt = SNR_prep(ksim_snrs1,loop_ksim_filt_ph)
ksim_snrs_bandpass = SNR_prep(ksim_snrs1,loop_ksim_bandpass_ph)

xsim_snrs1 = nan_replacer_with_1(xsim_snrs1, xsim_spec)
xsim_snrs = SNR_prep(xsim_snrs1,loop_xsim_ph)
xsim_snrs_filt = SNR_prep(xsim_snrs1,loop_xsim_filt_ph)
xsim_snrs_bandpass = SNR_prep(xsim_snrs1,loop_xsim_bandpass_ph)

psim_snrs1 = nan_replacer_with_1(psim_snrs1,psim_spec)
psim_snrs = SNR_prep(psim_snrs1,loop_psim_ph)


print('\nBeginning redshift loop.')
for i in range(len(test_zs)):
    loop_data_spec = redshifter(data_spec,base_z,test_zs[i])
    
    _,_,data_loop_wls,data_loop_fluxes = mag_calc(loop_data_spec,return_flux=True)
    loop_data_photo = np.zeros((2,10))
    loop_data_photo[0] += data_loop_wls[:10]
    loop_data_photo[1] += data_loop_fluxes[:10]
    
    data_loop_wls,data_loop_fluxes = mag_calc_bandpass(loop_data_spec,return_flux=True)[1:]
    loop_data_bandpass = np.zeros((2,9))
    loop_data_bandpass[0] += data_loop_wls
    loop_data_bandpass[1] += data_loop_fluxes
    
    #KIDSpec spectrum
    loop_data_ph = photons_conversion_with_bin_calc(loop_data_spec)
    loop_data_ks_ph = np.zeros_like(ksim_spec)
    loop_data_ks_ph[0] += ksim_spec[0]
    loop_data_ks_ph[1] += data_interpolater(ksim_spec[0], loop_data_ph)
    #loop_data_ks_ph[1] /= np.max(loop_data_ks_ph[1])
    #norm_ksim = np.zeros((2,len(loop_ksim_ph[1])))
    #norm_ksim[0] += loop_ksim_ph[0]
    #norm_ksim[1] += loop_ksim_ph[1] / np.max(loop_ksim_ph[1])
    chis_ksim[i] = chi_sq_sig(loop_ksim_ph,loop_data_ks_ph,ksim_snrs)
    #KIDSpec filter
    loop_data_photo_ph = photons_conversion_with_bin_calc(loop_data_photo)
    chis_k_filt[i] = chi_sq_sig(loop_ksim_filt_ph,loop_data_photo_ph,ksim_snrs_filt)
    #KIDSpec bandpass
    loop_data_bandpass_ph = photons_conversion_with_bin_calc(loop_data_bandpass)
    chis_k_bandpass[i] = chi_sq_sig(loop_ksim_bandpass_ph,loop_data_bandpass_ph,ksim_snrs_bandpass)
    
    #photometry
    chis_psim[i] = chi_sq_sig(loop_psim_ph,loop_data_photo_ph,psim_std_sets)
    
    #X-Shooter
    loop_data_x_ph = np.zeros_like(xsim_spec)
    loop_data_x_ph[0] += xsim_spec[0]
    loop_data_x_ph[1] += data_interpolater(xsim_spec[0], loop_data_ph)
    chis_xsim[i] = chi_sq_sig(loop_xsim_ph,loop_data_x_ph,xsim_snrs)
    #X-Shooter filter
    chis_x_filt[i] = chi_sq_sig(loop_xsim_filt_ph,loop_data_photo_ph,xsim_snrs_filt)
    #X-Shooter bandpass
    chis_x_bandpass[i] = chi_sq_sig(loop_xsim_bandpass_ph,loop_data_bandpass_ph,xsim_snrs_bandpass)
    
    if plots == True:
        if (i==np.array([500,1000,1696,2000,2500,3000,4000])).any():
            plt.figure()
            plt.plot(loop_psim_ph[0],loop_psim_ph[1])
            plt.plot(loop_data_photo_ph[0],loop_data_photo_ph[1])
            plt.title('%.3f'%test_zs[i])
            plt.figure()
            plt.plot(psim_spec[0],psim_spec[1])
            plt.plot(loop_data_photo[0],loop_data_photo[1])
            plt.title('%.3f'%test_zs[i])
    
    
    perc_done = ((i+1)/len(test_zs))*100
    print('\r%.2f %% of redshifts complete'%(perc_done),end='',flush=True)
    
chosen_lw = 1
coord_low = 100
coord_high = -100
plt.figure()
plt.title('z=%.3f'%base_z)
plt.plot(test_zs[coord_low:coord_high],chis_ksim[coord_low:coord_high],'r-',label='KIDSpec',linewidth=chosen_lw)
#plt.plot(test_zs,chis_psim,'b-',label='Imager Photometry',linewidth=chosen_lw)
#plt.plot(test_zs,chis_xsim,'k-',label='X-Shooter',linewidth=chosen_lw)
#plt.plot(test_zs,chis_k_filt,'g-',label='KIDSpec photometry',linewidth=chosen_lw)
#plt.plot(test_zs,chis_x_filt,'m-',label='X-Shooter photometry',linewidth=chosen_lw)
#plt.plot(test_zs,chis_k_bandpass,'c-',label='KIDSpec top hat bandpass photometry',linewidth=chosen_lw)
#plt.plot(test_zs,chis_x_bandpass,'y-',label='X-Shooter top hat bandpass photometry',linewidth=chosen_lw)
plt.xlabel('Redshift')
plt.ylabel('Chi-sq value')
plt.legend(loc='best')


plt.figure()
plt.title('z=%.3f'%base_z)
plt.plot(test_zs[coord_low:coord_high],chis_psim[coord_low:coord_high],'b-',label='Imager Photometry',linewidth=chosen_lw)
plt.xlabel('Redshift')
plt.ylabel('Chi-sq value')
plt.legend(loc='best')
plt.ylim((0,10))


plt.figure()
plt.title('z=%.3f'%base_z)
plt.plot(test_zs[coord_low:coord_high],chis_xsim[coord_low:coord_high],'k-',label='X-Shooter',linewidth=chosen_lw)
plt.xlabel('Redshift')
plt.ylabel('Chi-sq value')
plt.legend(loc='best')

plt.figure()
plt.title('z=%.3f'%base_z)
plt.plot(test_zs[coord_low:coord_high],chis_k_filt[coord_low:coord_high],'g-',label='KIDSpec photometry',linewidth=chosen_lw)
plt.xlabel('Redshift')
plt.ylabel('Chi-sq value')
plt.legend(loc='best')

plt.figure()
plt.title('z=%.3f'%base_z)
plt.plot(test_zs[coord_low:coord_high],chis_x_filt[coord_low:coord_high],'m-',label='X-Shooter photometry',linewidth=chosen_lw)
plt.xlabel('Redshift')
plt.ylabel('Chi-sq value')
plt.legend(loc='best')

plt.figure()
plt.title('z=%.3f'%base_z)
plt.plot(test_zs[coord_low:coord_high],chis_k_bandpass[coord_low:coord_high],'c-',label='KIDSpec top hat bandpass photometry',linewidth=chosen_lw)
plt.xlabel('Redshift')
plt.ylabel('Chi-sq value')
plt.legend(loc='best')


plt.figure()
plt.title('z=%.3f'%base_z)
plt.plot(test_zs[coord_low:coord_high],chis_x_bandpass[coord_low:coord_high],'y-',label='X-Shooter top hat bandpass photometry',linewidth=chosen_lw)
plt.xlabel('Redshift')
plt.ylabel('Chi-sq value')
plt.legend(loc='best')











