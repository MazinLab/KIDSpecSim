# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:30:17 2022

@author: BVH
"""


import numpy as np
import matplotlib.pyplot as plt
from useful_funcs import redshifter
from scipy.interpolate import interp1d

def chi_sq_sig(o,e,sig_sq):
    x = o[1][~np.isnan(o[1])]
    x_1 = x
    y = e[1][~np.isnan(o[1])]
    x = x[x < 1e308]
    x_2 = x
    x = x[x > 0]
    y = y[x_1 < 1e308]
    y = y[x_2 > 0]
    sig_sq2 = sig_sq[~np.isnan(o[1])]
    sig_sq2 = sig_sq2[x_1 < 1e308]
    sig_sq2 = sig_sq2[x_2 > 0]
    return np.sum( np.square(x-y) / np.square(sig_sq2) ) / (len(x)-1)

def spec_setup_for_chi(wls,spec,min_wl,max_wl):
    curr_z_prep_spec = np.zeros((2,len(wls)))
    curr_z_prep_spec[0] += wls
    x = np.copy(wls)
    x = np.insert(x,0,min_wl)
    x = np.insert(x,-1,max_wl)
    y = np.zeros_like(x)
    y[1:-1] += spec[1]
    curr_z_prep_spec[1] += np.interp(wls,x,y)
    return curr_z_prep_spec


##############################################################################################################
#KIDSpec spectrum
##############################################################################################################


base_z = 5.7
file_z = '5_7'

atmos_ir = np.loadtxt('Atmosphere_Data/cptrans_zm_43_15.dat')
atmos_ir[:,0] *= 1000 #convert to nm
atmos_opt = np.loadtxt('Atmosphere_Data/optical_extinction_curve_gemini.txt')[:-1,:]
atmos_opt[:,1] = np.exp(-(atmos_opt[:,1]*1.5)/2.5) #converting to transmission
atmos = np.append(atmos_opt,atmos_ir,axis=0)
atmos_func = interp1d(atmos[:,0],atmos[:,1])
stand_star_fac = np.load('FACTORS_STAND_STAR_GD71.npy')
stand_star_wls = np.load('DATA_SPECTRUM.npy')[0]
stand_star_func = interp1d(stand_star_wls,stand_star_fac)

ksim_spec1 = np.load('Misc/JWST_MOCK_SPEC_%s_KSIM.npy'%file_z)
data_spec = np.load('Misc/JWST_MOCK_SPEC_%s_DATA.npy'%file_z)

#ksim_spec1[1] *= (1000/max(ksim_spec1[1]))
#data_spec[1] *= (1000/max(data_spec[1]))

ksim_spec = ksim_spec1[:,np.isnan(ksim_spec1[1]) == False]
data_spec = data_spec[:,np.isnan(ksim_spec1[1]) == False]

wls = np.linspace(min(ksim_spec[0]),max(ksim_spec[0]),len(ksim_spec[0]))

test_zs = np.linspace(1,10,1000)

min_wl = min(redshifter(data_spec,base_z,test_zs[0])[0])-1
max_wl = max(redshifter(data_spec,base_z,test_zs[-1])[0])+1

ksim_spec_chi = spec_setup_for_chi(wls,ksim_spec,min_wl,max_wl)
data_spec_chi = spec_setup_for_chi(wls,data_spec,min_wl,max_wl)
stand_star_fac_chi = stand_star_func(ksim_spec_chi[0])
atmos_chi = atmos_func(ksim_spec_chi[0])

plt.figure()
plt.plot(ksim_spec_chi[0],ksim_spec_chi[1])
plt.plot(data_spec_chi[0],data_spec_chi[1],label='data')
plt.legend(loc='best')

chi_sigma = data_spec_chi[1] - ksim_spec_chi[1]

plt.figure()
plt.plot(ksim_spec_chi[0],abs(chi_sigma))
plt.plot(data_spec_chi[0],np.sqrt((data_spec_chi[1])+(data_spec_chi[1]/stand_star_fac_chi)+
                                  np.sqrt(data_spec_chi[1]/atmos_chi)),label='sqrt data')
plt.legend(loc='best')

chis = np.zeros(len(test_zs))

for i in range(len(test_zs)):
    curr_z_data_spec = redshifter(data_spec,base_z,test_zs[i])
    
    curr_z_data_chi = spec_setup_for_chi(wls,curr_z_data_spec,min_wl,max_wl)
    
    chis[i] += chi_sq_sig(ksim_spec_chi,curr_z_data_chi,chi_sigma)
    
plt.figure()
plt.semilogy(test_zs,chis)
plt.xlabel('Redshift')
plt.ylabel('Reduced Chi score')





##############################################################################################################
#Photometry spectrum
##############################################################################################################



psim_spec = np.load('PHOTOMETRY_z_5_7_SIM.npy')
psim_data = np.load('PHOTOMETRY_z_5_7_MODEL.npy')

psim_spec[1] *= (1000/max(psim_spec[1]))
psim_data[1] *= (1000/max(psim_data[1]))


min_wl = min(redshifter(psim_data,base_z,test_zs[0])[0])-1
max_wl = max(redshifter(psim_data,base_z,test_zs[-1])[0])+1

psim_spec_chi = spec_setup_for_chi(psim_spec[0],psim_spec,min_wl,max_wl)
#psim_spec_chi[1][psim_spec_chi[1]<0] = 0
psim_data_chi = spec_setup_for_chi(psim_spec[0],psim_data,min_wl,max_wl)

chi_sigma_p = psim_data_chi[1] - psim_spec_chi[1]
chi_sigma_p[chi_sigma_p == 0] = 1

chis_p = np.zeros(len(test_zs))

for i in range(len(test_zs)):
    curr_z_data_spec = redshifter(psim_data,base_z,test_zs[i])
    
    curr_z_data_chi = spec_setup_for_chi(psim_spec_chi[0],curr_z_data_spec,min_wl,max_wl)
    
    chis_p[i] += chi_sq_sig(psim_spec_chi,curr_z_data_chi,chi_sigma_p)
    
plt.figure()
plt.semilogy(test_zs,chis_p)
plt.xlabel('Redshift')
plt.ylabel('Reduced Chi score')









