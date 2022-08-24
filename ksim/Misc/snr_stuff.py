# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 14:21:50 2021

@author: BVH
"""



import numpy as np
import matplotlib.pyplot as plt

data = np.load('DATA_SPECTRUM.npy')
spec1 = np.load('SIM_FLUX_SPECTRUM_GD71.npy')
spec2 = np.load('SIM_FLUX_SPECTRUM_GD71_2.npy')

plt.figure()
plt.plot(spec1[0],spec1[1])
plt.plot(spec2[0],spec2[1])
plt.plot(data[0],data[1])

'''
SNRs_y = SNRs[1][SNRs[1] > 0]
SNRs_x = SNRs[0][SNRs[1] > 0]

SNRs_ps_y = predicted_SNRs_s[predicted_SNRs_s > 0]
SNRs_ps_x = SNRs[0][predicted_SNRs_s > 0]

SNRs_px_y = predicted_SNRs_x[predicted_SNRs_x > 0]
SNRs_px_x = SNRs[0][predicted_SNRs_x > 0]

plt.figure()
plt.plot(SNRs_x,SNRs_y,'b-',label='KSIM')
plt.plot(SNRs_ps_x,SNRs_ps_y,'r-',label='KSIM + SOXS noise')
plt.plot(SNRs_px_x,SNRs_px_y,'g-',label='KSIM + X-Shooter noise')
plt.ylabel('SNR')
plt.xlabel('Wavelength / nm')
plt.legend(loc='best')
'''
'''
plt.figure()
plt.plot(SNRs[0],SNRs[1],'b-',label='KSIM')
plt.plot(SNRs[0],predicted_SNRs_s,'r-',label='KSIM + SOXS noise')
plt.plot(SNRs[0],predicted_SNRs_x,'g-',label='KSIM + X-Shooter noise')
plt.ylabel('SNR')
plt.xlabel('Wavelength / nm')
plt.legend(loc='best')
'''





