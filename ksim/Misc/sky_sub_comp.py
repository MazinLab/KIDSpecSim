# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 11:42:42 2021

@author: BVH
"""

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 28})

#spec_no_skysub = np.load('STANDSTAR_NO_SKY_18062021_OLD.npy')
#spec_with_skysub = np.load('STANDSTAR_WITH_SKY_SUB_18062021_OLD.npy')
#spec_data = np.load('STANDSTAR_DATA_SPEC.npy')

spec_no_skysub = np.load('STANDSTAR_NO_SKY_SUB_22062021.npy')
spec_with_skysub = np.load('STANDSTAR_WITH_SKY_SUB_22062021.npy')
spec_data = np.load('STANDSTAR_DATA_SPEC_22062021.npy')

#spec_with_skysub = np.load('STANDSTAR_WITH_SKY_SUB_22062021_NOREDUCE.npy')
#spec_data = np.load('STANDSTAR_DATA_SPEC_22062021_NOREDUCE.npy')

plt.figure()

plt.plot(spec_no_skysub[0],spec_no_skysub[1],'g-',label='No sky sub.')
plt.plot(spec_with_skysub[0],spec_with_skysub[1],'b-',label='With sky sub.',alpha=0.6)
plt.plot(spec_data[0],spec_data[1],'r-',label='Data spectrum',alpha=0.6)

plt.xlabel('Wavelength / nm')
plt.ylabel('Flux / $ergcm^{-2}s^{-1}\AA^{-1}$ ')
plt.legend(loc='best')




