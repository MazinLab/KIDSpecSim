# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:44:50 2021

@author: BVH
"""

import numpy as np
import matplotlib.pyplot as plt


one_arm = np.zeros((10,3))
two_arm = np.zeros((10,3))

one_arm[:,0] += np.array([30,60,100,200,300,600,900,1200,1800,3600])#exp_time
one_arm[:,1] += np.array([7.172,7.158,6.873,7.190,7.277,7.246,7.289,7.420,7.410,7.460])#residual
one_arm[:,2] += np.array([16,23,30,42,52,73,90,104,127,179])#snr

two_arm[:,0] += np.array([30,60,100,200,300,600,900,1200,1800,3600])
two_arm[:,1] += np.array([3.208,2.297,1.740,1.266,1.012,0.747,0.635,0.562,0.490,0.401])
two_arm[:,2] += np.array([15,21,27,38,46,65,79,91,111,157])


plt.figure()
plt.plot(one_arm[:,0],one_arm[:,1],label='1 arm')
plt.plot(two_arm[:,0],two_arm[:,1],label='2 arm')
plt.xlabel('Exposure time / s')
plt.ylabel('Residuals / %')
plt.legend(loc='best')

plt.figure()
plt.plot(one_arm[:,0],one_arm[:,2],label='1 arm')
plt.plot(two_arm[:,0],two_arm[:,2],label='2 arm')
plt.xlabel('Exposure time / s')
plt.ylabel('SNR')
plt.legend(loc='best')





