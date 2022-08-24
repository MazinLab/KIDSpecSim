# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 14:47:19 2021

@author: BVH
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 17})

pix = np.array([1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000])
R_low = np.array([1331,1998,2664,3331,3998,4664,5331,5997,6664,7330,7997])
R_high = np.array([2354,3532,4710,5889,7067,8245,9424,10602,11780,12958,14137])
R_av = (R_low+R_high) / 2

plt.figure()
plt.fill_between(pix,R_low,R_high,color='r',alpha=0.5)
plt.plot(pix,R_av,'k-')
plt.plot(pix,R_av,'ko')
plt.xlim(1000,6000)
plt.xlabel('MKID pixels / arm')
plt.ylabel('Spectral Resolution')





