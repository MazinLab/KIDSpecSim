# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:26:35 2021

@author: BVH
"""

import numpy as np
import matplotlib.pyplot as plt


folder = '19012021_PTS'

data = np.load('%s/DATA_SPECTRUM.npy'%folder)
spec_1 = np.load('%s/SIM_FLUX_SPECTRUM_GD71.npy'%folder)
spec_2 = np.load('%s/SIM_FLUX_SPECTRUM_GD71_2.npy'%folder)


plt.figure()
plt.plot(spec_1[0],spec_1[1])
plt.plot(spec_2[0],spec_2[1])
plt.xlim(500,1000)
plt.ylim(0,7e-14)

plt.figure()
plt.plot(spec_2[0],spec_2[1])
plt.plot(data[0],data[1])


