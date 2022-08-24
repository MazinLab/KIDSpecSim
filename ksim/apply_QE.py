# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:45:27 2019

@author: BVH
"""

'''
Applies the quantum efficiency of the MKIDs to incoming photons.
This was taken straight from the repository.
'''


from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import numpy as np

def QE(spec,constant=False,plotting=False):
    
    if constant == True:
        QE = 0.75                   #used for GIGA-Z
        
        new_spec = (spec[0],spec[1]*QE)
        
    else:
        QE_lower_lambda = 200  # in nm
        QE_higher_lambda = 3000 
        QE_min = 0.22  # MKIDs quantum efficiency at 3000nm
        QE_max = 0.73  # MKIDs quantum efficiency at 200nm (Mazin et al, 2010)
        
        f1 = interp1d([QE_lower_lambda, QE_higher_lambda ], [QE_max,QE_min], bounds_error=False, fill_value=0)
        spec_new = spec[1] * f1(spec[0])
        new_spec = np.zeros_like(spec)
        new_spec[0] += spec[0]
        new_spec[1] += spec_new
        
    if plotting == True:
        plt.figure()
        plt.plot(spec[0],spec[1],'r-',label='Post telescope pre QE application')
        plt.plot(new_spec[0],new_spec[1],'b-',label='Post QE application',alpha=0.7)
        plt.xlabel('Wavelength / nm')
        plt.ylabel('Photons')
        plt.legend(loc='best')
    
    return new_spec
