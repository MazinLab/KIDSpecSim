# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:16:48 2021

@author: BVH
"""

from useful_funcs import gaussian_eq
import numpy as np
import matplotlib.pyplot as plt

R_E = np.linspace(10,30,10)

x_data = np.linspace(350,550,10000)

for i in range(len(R_E)):
    sig1 = 428 / (R_E[i] * (2*np.sqrt( 2*np.log(2) )))
    sig2 = 472 / (R_E[i] * (2*np.sqrt( 2*np.log(2) )))
    
    gauss1 = gaussian_eq(x_data,428,sig1)*100
    gauss2 = gaussian_eq(x_data,472,sig2)*100
    
    plt.figure()
    plt.plot(x_data,gauss1,'b-',label='Order A')
    plt.plot(x_data,gauss2,'r-',label='Order B')
    plt.xlabel('Wavelength / nm')
    plt.ylabel('Intensity')
    plt.text(355,40,'$R_E$ = %i'%R_E[i])
    plt.ylim(0,45)
    plt.legend(loc='best')
    
    plt.savefig('R_E_%i_animation.png'%R_E[i],dpi=1000)

