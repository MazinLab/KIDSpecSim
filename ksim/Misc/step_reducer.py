# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:23:26 2021

@author: BVH
"""

import numpy as np
from useful_funcs import data_extractor_TLUSTY,data_extractor_TLUSTY_joint_spec,nearest
import matplotlib.pyplot as plt

in_spec = np.load('SHIFTED_SPECTRA/T498g775_2_34kpc_317_steps.NPY')

desired_steps = 200
steps = np.arange(0,len(in_spec[0,:])-1,(len(in_spec[0,:])-1)/desired_steps)


in_spec2 = np.zeros((len(in_spec[:,0]),len(steps)+1))
 

for i in range(len(steps)):
    in_spec2[:,i] += in_spec[:,int(np.round(steps[i],decimals=0))]
    
in_spec2[:,-1] += in_spec[:,-1]

#np.save('T498g775_2_34kpc_100_steps.npy',in_spec2)
#np.save('T100g825_2_34kpc_100_steps.npy',in_spec2)

fluxes = np.zeros((317,10000))
for i in range(317):
    coordf = nearest(in_spec[:,i],486,'coord')
    fluxes[i,:] += in_spec[:,-1][coordf-5000:coordf+5000]

fig,ax = plt.subplots()
im = ax.imshow(fluxes,interpolation=None,cmap='plasma',aspect='auto')

fluxes2 = np.zeros((desired_steps,10000))
for i in range(desired_steps):
    coordf = nearest(in_spec2[:,i],486,'coord')
    fluxes2[i,:] += in_spec[:,-1][coordf-5000:coordf+5000]

fig2,ax2 = plt.subplots()
im2 = ax2.imshow(fluxes2,interpolation=None,cmap='plasma',aspect='auto')

'''



row=15
row_number = 50

object_file_1 = 'T498g775_2_34kpc_78_steps'
object_file_2 = 'T100g825_2_34kpc_78_steps'

data_file_spec_comb = data_extractor_TLUSTY(object_file_1,row,plotting=False)
data_file_spec_2 = data_extractor_TLUSTY(object_file_2,row_number-(2+row),plotting=False)
data_file_spec_1 = np.zeros_like(data_file_spec_comb)
data_file_spec_1[0] += data_file_spec_comb[0]
data_file_spec_1[1] += data_file_spec_comb[1]
coord = nearest(data_file_spec_2[0],data_file_spec_comb[0][0],'coord')
    
if coord == 0:
    data_file_spec_comb[1] += data_file_spec_2[1]
else:
    data_file_spec_comb[1,:-coord] += data_file_spec_2[1,coord:]

data_file_spec_comb_2 = data_extractor_TLUSTY_joint_spec(object_file_1,object_file_2,row,plotting=True)

'''










