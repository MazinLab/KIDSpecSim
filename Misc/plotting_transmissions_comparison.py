# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:07:09 2020

@author: BVH
"""

from matplotlib import pyplot as plt
import numpy as np

spec = np.load('slit_spectrum.npy')

colours = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']

slit_lengths = np.array([2.1,2.7,3.3])

trans_2_1_seeing_0_4 = np.load('slitlet_transmissions_seeing_0_4_slit_width_0_63_slits_3.npy')
trans_2_1_seeing_0_8 = np.load('slitlet_transmissions_seeing_0_8_slit_width_0_63_slits_3.npy')
trans_2_1_seeing_1_2 = np.load('slitlet_transmissions_seeing_1_2_slit_width_0_63_slits_3.npy')
trans_2_1_seeing_2_0 = np.load('slitlet_transmissions_seeing_2_0_slit_width_0_63_slits_3.npy')

trans_2_7_seeing_0_4 = np.load('slitlet_transmissions_seeing_0_4_slit_length_2_7_slits_3.npy')
trans_2_7_seeing_0_8 = np.load('slitlet_transmissions_seeing_0_8_slit_length_2_7_slits_3.npy')
trans_2_7_seeing_1_2 = np.load('slitlet_transmissions_seeing_1_2_slit_length_2_7_slits_3.npy')
trans_2_7_seeing_2_0 = np.load('slitlet_transmissions_seeing_2_0_slit_length_2_7_slits_3.npy')

trans_3_3_seeing_0_4 = np.load('slitlet_transmissions_seeing_0_4_slit_length_3_3_slits_3.npy')
trans_3_3_seeing_0_8 = np.load('slitlet_transmissions_seeing_0_8_slit_length_3_3_slits_3.npy')
trans_3_3_seeing_1_2 = np.load('slitlet_transmissions_seeing_1_2_slit_length_3_3_slits_3.npy')
trans_3_3_seeing_2_0 = np.load('slitlet_transmissions_seeing_2_0_slit_length_3_3_slits_3.npy')


trans_2_1_seeing_0_4_sum = np.sum(trans_2_1_seeing_0_4,axis=1)
trans_2_1_seeing_0_8_sum = np.sum(trans_2_1_seeing_0_8,axis=1)
trans_2_1_seeing_1_2_sum = np.sum(trans_2_1_seeing_1_2,axis=1)
trans_2_1_seeing_2_0_sum = np.sum(trans_2_1_seeing_2_0,axis=1)

trans_2_7_seeing_0_4_sum = np.sum(trans_2_7_seeing_0_4,axis=1)
trans_2_7_seeing_0_8_sum = np.sum(trans_2_7_seeing_0_8,axis=1)
trans_2_7_seeing_1_2_sum = np.sum(trans_2_7_seeing_1_2,axis=1)
trans_2_7_seeing_2_0_sum = np.sum(trans_2_7_seeing_2_0,axis=1)

trans_3_3_seeing_0_4_sum = np.sum(trans_3_3_seeing_0_4,axis=1)
trans_3_3_seeing_0_8_sum = np.sum(trans_3_3_seeing_0_8,axis=1)
trans_3_3_seeing_1_2_sum = np.sum(trans_3_3_seeing_1_2,axis=1)
trans_3_3_seeing_2_0_sum = np.sum(trans_3_3_seeing_2_0,axis=1)





fig = plt.figure()
AX = fig.add_subplot(111)
AX.spines['top'].set_color('none')
AX.spines['bottom'].set_color('none')
AX.spines['left'].set_color('none')
AX.spines['right'].set_color('none')
AX.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
AX.set_xlabel('Wavelength / nm')
AX.set_ylabel('Transmissions')
AX.set_title('Slit 1')
AX.yaxis.set_label_coords(-0.05,0.5)

ax = fig.add_subplot(411)
ax.plot(spec[0],trans_2_1_seeing_0_4_sum[:,0],'%s-'%colours[0],label='2.1" length')
ax.plot(spec[0],trans_2_7_seeing_0_4_sum[:,0],'%s--'%colours[0],label='2.7" length')
ax.plot(spec[0],trans_3_3_seeing_0_4_sum[:,0],'%s:'%colours[0],label='3.3" length')
ax.set_xticklabels([])
ax.legend(loc='best')
ax2 = ax.twinx()
ax2.set_ylabel('0.4" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax = fig.add_subplot(412)
ax.plot(spec[0],trans_2_1_seeing_0_8_sum[:,0],'%s-'%colours[0],label='2.1" length')
ax.plot(spec[0],trans_2_7_seeing_0_8_sum[:,0],'%s--'%colours[0],label='2.7" length')
ax.plot(spec[0],trans_3_3_seeing_0_8_sum[:,0],'%s:'%colours[0],label='3.3" length')
ax.set_xticklabels([])
ax2 = ax.twinx()
ax2.set_ylabel('0.8" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax = fig.add_subplot(413)
ax.plot(spec[0],trans_2_1_seeing_1_2_sum[:,0],'%s-'%colours[0],label='2.1" length')
ax.plot(spec[0],trans_2_7_seeing_1_2_sum[:,0],'%s--'%colours[0],label='2.7" length')
ax.plot(spec[0],trans_3_3_seeing_1_2_sum[:,0],'%s:'%colours[0],label='3.3" length')
ax.set_xticklabels([])
ax2 = ax.twinx()
ax2.set_ylabel('1.2" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax = fig.add_subplot(414)
ax.plot(spec[0],trans_2_1_seeing_2_0_sum[:,0],'%s-'%colours[0],label='2.1" length')
ax.plot(spec[0],trans_2_7_seeing_2_0_sum[:,0],'%s--'%colours[0],label='2.7" length')
ax.plot(spec[0],trans_3_3_seeing_2_0_sum[:,0],'%s:'%colours[0],label='3.3" length')
ax2 = ax.twinx()
ax2.set_ylabel('2.0" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


fig = plt.figure()
AX = fig.add_subplot(111)
AX.spines['top'].set_color('none')
AX.spines['bottom'].set_color('none')
AX.spines['left'].set_color('none')
AX.spines['right'].set_color('none')
AX.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
AX.set_xlabel('Wavelength / nm')
AX.set_ylabel('Transmissions')
AX.set_title('Slit 2')
AX.yaxis.set_label_coords(-0.05,0.5)


ax = fig.add_subplot(411)
ax.plot(spec[0],trans_2_1_seeing_0_4_sum[:,1],'%s-'%colours[1],label='2.1" length')
ax.plot(spec[0],trans_2_7_seeing_0_4_sum[:,1],'%s--'%colours[1],label='2.7" length')
ax.plot(spec[0],trans_3_3_seeing_0_4_sum[:,1],'%s:'%colours[1],label='3.3" length')
ax.set_xticklabels([])
ax.legend(loc='best')
ax2 = ax.twinx()
ax2.set_ylabel('0.4" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax = fig.add_subplot(412)
ax.plot(spec[0],trans_2_1_seeing_0_8_sum[:,1],'%s-'%colours[1],label='2.1" length')
ax.plot(spec[0],trans_2_7_seeing_0_8_sum[:,1],'%s--'%colours[1],label='2.7" length')
ax.plot(spec[0],trans_3_3_seeing_0_8_sum[:,1],'%s:'%colours[1],label='3.3" length')
ax.set_xticklabels([])
ax2 = ax.twinx()
ax2.set_ylabel('0.8" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax = fig.add_subplot(413)
ax.plot(spec[0],trans_2_1_seeing_1_2_sum[:,1],'%s-'%colours[1],label='2.1" length')
ax.plot(spec[0],trans_2_7_seeing_1_2_sum[:,1],'%s--'%colours[1],label='2.7" length')
ax.plot(spec[0],trans_3_3_seeing_1_2_sum[:,1],'%s:'%colours[1],label='3.3" length')
ax.set_xticklabels([])
ax2 = ax.twinx()
ax2.set_ylabel('1.2" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax = fig.add_subplot(414)
ax.plot(spec[0],trans_2_1_seeing_2_0_sum[:,1],'%s-'%colours[1],label='2.1" length')
ax.plot(spec[0],trans_2_7_seeing_2_0_sum[:,1],'%s--'%colours[1],label='2.7" length')
ax.plot(spec[0],trans_3_3_seeing_2_0_sum[:,1],'%s:'%colours[1],label='3.3" length')
ax2 = ax.twinx()
ax2.set_ylabel('2.0" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


fig = plt.figure()
AX = fig.add_subplot(111)
AX.spines['top'].set_color('none')
AX.spines['bottom'].set_color('none')
AX.spines['left'].set_color('none')
AX.spines['right'].set_color('none')
AX.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
AX.set_xlabel('Wavelength / nm')
AX.set_ylabel('Transmissions')
AX.set_title('Slit 3')
AX.yaxis.set_label_coords(-0.05,0.5)


ax = fig.add_subplot(411)
ax.plot(spec[0],trans_2_1_seeing_0_4_sum[:,2],'%s-'%colours[2],label='2.1" length')
ax.plot(spec[0],trans_2_7_seeing_0_4_sum[:,2],'%s--'%colours[2],label='2.7" length')
ax.plot(spec[0],trans_3_3_seeing_0_4_sum[:,2],'%s:'%colours[2],label='3.3" length')
ax.legend(loc='best')
ax.set_xticklabels([])
ax2 = ax.twinx()
ax2.set_ylabel('0.4" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


ax = fig.add_subplot(412)
ax.plot(spec[0],trans_2_1_seeing_0_8_sum[:,2],'%s-'%colours[2],label='2.1" length')
ax.plot(spec[0],trans_2_7_seeing_0_8_sum[:,2],'%s--'%colours[2],label='2.7" length')
ax.plot(spec[0],trans_3_3_seeing_0_8_sum[:,2],'%s:'%colours[2],label='3.3" length')
ax.set_xticklabels([])
ax2 = ax.twinx()
ax2.set_ylabel('0.8" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


ax = fig.add_subplot(413)
ax.plot(spec[0],trans_2_1_seeing_1_2_sum[:,2],'%s-'%colours[2],label='2.1" length')
ax.plot(spec[0],trans_2_7_seeing_1_2_sum[:,2],'%s--'%colours[2],label='2.7" length')
ax.plot(spec[0],trans_3_3_seeing_1_2_sum[:,2],'%s:'%colours[2],label='3.3" length')
ax.set_xticklabels([])
ax2 = ax.twinx()
ax2.set_ylabel('1.2" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


ax = fig.add_subplot(414)
ax.plot(spec[0],trans_2_1_seeing_2_0_sum[:,2],'%s-'%colours[2],label='2.1" length')
ax.plot(spec[0],trans_2_7_seeing_2_0_sum[:,2],'%s--'%colours[2],label='2.7" length')
ax.plot(spec[0],trans_3_3_seeing_2_0_sum[:,2],'%s:'%colours[2],label='3.3" length')
ax2 = ax.twinx()
ax2.set_ylabel('2.0" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


trans_2_1_seeing_0_4_sum_sum = np.sum(trans_2_1_seeing_0_4_sum,axis=1)
trans_2_1_seeing_0_8_sum_sum = np.sum(trans_2_1_seeing_0_8_sum,axis=1)
trans_2_1_seeing_1_2_sum_sum = np.sum(trans_2_1_seeing_1_2_sum,axis=1)
trans_2_1_seeing_2_0_sum_sum = np.sum(trans_2_1_seeing_2_0_sum,axis=1)

trans_2_7_seeing_0_4_sum_sum = np.sum(trans_2_7_seeing_0_4_sum,axis=1)
trans_2_7_seeing_0_8_sum_sum = np.sum(trans_2_7_seeing_0_8_sum,axis=1)
trans_2_7_seeing_1_2_sum_sum = np.sum(trans_2_7_seeing_1_2_sum,axis=1)
trans_2_7_seeing_2_0_sum_sum = np.sum(trans_2_7_seeing_2_0_sum,axis=1)

trans_3_3_seeing_0_4_sum_sum = np.sum(trans_3_3_seeing_0_4_sum,axis=1)
trans_3_3_seeing_0_8_sum_sum = np.sum(trans_3_3_seeing_0_8_sum,axis=1)
trans_3_3_seeing_1_2_sum_sum = np.sum(trans_3_3_seeing_1_2_sum,axis=1)
trans_3_3_seeing_2_0_sum_sum = np.sum(trans_3_3_seeing_2_0_sum,axis=1)

fig = plt.figure()
AX = fig.add_subplot(111)
AX.spines['top'].set_color('none')
AX.spines['bottom'].set_color('none')
AX.spines['left'].set_color('none')
AX.spines['right'].set_color('none')
AX.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
AX.set_xlabel('Wavelength / nm')
AX.set_ylabel('Transmissions')
AX.set_title('Total transmission through slitlets')
AX.yaxis.set_label_coords(-0.05,0.5)

ax = fig.add_subplot(411)
ax.plot(spec[0],trans_2_1_seeing_0_4_sum_sum,'r-',label='2.1" slit length')
ax.plot(spec[0],trans_2_7_seeing_0_4_sum_sum,'b-',label='2.7" slit length')
ax.plot(spec[0],trans_3_3_seeing_0_4_sum_sum,'m-',label='3.3" slit length')
ax.set_xticklabels([])
ax.legend(loc='best')
ax2 = ax.twinx()
ax2.set_ylabel('0.4" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax = fig.add_subplot(412)
ax.plot(spec[0],trans_2_1_seeing_0_8_sum_sum,'r-',label='2.1" slit length')
ax.plot(spec[0],trans_2_7_seeing_0_8_sum_sum,'b-',label='2.7" slit length')
ax.plot(spec[0],trans_3_3_seeing_0_8_sum_sum,'m-',label='3.3" slit length')
ax.set_xticklabels([])
ax2 = ax.twinx()
ax2.set_ylabel('0.8" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax = fig.add_subplot(413)
ax.plot(spec[0],trans_2_1_seeing_1_2_sum_sum,'r-',label='2.1" slit length')
ax.plot(spec[0],trans_2_7_seeing_1_2_sum_sum,'b-',label='2.7" slit length')
ax.plot(spec[0],trans_3_3_seeing_1_2_sum_sum,'m-',label='3.3" slit length')
ax.set_xticklabels([])
ax2 = ax.twinx()
ax2.set_ylabel('1.2" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax = fig.add_subplot(414)
ax.plot(spec[0],trans_2_1_seeing_2_0_sum_sum,'r-',label='2.1" slit length')
ax.plot(spec[0],trans_2_7_seeing_2_0_sum_sum,'b-',label='2.7" slit length')
ax.plot(spec[0],trans_3_3_seeing_2_0_sum_sum,'m-',label='3.3" slit length')
ax2 = ax.twinx()
ax2.set_ylabel('2.0" seeing')
ax2.set_yticklabels([])
ax2.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)





