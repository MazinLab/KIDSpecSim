# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:21:08 2021

@author: BVH
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 28})


def AB_converter(lim_mags,mags):
    AB_mags = np.zeros(len(lim_mags))
    
    for i in range(len(lim_mags)):
        if mags[i] == 'U':
            AB_mags[i] += lim_mags[i] + 0.79
        if mags[i] == 'B':
            AB_mags[i] += lim_mags[i] - 0.09
        if mags[i] == 'V':
            AB_mags[i] += lim_mags[i] + 0.02
        if mags[i] == 'R':
            AB_mags[i] += lim_mags[i] + 0.21
        if mags[i] == 'I':
            AB_mags[i] += lim_mags[i] + 0.45
        if mags[i] == 'Z':
            AB_mags[i] += lim_mags[i] + 0.54
        if mags[i] == 'Y':
            AB_mags[i] += lim_mags[i] + 0
        if mags[i] == 'J':
            AB_mags[i] += lim_mags[i] + 0.91
        if mags[i] == 'H':
            AB_mags[i] += lim_mags[i] + 1.39
        if mags[i] == 'K':
            AB_mags[i] += lim_mags[i] + 1.85
    
    return AB_mags


'''
lim_mags_no_rebin = np.array([17.480,18.481,18.303,20.185,20.101,20.563,21.060,20.173,21.559,21.842,21.549,21.429,
                              22.019,21.821,21.746])
#these both are 3600s
lim_mags_with_rebin = np.array([18.028,18.994,18.994,20.507,20.566,20.850,21.213,20.850,22.143,22.232,22.232,22.054,
                                22.710,22.497,22.175])

lim_mags_no_rebin_1800 = np.array([17.160,18.090,17.661,19.751,19.241,20.172,20.386,19.689,20.916,21.194,21.038,21.038,
                                   21.575,21.443,21.443])

lim_mags_with_rebin_1800 = np.array([17.504,18.919,18.500,20.012,20.300,19.369,21.079,19.046,21.570,22.065,21.841,21.841,
                                     22.013,22.013,21.897])

lim_mags_no_rebin_600 = np.array([16.206,16.647,16.914,18.659,18.445,19.144,19.493,18.996,20.088,20.583,20.398,20.175,
                                  20.699,20.542,20.542])

lim_mags_with_rebin_600 = np.array([16.620,17.621,17.621,19.192,19.364,18.826,20.200,18.299,20.877,21.001,20.879,20.813,
                                  21.400,21.239,21.239])

lim_mags_no_rebin_60 = np.array([14.121,14.190,14.820,16.164,16.000,16.902,17.400,16.902,17.657,18.153,17.819,17.521,
                                 18.335,18.335,18.176])

lim_mags_with_rebin_60 = np.array([14.687,15.274,15.594,16.687,17.073,17.357,18.025,16.902,18.530,18.822,18.822,18.822,
                                   19.478,19.227,18.597])
'''

mag_names = [
    'U', 
    'B', 
    'V' ,
    'R' ,
    'I' , 
    'Z' , 
    'Y' ,
    'J' , 
    'H' 
    ]

#mag_wls = np.array([357,416,525,596,801,904,1018,1240,1660])


lim_mags_5000_snr10_30s = np.array([
        16.265,
        17.6491,
        18.4291,
        18.5547,
        18.4844,
        17.9366,
        16.8493
        ])

'''
lim_mags_5000_snr5_1hr = np.array([
    20.0741,
    22.1604,
    23.3298,
    23.2383,
    23.5653,
    23.0925,
    22.4372
    ])
'''
lim_mags_5000_snr5_1hr = np.array([
    20.1536,
    22.1604,
    23.3298,
    23.2383,
    23.5653,
    23.2313,
    22.4372
    ])
lim_mags_5000_snr10_1hr = np.array([
        19.41358102   ,
        21.418181110975922,        
        22.41427743   ,
        22.27796265   ,
        22.7465211    ,
        22.52092282  ,
        21.18446778   
    ])

xshooter_AB_mags = np.array([19.9,20.7,21.0,21.2,21.4,21.6,21.6,21.7,21.7,21.6,21.5,20.8,20.8,20.8,20.9,20.9,21.0,21.0,
                             21.0,21.0,20.9,20.8,20.8,20.3,19.4,20.3,20.3,20.0,20.4,20.4,20.4,19.0,20.0,20.5,20.5,20.4,
                             18.2,18.5,18.7,17.6]) 
xshooter_AB_wls = np.array([314.15,326.73,341.61,358.82,374.03,390.25,422.94,436.50,471.92,499.25,532.60,567.90,589.70,
                            614.22,630.52,657.75,679.48,712.18,744.88,783.03,821.25,862.18,905.77,957.90,1007.5,1047.8,
                            1091.4,1137.9,1189.4,1246.7,1309.3,1378.4,1453.9,1538.1,1633.5,1742.6,1872.1,2013.6,2179.7,
                            2379.4])

'''
lim_mags_no_rebin_3600_5 = np.array([18.306,19.387,18.924,20.858,20.832,21.307,21.804,21.007,22.389,22.785,22.415,22.553,22.914,
                                     22.729,22.729])
lim_mags_with_rebin_3600_5 = np.array([18.663,20.124,19.653,21.217,21.509,20.589,22.171,20.183,22.848,23.344,23.138,22.976,23.441,
                                       23.331,23.071])
'''

LIRIS_mag_name = ['J','H','K']
LIRIS_mag_wls = np.array([1236,1646,2160])
LIRIS_mags_3600 = np.array([22.3,21.9,21.6]) #for snr 10
LIRIS_mags_1800 = np.array([21.9,21.5,21.2])
LIRIS_mags_600 = np.array([21.3,20.9,20.6])
LIRIS_mags_60 = np.array([20.0,19.6,19.3])

FORS_mag_name = ['R']
FORS_mag_wl = np.array([644])
FORS_mags_3600 = np.array([23.3])

blaze_wls = np.array([
  1568.90063388    ,1045.93372259   ,784.45031695   ,627.56022355   ,522.96691129   ,448.25730253   ,392.22518347   ])


blaze_mags = ['H', 'Y', 'R', 'V', 'B', 'B', 'U', 'U']

'''
lim_mags_no_rebin_AB = AB_converter(lim_mags_no_rebin,blaze_mags)
lim_mags_with_rebin_AB = AB_converter(lim_mags_with_rebin,blaze_mags)
lim_mags_no_rebin_AB_1800 = AB_converter(lim_mags_no_rebin_1800,blaze_mags)
lim_mags_with_rebin_AB_1800 = AB_converter(lim_mags_with_rebin_1800,blaze_mags)
lim_mags_no_rebin_AB_600 = AB_converter(lim_mags_no_rebin_600,blaze_mags)
lim_mags_with_rebin_AB_600 = AB_converter(lim_mags_with_rebin_600,blaze_mags)
lim_mags_no_rebin_AB_60 = AB_converter(lim_mags_no_rebin_60,blaze_mags)
lim_mags_with_rebin_AB_60 = AB_converter(lim_mags_with_rebin_60,blaze_mags)
'''

lim_mags_5000_snr10_1hr_AB = AB_converter(lim_mags_5000_snr10_1hr,blaze_mags)
lim_mags_5000_snr5_1hr_AB = AB_converter(lim_mags_5000_snr5_1hr,blaze_mags)
lim_mags_5000_snr10_30s_AB = AB_converter(lim_mags_5000_snr10_30s,blaze_mags)
#lim_mags_8500_snr5_30s_AB = AB_converter(lim_mags_8500_snr5_30s,blaze_mags)

LIRIS_mags_3600_AB = AB_converter(LIRIS_mags_3600,LIRIS_mag_name)
LIRIS_mags_1800_AB = AB_converter(LIRIS_mags_1800,LIRIS_mag_name)
LIRIS_mags_600_AB = AB_converter(LIRIS_mags_600,LIRIS_mag_name)
LIRIS_mags_60_AB = AB_converter(LIRIS_mags_60,LIRIS_mag_name)

FORS_mags_3600_AB = AB_converter(FORS_mags_3600,FORS_mag_name)

'''
lim_mags_no_rebin_AB_3600_5 = AB_converter(lim_mags_no_rebin_3600_5,blaze_mags)
lim_mags_with_rebin_AB_3600_5 = AB_converter(lim_mags_with_rebin_3600_5,blaze_mags)
'''
'''
plt.figure()
#plt.plot(blaze_wls[:6],lim_mags_no_rebin_AB[:6],'r-')
plt.plot(blaze_wls[:6],lim_mags_no_rebin_AB[:6],'ro',markersize=10,label='IR arm 1x2 binning')
#plt.plot(blaze_wls[:6],lim_mags_with_rebin_AB[:6],'r-')
plt.plot(blaze_wls[:6],lim_mags_with_rebin_AB[:6],'r^',markersize=10,label='IR arm 1x4 binning')
#plt.plot(blaze_wls[7:],lim_mags_no_rebin_AB[7:],'b-')
plt.plot(blaze_wls[7:],lim_mags_no_rebin_AB[7:],'bo',markersize=10,label='OPT arm 1x2 binning')
#plt.plot(blaze_wls[7:],lim_mags_with_rebin_AB[7:],'b-')
plt.plot(blaze_wls[7:],lim_mags_with_rebin_AB[7:],'b^',markersize=10,label='OPT arm 1x4 binning')
plt.xlabel('Wavelength / nm')
plt.ylabel('AB magnitude')
plt.legend(loc='best')
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])
#plt.grid()
'''

plt.figure()

plt.plot(xshooter_AB_wls,xshooter_AB_mags,'--',color='orange',label='X-Shooter 3600s')
plt.plot(xshooter_AB_wls,xshooter_AB_mags,'s',color='orange',markersize=10)


plt.plot(blaze_wls,lim_mags_5000_snr10_1hr_AB,'r--',label='KIDSpec 3600s')
plt.plot(blaze_wls,lim_mags_5000_snr10_1hr_AB,'ro',markersize=10)

plt.plot(blaze_wls,lim_mags_5000_snr5_1hr_AB,'b--',label='KIDSpec 3600s')
plt.plot(blaze_wls,lim_mags_5000_snr5_1hr_AB,'bd',markersize=10)

plt.plot(blaze_wls,lim_mags_5000_snr10_30s_AB,'g--',label='KIDSpec 30s')
plt.plot(blaze_wls,lim_mags_5000_snr10_30s_AB,'go',markersize=10)

#plt.plot(mag_wls,lim_mags_8500_snr5_30s_AB,'m--',label='KIDSpec 30s SNR>5')
#plt.plot(mag_wls,lim_mags_8500_snr5_30s_AB,'mo',markersize=10)

'''
plt.plot(blaze_wls[:6],lim_mags_no_rebin_AB[:6],'r--',label='KIDSpec 3600s')
plt.plot(blaze_wls[:6],lim_mags_no_rebin_AB[:6],'ro',markersize=10)
plt.plot(blaze_wls[:6],lim_mags_with_rebin_AB[:6],'r--')
plt.plot(blaze_wls[:6],lim_mags_with_rebin_AB[:6],'r^',markersize=10)
plt.plot(blaze_wls[7:],lim_mags_no_rebin_AB[7:],'r--')
plt.plot(blaze_wls[7:],lim_mags_no_rebin_AB[7:],'ro',markersize=10)
plt.plot(blaze_wls[7:],lim_mags_with_rebin_AB[7:],'r--')
plt.plot(blaze_wls[7:],lim_mags_with_rebin_AB[7:],'r^',markersize=10)

plt.plot(blaze_wls[:6],lim_mags_no_rebin_AB_1800[:6],'b--',label='KIDSpec 1800s')
plt.plot(blaze_wls[:6],lim_mags_no_rebin_AB_1800[:6],'bo',markersize=10)
plt.plot(blaze_wls[:6],lim_mags_with_rebin_AB_1800[:6],'b--')
plt.plot(blaze_wls[:6],lim_mags_with_rebin_AB_1800[:6],'b^',markersize=10)
plt.plot(blaze_wls[7:],lim_mags_no_rebin_AB_1800[7:],'b--')
plt.plot(blaze_wls[7:],lim_mags_no_rebin_AB_1800[7:],'bo',markersize=10)
plt.plot(blaze_wls[7:],lim_mags_with_rebin_AB_1800[7:],'b--')
plt.plot(blaze_wls[7:],lim_mags_with_rebin_AB_1800[7:],'b^',markersize=10)

plt.plot(blaze_wls[:6],lim_mags_no_rebin_AB_600[:6],'g--',label='KIDSpec 600s')
plt.plot(blaze_wls[:6],lim_mags_no_rebin_AB_600[:6],'go',markersize=10)
plt.plot(blaze_wls[:6],lim_mags_with_rebin_AB_600[:6],'g--')
plt.plot(blaze_wls[:6],lim_mags_with_rebin_AB_600[:6],'g^',markersize=10)
plt.plot(blaze_wls[7:],lim_mags_no_rebin_AB_600[7:],'g--')
plt.plot(blaze_wls[7:],lim_mags_no_rebin_AB_600[7:],'go',markersize=10)
plt.plot(blaze_wls[7:],lim_mags_with_rebin_AB_600[7:],'g--')
plt.plot(blaze_wls[7:],lim_mags_with_rebin_AB_600[7:],'g^',markersize=10)

plt.plot(blaze_wls[:6],lim_mags_no_rebin_AB_60[:6],'--',color='fuchsia',label='KIDSpec 60s')
plt.plot(blaze_wls[:6],lim_mags_no_rebin_AB_60[:6],'o',color='fuchsia',markersize=10)
plt.plot(blaze_wls[:6],lim_mags_with_rebin_AB_60[:6],'--',color='fuchsia')
plt.plot(blaze_wls[:6],lim_mags_with_rebin_AB_60[:6],'^',color='fuchsia',markersize=10)
plt.plot(blaze_wls[7:],lim_mags_no_rebin_AB_60[7:],'--',color='fuchsia')
plt.plot(blaze_wls[7:],lim_mags_no_rebin_AB_60[7:],'o',color='fuchsia',markersize=10)
plt.plot(blaze_wls[7:],lim_mags_with_rebin_AB_60[7:],'--',color='fuchsia')
plt.plot(blaze_wls[7:],lim_mags_with_rebin_AB_60[7:],'^',color='fuchsia',markersize=10)

plt.plot(blaze_wls[:6],lim_mags_no_rebin_AB_3600_5[:6],'--',color='darkviolet',label='KIDSpec 3600s SNR>5')
plt.plot(blaze_wls[:6],lim_mags_no_rebin_AB_3600_5[:6],'o',color='darkviolet',markersize=10)
plt.plot(blaze_wls[:6],lim_mags_with_rebin_AB_3600_5[:6],'--',color='darkviolet')
plt.plot(blaze_wls[:6],lim_mags_with_rebin_AB_3600_5[:6],'^',color='darkviolet',markersize=10)
plt.plot(blaze_wls[7:],lim_mags_no_rebin_AB_3600_5[7:],'--',color='darkviolet')
plt.plot(blaze_wls[7:],lim_mags_no_rebin_AB_3600_5[7:],'o',color='darkviolet',markersize=10)
plt.plot(blaze_wls[7:],lim_mags_with_rebin_AB_3600_5[7:],'--',color='darkviolet')
plt.plot(blaze_wls[7:],lim_mags_with_rebin_AB_3600_5[7:],'^',color='darkviolet',markersize=10)
'''

plt.plot(LIRIS_mag_wls,LIRIS_mags_3600_AB,'--',color='gold')
plt.plot(LIRIS_mag_wls,LIRIS_mags_3600_AB,'P',color='gold',markersize=10,label='LIRIS 3600s')

#plt.plot(FORS_mag_wl,FORS_mags_3600_AB,'--',color='lime')
plt.plot(FORS_mag_wl,FORS_mags_3600_AB,'X',color='lime',markersize=10,label='FORS 3600s ')


plt.xlabel('Wavelength / nm')
plt.xlim(250,1800)
plt.ylabel('AB magnitude')
plt.legend(bbox_to_anchor=(0.9, 0.4),fontsize=20)#0.87
#plt.legend(loc='lower right')
#plt.tight_layout()
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])


#plt.savefig('lim_mags_10062021',dpi=1000)

























