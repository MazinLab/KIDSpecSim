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


lim_mags_snr10_30s = np.array([

        ])


lim_mags_snr10_1hr = np.array([
    20.1849,
    19.5289,	
    21.7753,
    22.4056,
   	22.9601,	
   22.602,
	22.3659,	
    22.7192,	
    22.2993,
    21.3905,	
    20.9306
    ])

xshooter_AB_mags = np.array([19.9,20.7,21.0,21.2,21.4,21.6,21.6,21.7,21.7,21.6,21.5,20.8,20.8,20.8,20.9,20.9,21.0,21.0,
                             21.0,21.0,20.9,20.8,20.8,20.3,19.4,20.3,20.3,20.0,20.4,20.4,20.4,19.0,20.0,20.5,20.5,20.4,
                             18.2,18.5,18.7,17.6]) 
xshooter_AB_wls = np.array([314.15,326.73,341.61,358.82,374.03,390.25,422.94,436.50,471.92,499.25,532.60,567.90,589.70,
                            614.22,630.52,657.75,679.48,712.18,744.88,783.03,821.25,862.18,905.77,957.90,1007.5,1047.8,
                            1091.4,1137.9,1189.4,1246.7,1309.3,1378.4,1453.9,1538.1,1633.5,1742.6,1872.1,2013.6,2179.7,
                            2379.4])



LIRIS_mag_name = ['J','H','K']
LIRIS_mag_wls = np.array([1236,1646,2160])
LIRIS_mags_3600 = np.array([22.3,21.9,21.6]) #for snr 10
LIRIS_mags_1800 = np.array([21.9,21.5,21.2])
LIRIS_mags_600 = np.array([21.3,20.9,20.6])
LIRIS_mags_60 = np.array([20.0,19.6,19.3])

harmoni_mag_name = ['R','H']
harmoni_mag_wls = np.array([677,1760])
harmoni_mags = np.array([22.7,26.2])

FORS_mag_name = ['R']
FORS_mag_wl = np.array([644])
FORS_mags_3600 = np.array([23.3])

blaze_wls = np.array([
    1620.33,	
    1215.25,	
    972.199,	
    810.165,	
    694.428,	
    607.624,	
    540.11	,
    486.099,	
    441.908,	
    405.083,	
    373.923,
    ])


blaze_mags = ['H', 'J', 'I', 'R', 'V', 'V', 'V', 'B', 'B', 'U', 'U']

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

lim_mags_snr10_1hr_AB = AB_converter(lim_mags_snr10_1hr,blaze_mags)
lim_mags_snr10_30s_AB = AB_converter(lim_mags_snr10_30s,blaze_mags)

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

plt.plot(xshooter_AB_wls,xshooter_AB_mags,'--',color='orange',label='X-Shooter 3600s SNR>10')
plt.plot(xshooter_AB_wls,xshooter_AB_mags,'s',color='orange',markersize=10)

plt.plot(harmoni_mag_wls,harmoni_mags,'--',color='turquoise',label='HARMONI 900s SNR>5')
plt.plot(harmoni_mag_wls,harmoni_mags,'s',color='turquoise',markersize=10)

plt.plot(blaze_wls,lim_mags_snr10_1hr_AB,'r--',label='KIDSpec 3600s SNR>10')
plt.plot(blaze_wls,lim_mags_snr10_1hr_AB,'ro',markersize=10)

plt.plot(blaze_wls,lim_mags_snr10_30s_AB,'g--',label='KIDSpec 30s SNR>10')
plt.plot(blaze_wls,lim_mags_snr10_30s_AB,'go',markersize=10)


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
plt.plot(LIRIS_mag_wls,LIRIS_mags_3600_AB,'P',color='gold',markersize=10,label='LIRIS 3600s SNR>10')

#plt.plot(FORS_mag_wl,FORS_mags_3600_AB,'--',color='lime')
plt.plot(FORS_mag_wl,FORS_mags_3600_AB,'X',color='lime',markersize=10,label='FORS 3600s SNR>5')


plt.xlabel('Wavelength / nm')
plt.xlim(250,1800)
plt.ylabel('AB magnitude')
#plt.legend(loc='best')#bbox_to_anchor=(0.9, 0.4),fontsize=20)#0.87
#plt.legend(loc='lower right')
#plt.tight_layout()
ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])


#plt.savefig('lim_mags_10062021',dpi=1000)

























