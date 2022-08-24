# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:23:30 2021

@author: BVH
"""

import numpy as np
import scipy.stats


opt_crrates = np.array([
                    0.03870081, 	
                    1.075634, 	
                    0.1198905, 	
                    0.1061971, 	
                    0.08764023, 	
                    0.1082206 ,	
                    0.07220233, 	
                    1.501013 ,	
                    0.09278395 	,
                    0.08388422 	,
                    0.09021577 	,
                    0.06510416 	,
                    0.08617714 	,
                    0.0701182 	,
                    0.4581404 	,
                    0.1053156 	,
                    0.1961472 	,
                    5.588108 	,
                    0.08805755 ,
                    0.08126175 	]) #/cm2/s

ir_crrates = np.array([
                    18.11494,
                    81.12204,
                    7.956841,
                    6.176841,
                    11.25759,
                    18.78581,
                    3.122371,
                    146.3522,
                    4.276586,
                    11.23395,
                    7.967706,
                    12.3264,
                    11.75307,
                    11.66424,
                    45.74466,
                    9.880028,
                    1.190748,
                    243.5622,
                    6.7758039,
                    21.08962]) #/cm2/s

opt_crrate_mean = np.mean(opt_crrates)
opt_crrate_std = np.std(opt_crrates)
opt_crrate_median = np.median(opt_crrates)
opt_crrate_med_dev = scipy.stats.median_abs_deviation(opt_crrates)

ir_crrate_mean = np.mean(ir_crrates)
ir_crrate_std = np.std(ir_crrates)
ir_crrate_median = np.median(ir_crrates)
ir_crrate_med_dev = scipy.stats.median_abs_deviation(ir_crrates)

print('Mean OPT CRRATE:',opt_crrate_mean,'+/-',opt_crrate_std)
print('Median OPT CRRATE:',opt_crrate_median,'+/-',opt_crrate_med_dev)


print('Mean IR CRRATE:',ir_crrate_mean,'+/-',ir_crrate_std)
print('Median IR CRRATE:',ir_crrate_median,'+/-',ir_crrate_med_dev)

opt_pix_size = 15 #um for x-shooter
ir_pix_size = 18 #um for x-shooter
#1 cm2 = 1e8 um2
opt_pix_num = 8388608
ir_pix_num = 4000000
pix_scale_fac = 4

opt_crrate_mean_um2 = opt_crrate_mean/1e8
ir_crrate_mean_um2 = ir_crrate_mean/1e8 #/um2 /s

opt_crrate_mean_um2_1pix = opt_crrate_mean_um2*(opt_pix_size**2)
ir_crrate_mean_um2_1pix = ir_crrate_mean_um2*(ir_pix_size**2) #/s on one pixel









