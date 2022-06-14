# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:01:39 2019

@author: BVH
"""

import numpy as np
import os

'''
This file stores all the parameters for the simulation.
For all file names don't forget to include the path if the file is not in 
the same directory as the SIM scripts.
'''

def str_2_bool(val):
    if val == 'True':
        return True
    elif val == 'False':
        return False
    else:
        raise ValueError("Accepted inputs are True or False")
        

parameter_file = 'KSIM_INPUT_PARAMETERS.txt'

file_param = open('SETUP_KSIM/KSIM_INPUT_PARAMETERS.txt','r')

params = file_param.readlines()

file_param.close()

folder = params[19].split(' ')[2]

if os.path.isdir(folder) == False:
    os.mkdir(folder)

object_name = params[0].split(' ')[2]
object_file = params[1].split(' ')[2]
object_x = params[2].split(' ')[2:5]#what will be the xlabel of the spectrum graph
object_y = params[3].split(' ')[2:5] #what will be the ylabel of the spectrum graph

object_x = object_x[0] + ' ' + object_x[1] + ' ' + object_x[2]
object_y = object_y[0] + ' ' + object_y[1] + ' ' + object_y[2]


binstep = float(params[4].split(' ')[2]) * 1e-7
mirr_diam = float(params[5].split(' ')[2]) #diameter of the VLT primary mirror(in cm)
seeing = float(params[6].split(' ')[2]) #again working number, will depend on the telescope being used
h = float(params[7].split(' ')[2]) # in ergs/s
c = float(params[8].split(' ')[2]) #in cm/s
exposure_t = float(params[9].split(' ')[2])

fov = 30.0*60.0
area = (0.5*mirr_diam)**2.

tele_file = params[10].split(' ')[2]

lambda_low_val = float(params[11].split(' ')[2]) #in nm
lambda_high_val = float(params[12].split(' ')[2]) #in nm
n_pixels = int(params[13].split(' ')[2])
alpha_val = float(params[14].split(' ')[2]) # Angle of incidence in degrees
phi_val = float(params[15].split(' ')[2])  # Grating blaze angle in degrees
refl_deg_opt = float(params[16].split(' ')[2]) 
refl_start_val = alpha_val - refl_deg_opt # Diffraction angles in degrees
refl_end_val = alpha_val + refl_deg_opt    # Beta start and stop NEED NOT be symmetric around alpha in degrees
OPT_grooves = float(params[17].split(' ')[2])
norders = float(params[18].split(' ')[2])
OPT_a = 1000. / OPT_grooves #distance between grooves in micrometers
ER_band_low = float(params[20].split(' ')[2]) #energy resolution of MKID pixels for low wavelength
ER_band_wl = float(params[21].split(' ')[2]) 

raw_sky_file = params[22].split(' ')[2] #sky photon counts

slit_width = float(params[23].split(' ')[2])
pix_fov = float(params[24].split(' ')[2]) #pixel fov

eff_models = ['Gaussian','Sinc','Schroeder','Casini'] #grating efficiency model names

off_centre = float(params[25].split(' ')[2]) #how much the object is not in the centre, between 0 and 1
airmass = float(params[26].split(' ')[2])

IR_grooves = float(params[29].split(' ')[2])
IR_a = 1000. / IR_grooves #distance between grooves in micrometers
IR_alpha = float(params[30].split(' ')[2])
IR_phi = float(params[31].split(' ')[2])
refl_deg_ir = float(params[32].split(' ')[2]) 
refl_start_val_ir = IR_alpha - refl_deg_ir #6.0 # Diffraction angles in degrees
refl_end_val_ir = IR_alpha + refl_deg_ir #6.0  # Beta start and stop NEED NOT be symmetric around alpha in degrees

dead_pixel_perc = float(params[31].split(' ')[2])
r_e_spread = str_2_bool(params[32].split(' ')[2])

IR_arm = str_2_bool(params[33].split(' ')[2])
cutoff = float(params[34].split(' ')[2])
TLUSTY = str_2_bool(params[35].split(' ')[2])
object_file_1 = params[36].split(' ')[2]
object_file_2 = params[37].split(' ')[2] 
row = int(params[38].split(' ')[2])
redshift = float(params[39].split(' ')[2])
redshift_orig = float(params[40].split(' ')[2])
mag_reduce = float(params[41].split(' ')[2])
sky = str_2_bool(params[42].split(' ')[2])
delete_folders = str_2_bool(params[43].split(' ')[2])
gen_sky_seeing_eff = str_2_bool(params[44].split(' ')[2])
sky_seeing_eff_file_save_or_load = params[45].split(' ')[2]
gen_model_seeing_eff = str_2_bool(params[46].split(' ')[2])
model_seeing_eff_file_save_or_load = params[47].split(' ')[2]
extra_plots = str_2_bool(params[48].split(' ')[2])
stand_star_factors_run = str_2_bool(params[49].split(' ')[2])
stand_star_filename_detail = params[50].split(' ')[2]
supported_file_extraction = str_2_bool(params[51].split(' ')[2])
fwhm_fitter = str_2_bool(params[52].split(' ')[2])
cen_wl = float(params[53].split(' ')[2])
cont_rem_poly = str_2_bool(params[54].split(' ')[2])
reset_R_Es = str_2_bool(params[55].split(' ')[2])
reset_dead_pixels = str_2_bool(params[56].split(' ')[2])
reg_grid_factor = float(params[57].split(' ')[2])

pix_mult = np.ndarray.astype((np.linspace(1,n_pixels,10)),dtype='int') #for progress updates during sections of the code involving looping through pixels
 
seeing_str = str(seeing)[0] + '_' + str(seeing)[2]
#refl_start_val = alpha_val - ((n_pixels*pix_fov)/3600) # Diffraction angles in degrees
#refl_end_val = alpha_val + ((n_pixels*pix_fov)/3600) 

slit_R_factor = slit_width/pix_fov


