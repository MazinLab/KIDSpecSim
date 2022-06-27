# KIDSpec-Simulator-KSIM
Observation simulator for the upcoming KIDSpec instrument.


# KSIM how to use

## 14/06/2022

Please send feedback to `benedict.hofmann@durham.ac.uk`.<br>


## Initialising desired simulation
<br>

When simulating an observation with KSIM all changes can be done within the `SETUP_KSIM/KSIM_INPUT_PARAMETERS.txt` file, which contains all the parameters which can currently be changed (list shown below). <br><br>

|Parameter|Description|
|------|-------|
|object_name| Name/identifier of simulated astronomical object. |
|object_file | File containing spectrum of object, should be layed out in two columns. First column wavelength in nm, and second column flux in  $ergcm^{-2}s^{-1}\AA^{-1}$.  |                         
|object_x | Units of wavelength for plots. |   
|object_y | The same but for flux. |      
|binstep | Bin size of `object_file` wavelength bins in nm |                            
|mirr_diam | Diameter of telescope primary mirror in cm. |        
|seeing | Atmospheric seeing in arcseconds. |         
|h | Plancks constant. | 
|c | Speed of light in cm/s. |
|exposure_t | Exposure time in seconds. |                          
|tele_file | Telescope material reflectivity data filename. |             
|lambda_low_val | Start of KIDSpec bandpass (nm). |          
|lambda_high_val | End of KIDSpec bandpass (nm). |        
|n_pixels | Number of MKIDs desired.   |                 
|alpha_val | Angle of incidence for grating (degrees). |        
|phi_val | Blaze angle for grating (degrees)    |
|opt_refl_deg | Angle of reflection for grating (degrees). |                                                               
|opt_grooves | Grooves/mm. |                                                                     
|norders | Number of orders to test.| 
|folder_dir | Folder directory name for results and other files required for simulation. |                    	          
|fudicial_enrgy_res | Energy resolution of MKIDs at desired fudicial wavelength.|    
|fudicial_wavelength | Fudicial wavelength in nm.| 
|raw_sky_file |  Sky data from ESO SkyModel.|
|slit_width | Slit width in arcseconds.|
|pixel_fov | Pixel FOV in arcseconds.|
|off_centre | How off slit centre the pixel is (arcseconds).|
|ir_grooves | If using two arms then these grating parameters are for the infrared (IR) arm, and previous parameters are for the optical (OPT) arm. Same definitions.|                                       
|dead_pixel_perc | Percentage of MKIDs which are dead.|         
|R_E_spread | Option to activate a 3-sigma spread in energy resolution around the fudicial energy resolution.|       
|IR_arm | Option to activate a second arm.|
|cutoff | Cutoff wavelength (nm) between IR and OPT arms. If using only one arm set this to be the same as `lambda_high_val`|
|TLUSTY | If the input to KSIM will be two spectra (binary simulation) generated by TLUSTY, set this to True.|
|object_file_1 | Filename for one of the two TLUSTY objects.|
|object_file_2 | Filename for one of the two TLUSTY objects.|
|row | Which timestep in the orbit to simulate.|
|redshift | If the spectrum is desired to redshifted, this can be applied here. Default is `0.0`. |
|redshift_orig | The original redshift of the spectrum to be simulated/redshifted.| 
|mag_reduce | Factor to either increase or decrease the magnitude of the spectrum to be simulated. Above 1 will decrease the magnitude.|      
|sky | If set to False sky will not be included in simulation.| 
|delete_folders | Delete folders of byproduct produced by the simulation in process of results.| 
|generate_sky_seeing_eff | If a seeing file for sky does not exist set this to True.|             
|sky_seeing_eff_file_save_or_load | Name of sky seeing file to save or load.|     
|generate_model_seeing_eff | If a seeing file for the object does not exist set this to True.|                             
|model_seeing_eff_file_save_or_load |Name of object seeing file to save or load.|  
|generate_additional_plots | Generates plots detailing more of the steps taken during the simulation, grating efficiencies, photons seen by each MKID, etc.|                     
|generate_standard_star_factors | If standard star factors for desired setup do not exist, simulate the GD71 file included in the KSIM suite and set this to True.|                      
|stand_star_run_filename_details | Name of standard star factors file to save or load.|               
|supported_file_extraction | Leave this on False. |              
|fwhm_fitter | Activates a fitter to find the FWHM of features which have a Lorentzian shape.|                 
|fwhm_fitter_central_wavelength | Location of feature to fit.|     
|continuum_removal_use_polynomial | Fit polynomial to remove continuum.|    
|reset_R_E_spread_array | If a new energy resolution spread is desired set this to True.|               
|reset_dead_pixel_array | If a new dead pixel spread is desired set this to True.|
|reg_grid_factor | For order merging, factor for how large the regular grid should be.|    



## Using KSIM
<br>

KSIM can be run using the `KSIM_V4_0.py` script which will produce plots and a results text file.<br><br>

If the observation parameters are changed beyond the object being 'observed' (such as number of MKIDs) or this is the first run using KSIM, new standard star factors will need to be generated. This can be done by setting `generate_standard_star_factors` to True and simulating the standard star GD71 using the GD71.txt file in `STANDARD_STAR/`. Note the other changes to the parameters file which will be needed such as the correct `folder_dir` input. <br><br>

Once these factors have been generated any other appropriate file can be simulated. If two TLUSTY spectra for a binary simulation in a `.npy` are chosen then this can be entered in the parameters file. Otherwise combine them externally and enter it as a `.txt` file normally in `object_file`.<br><br>






