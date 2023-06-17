# The MKID Spectrometer Simulation Series

### Introduction

This package comprises three main scripts (and their supporting modules) that may act in series or separately:
1) script_simulate.py reduces a model spectrum to a photon table similar to how the MKID spectrometer will.
In the command line arguments, essentially all spectrograph properties may be tailored.
The option to generate a calibration spectrum is much like an on-sky source minus the atmospheric and telescopic
bandpasses.
2) script_msf.py loads either a real or simulated calibration photon table (specific file syntax is used) and extracts
the MKID Spread Function (MSF) from it via least-squares fitting. The intersections between these fitted Gaussian
functions defines the bin edges for the orders. The covariance matrix defines what fraction of each 
order Gaussian may be part of another order, and thus will be used to generate the error band on an extracted spectrum.
3) script_extract.py loads either a real or simulated observation photon table and applies
the MSF products to it to extract an output spectrum which has been order-sorted, unblazed, and has an error band
on the counts.

### Tutorial on running a sample simulation to extraction sequence:

Note: an MKIDPipeline enviroment is required to run these files without issue.

After cloning the repository, the Cython files need to be compiled.
In a command line terminal in the folder where KIDSpecSim lives, run the following lines: 

`conda activate pipeline`
`cd KIDSpecSim/ucsbsim/`
`python ../setup.py build_ext --inplace`

This sets up the Cython files needed to run some of the modules. 

#### Spectrum simulation steps:
Now that everything is ready, run the following calibration settings with desired path and R0s file name:

`python script_simulate.py 'path/to/outdir' 'path/to/R0s_file.csv' 'blackbody' --filter_bandpass --littrow`

Note: If there is no R0s_file, one will be generated mid-script (it's not a problem to not have one!), 
so ensure proper syntax is used if you have one.
This creates a blackbody calibration spectrum. A plot will show and save to file so you can 
verify that the simulation is functioning properly.

Now, run the following observation settings with desired path and R0s file name:

`python script_simulate.py 'path/to/outdir' 'path/to/R0s_file.csv' 'phoenix' --atmo_bandpass --filter_bandpass --tele_bandpass --littrow`

This creates a Phoenix model observation spectrum.

#### MKID Spread Function steps:
Run the following MSF extraction settings, making sure to use the **calibration** file generated above:

`python script_msf.py 'path/to/outdir/calibration_file.h5' 'path/to/R0s_file.csv'`

This generates the MSF products for use in spectrum extraction.
A few comprehensive plots will show and save so you can verify the goodness of fit.


## NOT FINISHED FROM THIS POINT ON, DO NOT RUN (above ok)
#### Spectrum extraction steps:
Run the following extraction settings:

`python script_extract.py`

Save and run file. This generates the extracted spectrum with error band.

### Changing settings according to preference:

Different types of spectra can be simulated, but it is important to note that the 
`'type_of_spectra'` must match when generating and extracting the calibration spectrum. 
And of course, there must be a photon table file of some spectra in order for that to be extracted. 
The key is basically to ensure all files are created/present before running next steps. 
Files have naming schemes and directories that should be followed strictly.
Currently, this package only supports Phoenix, blackbody, and 'delta' model spectra.