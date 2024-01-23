# The MKID Spectrometer Simulation Series

### Introduction

**New changes in code are not reflected in this guide at the moment.**

This package comprises three main scripts (and their supporting modules) that may act in series or separately:
1) script_simulate.py reduces a model spectrum to a photon table similar to how the MKID spectrometer will.
In the command line arguments, essentially all spectrograph properties may be tailored.
The option to generate a calibration spectrum is much like an on-sky source minus the atmospheric and telescopic
bandpasses.
2) script_msf.py loads either a real or simulated calibration photon table (specific file syntax is used) and extracts
the MKID Spread Function (MSF) from it via non-linear least-squares fitting. The intersections between these fitted
Gaussian functions defines the bin edges for the orders. The covariance matrix defines what fraction of each 
order Gaussian may be part of another order, and thus will be used to generate the error band on an extracted spectrum.
4) script_extract.py loads either a real or simulated observation photon table and applies
the MSF products to it to extract an output spectrum which has been order-sorted, unblazed, and has an error band
on the counts.

### Tutorial on running a sample simulation to extraction sequence:

Note: an MKIDPipeline enviroment is required to run these files without issue.

After cloning the repository, the Cython files need to be compiled.
In a command line terminal in the folder where KIDSpecSim lives, run the following lines: 

`conda activate pipeline`

`cd KIDSpecSim/`

`pip install -e .`

`cd ucsbsim`

This sets up the Cython files needed to use some of the modules. 

#### Spectrum simulation steps:
Now that everything is ready, run the following calibration settings with desired path and R0s file name:

`python script_simulate.py 'path/to/cal_outdir' 'path/to/R0s_file.csv' 'blackbody' -fb`

Note: If there is no R0s_file, one will be created mid-script with the given file name 
(it's not a problem to not have one), so ensure proper syntax is used if you have one.
This creates a blackbody calibration spectrum. A plot will show and save to file so you can 
verify that the simulation is functioning properly.

Now, run the following observation settings with desired path and R0s file name:

`python script_simulate.py 'path/to/obs_outdir' 'path/to/R0s_file.csv' 'phoenix' -ab -fb -tb`

This creates a Phoenix model observation spectrum.

#### MKID Spread Function steps:
Run the following MSF extraction settings, making sure to use the **calibration** file generated above:

`python script_msf.py 'path/to/msf_outdir' 'path/to/cal_outdir/calibration_photontable.h5'`

This generates the MSF products for use in spectrum extraction.
A few comprehensive plots will show and save so you can verify the goodness of fit.

#### Spectrum extraction steps:
Run the following extraction settings:

`python script_extract.py 'path/to/some_outdir' 'path/to/msf_outdir/msf_file.npz' 'path/to/obs_outdir/observation_photontable.h5'`

Save and run file. This generates the extracted spectrum with error band.

### Changing settings according to preference:

Different types of spectra can be simulated, but functional support for anything other than a blackbody or phoenix spectrum is missing.
Spectrograph properties may be easily changed by supplying arguments in the command line. 
Type `python script_simulate.py --help` in the same directory for descriptions.
The key is to ensure all files are created/present before running next steps. 
Some generated files have naming schemes and directories that should be followed strictly.
