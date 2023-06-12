# The MKID Spectrometer Simulation Series

### Introduction

This package comprises three main scripts (and their supporting modules) that may act in series or separately:
1) script_simulate.py reduces a model spectrum to a photon table similar to how the MKID spectrometer will.
Inside the script, the spectrum type, photon limit per pixel, and other properties may be tailored.
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
In a command line terminal, run `conda activate pipeline`.

`cd` into `KIDSpecSim/ucsbsim/` and run:

`python ../setup.py build_ext --inplace`

Now that everything is ready, run the following calibration settings:

`python script_simulate.py 'path/to/calibration/output_file.h5' 'path/to/R0s_file.csv' 'blackbody' --littrow`

This creates a blackbody calibration spectrum. A plot will show so you can
verify that the simulation is functioning properly.

Now, run the following observation settings:

`python script_simulate.py 'path/to/observation/output_file.h5' 'path/to/R0s_file.csv' 'phoenix' --atmo_bandpass --filter_bandpass --tele_bandpass --littrow`

This creates a Phoenix model observation spectrum.

## DO NOT RUN FROM THIS POINT ON, NEEDS WORK, above is ok.
Run the following MSF extraction settings:

`type_of_spectra = 'blackbody'`

`plot_fits = False`

Save and run file. This generates the MSF products for use in spectrum extraction.
A few comprehensive plots will show so you can verify the goodness of fit.

Open `script_extract.py` and ensure you have the following extraction settings:

`type_of_spectra = 'phoenix'`

Save and run file. This generates the extracted spectrum with error band.

### Changing settings according to preference:

Different types of spectra can be simulated, but it is important to note that the 
`'type_of_spectra'` must match when generating and extracting the calibration spectrum. 
And of course, there must be a photon table file of some spectra in order for that to be extracted. 
The key is basically to ensure all files are created/present before running next steps. 
Files have naming schemes and directories that should be followed strictly.
Currently, this package only supports Phoenix, blackbody, and 'delta' model spectra.