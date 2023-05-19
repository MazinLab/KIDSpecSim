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

Open `script_simulate.py` and ensure you have the following simulation properties:

`calibration = True`

`full_convol = True`

`type_of_spectra, temp = 'blackbody', 4300`

`pixel_lim = 50000`

Save and run file. This creates a blackbody calibration spectrum. A plot will show so you can
verify that the simulation is functioning properly.

Now, change the simulation settings to reflect:

`calibration = False`

`full_convol = True`

`type_of_spectra, temp = 'phoenix', 4300`

`pixel_lim = 10000`

Save and run file. This creates a Phoenix model observation spectrum.

Open `script_msf.py` and ensure you have the following MSF extraction settings:

`type_of_spectra = 'blackbody'`

`pixel_lim = 50000`

`plot_fits = False`

Save and run file. This generates the MSF products for use in spectrum extraction.
A few comprehensive plots will show so you can verify the goodness of fit.

Open `script_extract.py` and ensure you have the following extraction settings:
#### * work in progress from this point on *
