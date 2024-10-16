# The MKID Spectrometer Simulation and Pipeline
### Introduction

This package comprises 2 scripts:
1) simulate.py reduces a model spectrum to a photon table in a simulated optical path.
In the command line arguments, essentially all spectrograph and observation properties may be tailored.

2) mkidspec.py loads 3 photon tables, one each of a flat-field or known blackbody, emission lamp,
   and observed star.
   
     a. MSF: The flat-field or known blackbody photon table is separated into pixels and is formatted
        as histograms. Gaussian models are fit to the orders. The virtual pixel boundaries are extracted
        from these. The covariance among orders is calculated. These various properties make up the
        MKID Spread Function.
   
     b. Order-sorting: The other 2 photon tables (emission lamp and observed star) are sorted using the
        MSF from the previous step.
   
     c. Wavecal: The sorted emission lamp spectrum is compared to its atlas to determine the wavecal solution.
   
     d. Extraction: The observed star is updated with the wavecal solution and extraction is complete.

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
Now that everything is ready, run the following to obtain a flat-field photon table:

`python scripts/simulate.py --type_spectra flat`

Run the following to obtain a HgAr lamp photon table:

`python scripts/simulate.py --type_spectra emission -ef mkidspec/linelists/hgar.csv`

Run the following to obtain a Phoenix star model photon table:

`python scripts/simulate.py --type_spectra phoenix --on_sky`

      Optional: writing a .txt file containing command line arguments exactly as shown is 
      another option if you expect the need to edit many properties. For example, run:

      python scripts/simulate.py --option_file path/to/option_file.txt

      where option_file.txt contains:

      --type_spectra phoenix

      --on_sky

Move all generated .h5 files (their file locations are printed in the command line output) into `ucsbsim/mkidspec/testfiles`.
(These test files are too large to be uploaded to Git.)

Now to recover the Phoenix star spectrum, run:

`python scripts/mkidspec.py --plot`

You may omit the '--plot' if you do not wish to view intermediate plots.


   #### Help: 
      
      Run
      
      python scripts/simulate.py --help
      
      or
      
      python scripts/mkidspec.py --help
      
      for the script arguments to see the different configurations and
      how to skip certain steps.
