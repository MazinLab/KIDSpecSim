Run setup.py to compile C code.

Run main.py with blackbody spectrum and generate_R0=True to create calibration data.

Run MSF.py to fit for Gaussian parameters and generate covariance matrices.

Run main.py with desired initial parameters for observation.

Use cal_bins.csv (bin edges) and cov_matrix#.csv from MSF.py to bin and plot resulting photon table.
