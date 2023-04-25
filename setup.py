from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name='ucsbsim',
    version='0.1',
    author='MazinLab',
    author_email='mazinlab@ucsb.edu',
    description='The MKID Spectrometer Simulation',
    readme='README.md',
    packages=find_packages(),
    keywords=['mkid', 'spectrometer', 'simulation', 'spectra', 'order sorting'],
    classifiers=[
        "Private :: Do Not Upload",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3"
    ],
    ext_modules=cythonize("sortarray.pyx", "filterphot.pyx"),
    include_dirs=[np.get_include()]
)
