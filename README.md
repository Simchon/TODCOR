# TODCOR
This is a thin, no installation required, implementation of the Two-dimensional Correlation algorithm (TODCOR) to derive the radial velocities of the two components of a spectrum ([ref](https://ui.adsabs.harvard.edu/abs/1994ApJ...420..806Z/abstract)).

This code implements the TODCOR algorithm with the following improvements:
1. More accurate alpha-fitting mode, including ignoring non-physical negative-alpha solutions and corrections to the (A4) equations in the TODCOR paper ([ref](https://ui.adsabs.harvard.edu/abs/1994ApJ...420..806Z/abstract)).
2. More accurate results for a short spectrum, by better handling of spectrum/templates edge effects.

   
# Content:
**todcor.py** - The TODCOR functions including 1d and 2d correlation functions.

**example.py** - A simple example for a TODCOR analysis of a simulated spectrum generated by combining two shifted templates.

**template_6000K_45_0_6198A-6402A_10K.csv** - An example processed Phoenix template of Teff=6000K, logg=4.5, M/H=0, with resolution R=10000, in the spectral range 6198A - 6402A, with wavelength resolution of 1 km/s.

**template_4500K_45_0_6198A-6402A_10K.csv** - An example processed Phoenix template of Teff=4500K, logg=4.5, M/H=0, with resolution R=10000, in the spectral range 6198A - 6402A, with wavelength resolution of 1 km/s.

**Phoenix_Templates_1d_CCFs.png** - The first plot produced by example.py

**Phoenix_Templates_TODCOR.png** - The second plot produced by example.py

# Acknowledgment:
If you use this code in your research, please acknowledge by citing Zucker, S., & Mazeh, T. 1994, ApJ, 420, 806.

# Installation instructions: 
The repository is compatible with Python 3.7.
**Download or clone** the repository from GitHub. Unpack the zip file in a designated folder. 
