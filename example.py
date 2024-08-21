import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from todcor import *

def exampleRandTemplate(alpha=0.6, rv=[3,-5]):
    # Returns a simulated observed spectrum built by combining two shifted random vectors   
    np.random.seed(42)
    N = 100
    t1 = np.random.rand(N)                                # First template
    t2 = np.random.rand(N)                                # Second template
    obs = np.roll(t1, rv[0]) + alpha * np.roll(t2, rv[1]) # Simulated observed spectrum
    m = 10                                                # CCF range from lag -m to +m
    return obs, t1, t2, rv, alpha, m

def examplePhoenixTemplate(alpha=0.6, rv=[30,-20]):
    # Returns a simulated observed spectrum built by combining two shifted, Phoenix-based, templates
    # Templates flux is normalized and wavelength is at even log steps of 1 km/s
    #df1 = pd.read_csv('template_6000K_45_0_6198A-6402A.csv')
    df1 = pd.read_csv('template_6000K_45_0_6198A-6402A_10K.csv')
    wv1 = df1.values[:,0]; t1 = df1.values[:,1]            # First template
    #df2 = pd.read_csv('template_4500K_45_0_6198A-6402A.csv')
    df2 = pd.read_csv('template_4500K_45_0_6198A-6402A_10K.csv')
    wv2 = df2.values[:,0]; t2 = df2.values[:,1]            # Second template
    if not np.array_equal(wv1,wv2):
        raise ValueError("The two templates must have the same wavelengths array.")    
    obs = np.roll(t1, rv[0]) + alpha * np.roll(t2, rv[1])  # Simulated observed spectrum
    m = 200                                                # CCF range from lag -m to +m
    return obs, t1, t2, rv, alpha, m



# TODCOR Example Code:
# Produce simulated observed spectrum by combining two shifted templates
obs, t1, t2, rv, alpha, m = examplePhoenixTemplate(alpha=0.4)   # Phoenix templates based simulated observed spectrum
#obs, t1, t2, rv, alpha, m = exampleRandTemplate(alpha=0.4)     # Short random templates based simulated observed spectrum

# For best results on a short spectrum, templates should be wider than the spectrum by m elements on each side
#shortSpectrum = False
shortSpectrum = True   # More accurate results on a short spectrum - try exampleRandTemplate()
if shortSpectrum:
    obs = obs[m:-m]
else:
    pass               # Equal-length templates and spectrum. Should be fine if spectrum_length >> m

# Compute the templates 1d CCFs
ccf1 = genNormCorr(obs, t1, m)            # CCF of obs vs. t1
ccf2 = genNormCorr(obs, t2, m)            # CCF of obs vs. t2
ccf12 = genNormCorr(t1, t2, m)            # CCF of t1  vs. t2

# Plot the templates 1d CCFs
plt.figure(1)
ccfX = np.arange(-m, m + 1)
plt.plot(ccfX, ccf1, ccfX, ccf2, ccfX, ccf12); plt.grid(True)
plt.title('Templates 1d CCFs')
plt.legend(['Observed vs. Temp1', 'Observed vs. Temp2', 'Temp1 vs. Temp2'])
plt.xlabel('Lag (km/s)')
plt.ylabel('Correlation')

# Compute the TODCOR correlation matrix, first for finding the best flux ratio (alpha)
corrM1, alphaM = todcor(obs, t1, t2, m)                     # TODCOR alpha-fitting mode

# Find best alpha using the maximum-CCF indices
maxIdx1 = np.unravel_index(np.argmax(corrM1), corrM1.shape) # max TODCOR indices
bestAlpha = alphaM[maxIdx1]                                 # best alpha at max TODCOR indices

# Recalculate TODCOR using the best alpha
corrM, _ = todcor(obs, t1, t2, m, bestAlpha)                # TODCOR with input alpha
maxIdx = np.unravel_index(np.argmax(corrM), corrM.shape)    # max TODCOR indices
maxVal = corrM[maxIdx]                                      # max TODCOR value

# Plot the resulting best-alpha TODCOR matrix as an image
plt.figure(2)
im1 = plt.imshow(corrM, extent=[-m-0.5, m+0.5, -m-0.5, m+0.5], origin='lower', aspect='auto', cmap='viridis')
line = np.arange(-m-0.5,m+0.55,0.1); point = line * 0
plt.plot(line,point+maxIdx[0]-m,'-r', alpha=0.25)
plt.plot(point+maxIdx[1]-m, line,'-r', alpha=0.25)
plt.colorbar(im1, label='Correlation')
plt.xlabel('Lag 2 (km/s)')
plt.ylabel('Lag 1 (km/s)')
plt.title('TODCOR:  CCF[%d,%d]=%1.3f  Best_Alpha=%1.3f'%(maxIdx[0]-m, maxIdx[1]-m, maxVal, bestAlpha))

plt.show()

