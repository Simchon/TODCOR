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
    dRV = 1.0
    return obs, t1, t2, rv, alpha, m, dRV

def examplePhoenixTemplate(alpha=0.6, rv=[30,-20], maxRV=200):
    """
    Returns a simulated observed spectrum built by combining two shifted, Phoenix-based, templates.
    Templates flux is normalized and wavelength is at even log steps

    Parameters:
    alpha (float): The flux ratio of the two templates.
    rv [float,float]: The RV shifts of the two templates (km/s)
    maxRV (float): The maximum RV shift to be considered by TODCOR (km/s)

    Returns:
    obs (float np array): The simulated observed spectrum
    t1 (float np array): The primary template
    t1 (float np array): The secondary template
    rv [float,float]: The RV shifts of the two templates (km/s)
    alpha (float): The flux ratio of the two templates
    m (int): The maximum lag to consider in each direction
    dRV (float): The obs, t1 and t2 RV resolution (km/s)
    """
    C = 299792.458                                         # speed of light (km/s)
    #df1 = pd.read_csv('template_6000K_45_0_6198A-6402A.csv')
    df1 = pd.read_csv('template_6000K_45_0_6198A-6402A_10K.csv')
    wv1 = df1.values[:,0]; t1 = df1.values[:,1]            # First template
    #df2 = pd.read_csv('template_4500K_45_0_6198A-6402A.csv')
    df2 = pd.read_csv('template_4500K_45_0_6198A-6402A_10K.csv')
    wv2 = df2.values[:,0]; t2 = df2.values[:,1]            # Second template

    if not np.array_equal(wv1,wv2):
        raise ValueError("The two templates must have the same wavelengths array.")
    diffVec = np.diff(np.log(wv1))
    meanLogDiff = diffVec.mean()                           # Mean resolution of log(wv1)
    if diffVec.std()/meanLogDiff > 0.01:
        warnings.warn("The templates wavelength array should be of even log steps, but they are probably not - Please Check")
    dRV = (np.exp(meanLogDiff)-1)*C                        # The templates RV resolution (km/s)
    m = round(maxRV/dRV)                                   # CCF range from lag -m to +m

    obs = np.roll(t1, round(rv[0]/dRV)) + alpha * np.roll(t2, round(rv[1]/dRV))  # Simulated observed spectrum
    return obs, t1, t2, rv, alpha, m, dRV



# TODCOR Example Code:

# Produce simulated observed spectrum by combining two shifted templates
templateType = 'Phoenix'
#templateType = 'Random'
if templateType == 'Phoenix':
    obs, t1, t2, rv, alpha, m, dRV = examplePhoenixTemplate(alpha=0.4)   # Phoenix templates based simulated observed spectrum
elif templateType == 'Random':
    obs, t1, t2, rv, alpha, m, dRV = exampleRandTemplate(alpha=0.4)     # Short random templates based simulated observed spectrum
else:
    raise ValueError('Unknown templateType: %s' % templateType)
maxRV = m * dRV        # Maximum RV shift to be considered by TODCOR (km/s)

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
ccfX = np.arange(-m, m + 1) * dRV
plt.plot(ccfX, ccf1, ccfX, ccf2, ccfX, ccf12); plt.grid(True)
plt.title('%s-Templates 1d CCFs' % templateType)
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
bestRV = (np.array(maxIdx)-m) * dRV                         # The best RV shifts of the two templates
maxVal = corrM[maxIdx]                                      # max TODCOR value

# Plot the best-alpha TODCOR matrix as an image + cuts at maximum correlation
#plt.figure(2)
fig, ((ax, ax1),(ax2, axn)) = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'height_ratios': [3, 1], 'width_ratios': [3, 1]}, num=2)
im1 = ax.imshow(corrM, extent=np.array([-m-0.5, m+0.5, -m-0.5, m+0.5])*dRV, origin='lower', aspect='auto', cmap='viridis')
line = np.arange(-m-0.5,m+0.55,0.1)*dRV; point = line * 0
ax.plot(line,point+bestRV[0],'-r', alpha=0.25)
ax.plot(point+bestRV[1], line,'-r', alpha=0.25); #ax.grid(True)
ax.set_xlim([-maxRV,maxRV]); ax.set_ylim([-maxRV,maxRV])
#plt.colorbar(im1, label='Correlation')
ax2.plot(ccfX, corrM[maxIdx[0],:],'r',alpha=0.4); ax2.grid(True)
ax2.set_xlim([-maxRV,maxRV])
ax2.set_ylabel('Correlation')
ax2.legend(['Lag 1 = %1.1f km/s'%bestRV[0]])
ax2.set_xlabel('Lag 2 (km/s)')
ax1.plot(corrM[:,maxIdx[1]], ccfX,'r',alpha=0.4); ax1.grid(True)
ax1.set_ylim([-maxRV,maxRV])
ax1.set_xlabel('Correlation')
ax1.legend(['Lag 2 = %1.1f km/s'%bestRV[1]])
ax.set_ylabel('Lag 1 (km/s)')
ax.set_title('%s-Templates TODCOR:  CCF[%1.1f,%1.1f]=%1.3f  Best_Alpha=%1.3f'%(templateType, bestRV[0], bestRV[1], maxVal, bestAlpha))
axn.axis('off')

plt.show()

