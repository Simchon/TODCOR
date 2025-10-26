import numpy as np
import warnings
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import shift
from scipy.ndimage import gaussian_filter1d

from todcor import *

def exampleRandTemplate(alpha=0.6, rv=[3.3,-5.5], tempLen=200, snr=None):
    # Returns a simulated observed spectrum built by combining two shifted random vectors
    smoothSig = 4.                                                                    # Gaussian smooth sigma
    t1 = gaussian_filter1d(np.random.rand(tempLen), sigma=smoothSig).astype('f4')     # First template
    t2 = gaussian_filter1d(np.random.rand(tempLen), sigma=smoothSig).astype('f4')     # Second template
    # Simulated observed spectrum
    obs = shift(t1, rv[0], mode='grid-wrap', order=3) + alpha * shift(t2, rv[1], mode='grid-wrap', order=3)
    if snr and np.isfinite(snr) and snr > 0:
        obs = np.random.poisson(obs * snr**2) / snr**2
    m = 20                                                # CCF range from lag -m to +m
    dRV = 1.0
    return obs, t1, t2, rv, alpha, m, dRV


def examplePhoenixTemplate(alpha=0.6, rv=[30.3,-20.2], maxRV=200, tempLen=None, snr=None):
    """
    Returns a simulated observed spectrum built by combining two shifted, Phoenix-based, templates.
    Templates flux is normalized and wavelength is at even log steps

    Parameters:
    alpha (float): The flux ratio of the two templates.
    rv [float,float]: The RV shifts of the two templates (km/s)
    maxRV (float): The maximum RV shift to be considered by TODCOR (km/s)
    snr: Simulated spectrum Signal To Noise

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
    wv1 = df1.values[:,0]; t1 = df1.values[:,1].astype('f4')*0.1 + 1    # First template
    #df2 = pd.read_csv('template_4500K_45_0_6198A-6402A.csv')
    df2 = pd.read_csv('template_4500K_45_0_6198A-6402A_10K.csv')
    wv2 = df2.values[:,0]; t2 = df2.values[:,1].astype('f4')*0.1 + 1    # Second template

    if type(tempLen)==int and tempLen>0 and tempLen<t1.size:            # shorten the templates
        wv1 = wv1[:tempLen];  t1 = t1[:tempLen]
        wv2 = wv2[:tempLen];  t2 = t2[:tempLen]

    if not np.array_equal(wv1,wv2):
        raise ValueError("The two templates must have the same wavelengths array.")
    diffVec = np.diff(np.log(wv1))
    meanLogDiff = diffVec.mean()                           # Mean resolution of log(wv1)
    if diffVec.std()/meanLogDiff > 0.01:
        warnings.warn("The templates wavelength array should be of even log steps, but they are probably not - Please Check")
    dRV = (np.exp(meanLogDiff)-1)*C                        # The templates RV resolution (km/s)
    m = round(maxRV/dRV)                                   # CCF range from lag -m to +m

    # Simulated observed spectrum
    obs = shift(t1, rv[0]/dRV, mode='grid-wrap', order=3) + alpha * shift(t2, rv[1]/dRV, mode='grid-wrap', order=3)
    obs /= (1 + alpha)                                     # Normalize the simulated spectrum
    if snr and np.isfinite(snr) and snr > 0:
        obs = np.random.poisson(obs * snr**2) / snr**2

    return obs, t1, t2, rv, alpha, m, dRV


def examplePlot(corrM, maxCCF, trueRV, bestRV, rvErr, trueAlpha, bestAlpha, alphaErr=None, dv=1., name=''):
    # Plot the best-alpha TODCOR matrix as an image + cuts at maximum correlation
    m = corrM.shape[0]//2
    maxRV =  m * dRV                                         # Maximum RV shift to be considered by TODCOR (km/s)
    ccfX = np.arange(-m, m + 1) * dRV                        # CCF x axis (RV axis)
    maxIdx = np.unravel_index(np.argmax(corrM), corrM.shape) # max TODCOR indices
    intBestRV = (np.array(maxIdx)-m) * dRV                   # The best integer RV shifts of the two templates
    fig, ((ax, ax1),(ax2, axn)) = plt.subplots(2, 2, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1], 'width_ratios': [3, 1]})
    im1 = ax.imshow(corrM, extent=np.array([-m-0.5, m+0.5, -m-0.5, m+0.5])*dRV, origin='lower', aspect='auto', cmap='viridis')
    line = np.arange(-m-0.5,m+0.55,0.1)*dRV; point = line * 0
    ax.plot(line,point+bestRV[0],'-r', alpha=0.25)
    ax.plot(point+bestRV[1], line,'-r', alpha=0.25); #ax.grid(True)
    ax.set_xlim([-maxRV,maxRV]); ax.set_ylim([-maxRV,maxRV])
    #plt.colorbar(im1, label='Correlation')
    ax2.plot(ccfX, corrM[maxIdx[0],:],'r',alpha=0.4); ax2.grid(True)
    ax2.set_xlim([-maxRV,maxRV])
    ax2.set_ylabel('Correlation')
    ax2.legend(['Lag 1 = %1.0f km/s'%intBestRV[0]])
    ax2.set_xlabel('Lag 2 (km/s)')
    ax1.plot(corrM[:,maxIdx[1]], ccfX,'r',alpha=0.4); ax1.grid(True)
    ax1.set_ylim([-maxRV,maxRV])
    ax1.set_xlabel('Correlation')
    ax1.legend(['Lag 2 = %1.0f km/s'%intBestRV[1]])
    ax.set_ylabel('Lag 1 (km/s)')
    sigErr = (bestRV-trueRV) / rvErr
    if alphaErr:
        aSigErr = (bestAlpha - trueAlpha) / alphaErr if alphaErr else 0.
        alphaStr = fr'${bestAlpha:1.3f} \pm {alphaErr:1.3f} ({aSigErr:+1.1f} \sigma)$'
    else:
        alphaStr = fr'{bestAlpha:1.3f}'
    if len(name) > 0:
        name = name + ' - '
    fig.suptitle(
    fr'{name}{templateType}-Template TODCOR:  CCF['
    fr'${bestRV[0]:1.3f} \pm {rvErr[0]:1.3f} ({sigErr[0]:+1.1f} \sigma), '
    fr'{bestRV[1]:1.3f} \pm {rvErr[1]:1.3f} ({sigErr[1]:+1.1f} \sigma)$]'
    fr'={maxCCF:1.3f}  Best_Alpha=' + alphaStr)
    axn.axis('off')

# TODCOR Example Code:

# Produce simulated observed spectrum by combining two shifted templates
trueAlpha = 0.4
snr = 100
templateType = 'Phoenix'
#templateType = 'Random'
np.random.seed(41)
if templateType == 'Phoenix':
    trueRV = np.array([30.3,-20.2])
    #obs, t1, t2, rv, alpha, m, dRV = examplePhoenixTemplate(alpha=trueAlpha, rv=trueRV, snr=snr, maxRV=125, tempLen=5700)   # Phoenix templates based simulated observed spectrum
    obs, t1, t2, rv, alpha, m, dRV = examplePhoenixTemplate(alpha=trueAlpha, rv=trueRV, snr=snr)   # Phoenix templates based simulated observed spectrum
elif templateType == 'Random':
    trueRV = np.array([3.3,-5.5])
    obs, t1, t2, rv, alpha, m, dRV = exampleRandTemplate(alpha=trueAlpha, rv=trueRV, snr=snr)      # Short random templates based simulated observed spectrum
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
s1 = time.time()
ccf1 = genNormCorr(obs, t1, m)            # CCF of obs vs. t1
print(f"Spectrum/Template length = {obs.size}/{t1.size}, lags = {2*m+1}, genNormCorr runtime = {(time.time()-s1):1.4f}")

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

# TODCOR example 1:
# First derive the best alpha, and then, for that alpha, the velocities and their uncertainties
# (This example does not provide alpha uncertainty)
# Compute the TODCOR correlation matrix, first for finding the best flux ratio (alpha)
s1 = time.time()
corrM1, alphaM, ccfs = todcor(obs, t1, t2, m, outAll=True)  # TODCOR alpha-fitting mode
print(f"Spectrum length = {obs.size}, TODCOR matrix shape = {corrM1.shape}, Alpha-fit TODCOR runtime = {(time.time()-s1):1.4f}")

# Find best alpha using the maximum-CCF indices
maxIdx = np.unravel_index(np.argmax(corrM1), corrM1.shape)  # max TODCOR indices
bestAlpha = alphaM[maxIdx]                                  # best alpha at max TODCOR indices

# Recalculate TODCOR using the best alpha
s1 = time.time()
#corrMo, _ = todcor(obs, t1, t2, m, bestAlpha)               # TODCOR with input alpha
corrM, _ = todcor(alpha = bestAlpha, ccfInput = ccfs)        # Fast TODCOR with input alpha (using the first todcor 1d ccfs)
bestRV, maxVal, rvErr = todcorVel(corrM, obs.size, dv=dRV)   # Sub-pixel best RV shifts, max CCF and RV shifts uncertainties
print(f"Spectrum length = {obs.size}, TODCOR matrix shape = {corrM1.shape}, fixed-alpha TODCOR runtime = {(time.time()-s1):1.4f}")
examplePlot(corrM, maxVal, trueRV, bestRV, rvErr, trueAlpha, bestAlpha, name='Example 1')


# TODCOR example 2:
# Simultanously derives the sub-pixel v1, v2 and alpha and their uncertainties
s1 = time.time()
bestRV3, bestAlpha3, maxVal3, err3, ccfs3 = todcorVelAlpha(obs, t1, t2, m, dv=dRV)
corrM3, _ = todcor(alpha = bestAlpha3, ccfInput = ccfs3)     # Fast TODCOR with input alpha (using the first todcor 1d ccfs)
print(f"Spectrum length = {obs.size}, TODCOR matrix shape = {corrM1.shape}, sub-pixel v1, v2, alpha fit TODCOR runtime = {(time.time()-s1):1.4f}")
examplePlot(corrM3, maxVal3, trueRV, bestRV3, err3[:2], trueAlpha, bestAlpha3, alphaErr=err3[2], name='Example 2')

plt.show()

