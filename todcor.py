import numpy as np
import warnings

largeNum = 1e+6

def winNormCorr(x, y, m, n=None):
    """
    Calculate the normalized-correlation matrix of two equal-length arrays, within a partial window n, for lags from -m to +m.
    At lag=0, the correlation is calculated for the n center elements of the two arrays
    Parameters:
    x (np.ndarray): The first input array.
    y (np.ndarray): The second input array.
    m (int): The maximum lag to consider in both directions.
    n (int): The correlation window length (By default, n = len(x)-2*m)

    Returns:
    np.ndarray[2*m+1,2*m+1]: correlation matrix Cij, for xLag=i & yLag=j
    np.ndarray[2*m+1] : x STD array
    np.ndarray[2*m+1] : y STD array
    """
    
    l = len(x)
    if l != len(y):
        raise ValueError("The input arrays must have the same length.")
    
    if n is None: n = l-2*m
    
    if (l - n) % 2 != 0:
        warnings.warn("The difference in length between the arrays and window is odd. Subtracting 1 from n.")
        n -= 1

    k = (l - n) // 2             # Number of extra arrays elements, in each side, relative to window
    
    if k<m:                      # Window length n is too large. Switching to 1D correlation.
        warnings.warn("Only k=%d extra elements in each side of x & y (l=%d), outside the window n=%d, where m=%d are needed. Switching to 1D correlation"%(k,l,n,m))
        ccf12 = exactNormCorr(x, y, 2*m)                # 1D normalized cross correlation
        lagV = np.arange(-m,m+1)
        ccfIdx = (lagV[None,:] - lagV[:,None]) + 2*m    # s2-s1 index of ccf12
        corr = ccf12[ccfIdx]                            # Fill the correlation matrix
        xStd = np.zeros(2*m+1) + np.std(x);  yStd = np.zeros(2*m+1) + np.std(y)
        return corr, xStd, yStd
    
    xn = x[k-m : l-(k-m)]; yn = y[k-m : l-(k-m)]        # Remove unneeded elements
    
    # Normalize the arrays
    xInStd = np.std(xn); yInStd = np.std(yn)
    xn = (xn - np.mean(xn)) / xInStd
    yn = (yn - np.mean(yn)) / yInStd
    #xn = x
    #yn = y
    
    # Cumulative sums (for window mean & std)
    xSum = np.append(0, np.cumsum(xn))        # zero appended to enable summation from start
    ySum = np.append(0, np.cumsum(yn))
    x2Sum = np.append(0, np.cumsum(xn * xn))
    y2Sum = np.append(0, np.cumsum(yn * yn))
    
    corr = np.zeros((2*m+1, 2*m+1))
    
    # Calculate the zero-lag correlation
    xLagV = np.arange(2*m+1)
    xySum = np.append(0, np.cumsum(xn * yn))      # comulative cross corr
    xWinS = xSum[xLagV+n] - xSum[xLagV]           # x sum within window
    yWinS = ySum[xLagV+n] - ySum[xLagV]           # y sum within window
    xStd  = np.sqrt( (x2Sum[xLagV+n] - x2Sum[xLagV])/n - (xWinS/n)**2 )[::-1]   # x STD within window
    yStd  = np.sqrt( (y2Sum[xLagV+n] - y2Sum[xLagV])/n - (yWinS/n)**2 )[::-1]   # y STD within window
    corr[(xLagV,xLagV)] = ((xySum[xLagV+n] - xySum[xLagV]) - xWinS * yWinS / n)[::-1]   # correlation in n-elements window
    
    # Calculate positive & negative delta-lag correlations
    dLagV = np.arange(1,2*m + 1)            # delta-lag Vector (yLag-xLag)
    for dLag in dLagV:
        yLagV = np.arange(dLag,2*m+1); xLagV = np.arange(2*m+1-dLag)
        xySum = np.append(0, np.cumsum(xn[dLag:] * yn[:l-dLag]))      # comulative cross corr
        corr[(xLagV,yLagV)] = ((xySum[xLagV+n] - xySum[xLagV]) - xWinS[xLagV+dLag] * yWinS[xLagV] / n)[::-1]  # correlation in n-elements window
        
        xLagV = np.arange(dLag,2*m+1); yLagV = np.arange(2*m+1-dLag)
        xySum = np.append(0, np.cumsum(xn[:l-dLag] * yn[dLag:]))      # comulative cross corr
        corr[(xLagV,yLagV)] = ((xySum[yLagV+n] - xySum[yLagV]) - xWinS[yLagV] * yWinS[yLagV+dLag] / n)[::-1]  # correlation in n-elements window        

    # Normalize by the denominator: N*xStd*yStd
    corr /= (n * xStd[:,None] * yStd[None,:])
    
    xStd *= xInStd;  yStd *= yInStd    # normalize by the input STDs
    
    return corr, xStd, yStd


def genNormCorr(x, y, m):
    """
    Calculate the general exact normalized correlation of two arrays over a range of lags from -m to +m.
    y is expected to be longer than x by 2*k (ideally k==m, but any k is supported).
    lag=0 is defined as when x overlaps the center elements of y.
    Parameters:
    x (np.ndarray): The first input array.
    y (np.ndarray): The second input array.
    m (int): The maximum lag to consider in both directions.

    Returns:
    np.ndarray: An array of correlations, normalized by the overlap length and std, for lags from -m to +m.
    """
    n = len(x); l = len(y)
    swaped = False
    xn = x; yn = y
    if l==n:
        return exactNormCorr(x, y, m)
    elif l<n:                         # swap the arrays if len(y)<len(x)
        xn = y; yn = x
        n = len(xn); l = len(yn)
        swaped = True
    
    if (l - n) % 2 != 0:
        warnings.warn("The difference in length between y and x is odd. Adding one element with value of mean(y) to y.")
        yn = np.append(yn, np.mean(yn))
        l = len(yn)

    k = (l - n) // 2                   # number of extra elements, in each side, in y relative to x
    
    # Normalize the arrays
    xn = (xn - np.mean(xn)) / np.std(xn)
    yn = (yn - np.mean(yn)) / np.std(yn)
    #xn = x
    #yn = y
    
    # Cumulative sums (for overlap mean & std)
    xSum = np.cumsum(xn)
    ySum = np.cumsum(yn)
    x2Sum = np.append(0, np.cumsum(xn * xn))   # zero appended to enable summation from start
    y2Sum = np.append(0, np.cumsum(yn * yn))
    
    corr = np.zeros(2 * m + 1); denom = np.zeros(2 * m + 1)
    
    # Calculate the zero-lag correlation
    corr[m] = np.sum(xn * yn[k:-k]);
    denom[m] = np.sqrt(n * ((y2Sum[-k-1]-y2Sum[k]) - (ySum[-k-1]-ySum[k-1])**2 / n))
    
    # Calculate positive and negative lag correlations
    lagV = np.arange(1, m + 1)            # lag Vector
    lenV = n-np.clip(lagV-k,0,None)       # overlap-length Vector
    for lag in lagV:
        corr[m + lag] = np.sum(xn[max(lag-k,0):] * (yn[max(k-lag,0):-k-lag]-(ySum[-k-lag-1]-ySum[max(k-lag-1,-1)])/(n-max(lag-k,0))))
        corr[m - lag] = np.sum((xn[:min(n+k-lag,n)]-xSum[min(n+k-lag,n)-1]/(n-max(lag-k,0))) * yn[lag+k:min(l+lag-k,l)])

    # Positive & negative lag denominators: sqrt(N*Var1 * N*Var2) = N*Std1*Std2
    denom[m+1:] = np.sqrt(((x2Sum[-1]-x2Sum[np.clip(lagV-k,0,None)]) - (xSum[-1]-xSum[np.clip(lagV-k-1,-1,None)])**2 / lenV) * \
                          (y2Sum[-k-lagV-1]-y2Sum[np.clip(k-lagV,0,None)] - (ySum[-k-lagV-1]-ySum[np.clip(k-lagV-1,-1,None)])**2 / lenV))
    denom[:m] = np.sqrt(((y2Sum[np.clip(lagV-k-1,None,-1)]-y2Sum[lagV+k]) - (ySum[np.clip(lagV-k-1,None,-1)]-ySum[lagV+k-1])**2 / lenV) * \
                        (x2Sum[np.clip(k-lagV-1,None,-1)] - xSum[np.clip(k-lagV-1,None,-1)]**2 / lenV))[::-1]

    # Normalize by the denominator: N*Std1*Std2
    corr /= denom
    
    outCorr = corr[::-1] if swaped else corr
    return outCorr


def exactNormCorr(x, y, m):
    """
    Calculate the exact normalized correlation of two same-length arrays over a range of lags from -m to +m.

    Parameters:
    x (np.ndarray): The first input array.
    y (np.ndarray): The second input array.
    m (int): The maximum lag to consider in both directions.

    Returns:
    np.ndarray: An array of correlations, normalized by the overlap length and std, for lags from -m to +m.
    """
    if len(x) != len(y):
        raise ValueError("The input arrays must have the same length.")
    
    # Normalize the arrays
    xn = (x - np.mean(x)) / np.std(x)
    yn = (y - np.mean(y)) / np.std(y)
    #xn = x
    #yn = y
    n = len(x)
    
    # Cumulative sums (for overlap mean & std)
    xSum = np.cumsum(xn)
    ySum = np.cumsum(yn)
    x2Sum = np.cumsum(xn * xn)
    y2Sum = np.cumsum(yn * yn)
    
    corr = np.zeros(2 * m + 1); denom = np.zeros(2 * m + 1)
    
    # Calculate the zero-lag correlation
    #corr[m] = np.sum((xn - xSum[-1]/n) * yn);  denom[m] = n * np.std(xn) * np.std(yn)
    corr[m] = np.sum(xn * yn);  denom[m] = n
    
    # Calculate positive and negative lag correlations
    for lag in range(1, m + 1):
        corr[m + lag] = np.sum(xn[lag:] * (yn[:-lag]-ySum[-lag-1]/(n-lag)))
        corr[m - lag] = np.sum((xn[:-lag]-xSum[-lag-1]/(n-lag)) * yn[lag:])

    # Positive & negative lag denominators: sqrt(N*Var1 * N*Var2) = N*Std1*Std2
    lagV = np.arange(1, m + 1)            # lag vector
    denom[m+1:] = np.sqrt(((x2Sum[-1]-x2Sum[lagV-1]) - (xSum[-1]-xSum[lagV-1])**2 / (n-lagV)) * (y2Sum[-lagV-1] - ySum[-lagV-1]**2 / (n-lagV)))
    denom[:m] = np.sqrt(((y2Sum[-1]-y2Sum[lagV-1]) - (ySum[-1]-ySum[lagV-1])**2 / (n-lagV)) * (x2Sum[-lagV-1] - xSum[-lagV-1]**2 / (n-lagV)))[::-1]

    # Normalize by the denominator: N*Std1*Std2
    corr /= denom

    return corr


def todcor(obs, t1, t2, m, alpha=None):
    """
    The exact TODCOR algorithm (including fixes to the original TODCOR) to find the best radial-velocity shifts for a binary star system.

    Parameters:
    obs (np.ndarray): The observed spectrum of the binary star system.
    t1 (np.ndarray): The template spectrum of the first star.
    t2 (np.ndarray): The template spectrum of the second star.
    m (int): The maximum lag to consider in both directions.
    alpha (float): The flux ratio of the two components (to be normalized).
    If alpha==None, the optimal positive alpha (highest CCF), per matrix element, is derived and used.
    
    To derive the Exact TodCor result, inputs should fulfill: len(t1)=len(t2)=len(obs)+2*m
    Otherwise, the regular TodCor result is returned if: len(t1)=len(t2) >= len(obs)

    Returns:
    np.ndarray: A 2D array of cross-correlation values.
    np.ndarray: A 2D array of optimal alpha(s1_index, s2_index)
    """
    l = len(t1); n = len(obs)
    if l != len(t2):
        raise ValueError("The two template arrays must have the same length.")
    if n > l:
        raise ValueError("The obs array cannot be longer than the template arrays.")
    if (alpha is not None) and not (np.isfinite(alpha) and (alpha>0)):
        alpha = None
        warnings.warn("alpha must be a finite positive. Switched to alpha-fitting mode")
    
    # Calculate the 1D cross-correlation for each template with the observed spectrum
    ccf1V = genNormCorr(obs, t1, m)                     # General Normalized-Correlation array
    ccf2V = genNormCorr(obs, t2, m)                     # General Normalized-Correlation array
    ccf12, std1, std2 = winNormCorr(t1, t2, m, n)       # Windowed Normalized-Correlation matrix & STD arrays
    ccf1 = ccf1V[:,None] + np.zeros(ccf2V.size)[None,:] # ccf1 matrix
    ccf2 = ccf2V[None,:] + np.zeros(ccf1V.size)[:,None] # ccf2 matrix
    
    stdM = std2[None,:] / std1[:,None]                  # std matrix
    if alpha is None:                                   # The extreme-point normalized alpha matrix
        alphaM = ( (ccf1 * ccf12 - ccf2) / (ccf2 * ccf12 - ccf1) ).clip(min=0)
    else:
        alphaM = alpha * stdM                           # Normalized alpha matrix
    
    # The TodCor matrix
    corrM = ((ccf1 + alphaM * ccf2) / np.sqrt(1.0 + 2.0 * alphaM * ccf12 + alphaM**2))
    
    if alpha is None:
        hiC1 = (ccf1 > corrM)                           # Fix ccf1 > corrM elements
        alphaM[hiC1] = 0;         corrM[hiC1] = ccf1[hiC1]
        hiC2 = (ccf2 > corrM)                           # Fix ccf2 > corrM elements
        alphaM[hiC2] = largeNum;  corrM[hiC2] = ccf2[hiC2]

    alphaM /= stdM                                 # Convert back to alpha matrix

    return corrM, alphaM
