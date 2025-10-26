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
        xStd = np.zeros(2*m+1, dtype='f4') + np.std(x);  yStd = np.zeros(2*m+1, dtype='f4') + np.std(y)
        return corr, xStd, yStd
    
    xn = x[k-m : l-(k-m)]; yn = y[k-m : l-(k-m)]; l = len(xn)   # Remove unneeded elements
    
    # Normalize the arrays
    xInStd = np.std(xn); yInStd = np.std(yn)
    xn = (xn - np.mean(xn)) / xInStd
    yn = (yn - np.mean(yn)) / yInStd
    #xn = x
    #yn = y
    
    # Cumulative sums (for window mean & std)
    zero = np.float32(0)
    xSum = np.append(zero, np.cumsum(xn))        # zero appended to enable summation from start
    ySum = np.append(zero, np.cumsum(yn))
    x2Sum = np.append(zero, np.cumsum(xn * xn))
    y2Sum = np.append(zero, np.cumsum(yn * yn))
    
    corr = np.zeros((2*m+1, 2*m+1), dtype='f4')
    
    # Calculate the zero-lag correlation
    xLagV = np.arange(2*m+1)
    xySum = np.append(zero, np.cumsum(xn * yn))   # comulative cross corr
    xWinS = xSum[xLagV+n] - xSum[xLagV]           # x sum within window
    yWinS = ySum[xLagV+n] - ySum[xLagV]           # y sum within window
    xStd  = np.sqrt( (x2Sum[xLagV+n] - x2Sum[xLagV])/n - (xWinS/n)**2 )[::-1]   # x STD within window
    yStd  = np.sqrt( (y2Sum[xLagV+n] - y2Sum[xLagV])/n - (yWinS/n)**2 )[::-1]   # y STD within window
    corr[(xLagV,xLagV)] = ((xySum[xLagV+n] - xySum[xLagV]) - xWinS * yWinS / n)[::-1]   # correlation in n-elements window
    
    # Calculate positive & negative delta-lag correlations
    xySum = np.zeros_like(xn)
    dLagV = np.arange(1,2*m + 1)            # delta-lag Vector (yLag-xLag)
    for dLag in dLagV:
        yLagV = np.arange(dLag,2*m+1); xLagV = np.arange(2*m+1-dLag)
        #xySum = np.append(zero, np.cumsum(xn[dLag:] * yn[:l-dLag]))      # comulative cross corr
        xySum[1:l-dLag+1] = np.cumsum(xn[dLag:] * yn[:l-dLag])            # comulative cross corr
        corr[(xLagV,yLagV)] = ((xySum[xLagV+n] - xySum[xLagV]) - xWinS[xLagV+dLag] * yWinS[xLagV] / n)[::-1]  # correlation in n-elements window
        
        #xLagV = np.arange(dLag,2*m+1); yLagV = np.arange(2*m+1-dLag)
        #xySum = np.append(zero, np.cumsum(xn[:l-dLag] * yn[dLag:]))      # comulative cross corr
        xySum[1:l-dLag+1] = np.cumsum(xn[:l-dLag] * yn[dLag:])            # comulative cross corr
        corr[(yLagV,xLagV)] = ((xySum[xLagV+n] - xySum[xLagV]) - xWinS[xLagV] * yWinS[xLagV+dLag] / n)[::-1]  # correlation in n-elements window

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
    zero = np.float32(0)
    xSum = np.cumsum(xn)
    ySum = np.cumsum(yn)
    x2Sum = np.append(zero, np.cumsum(xn * xn))   # zero appended to enable summation from start
    y2Sum = np.append(zero, np.cumsum(yn * yn))
    
    corr = np.zeros(2 * m + 1,dtype='f4'); denom = np.zeros_like(corr)
    
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
    
    corr = np.zeros(2 * m + 1, dtype='f4'); denom = np.zeros_like(corr)
    
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


def calcWeights(f, snr=None, w=None):
    """
    Compute the weights of a multi-order spectrum.
    Parameters:
        f (list of arrays): List of signal arrays (one array per order).
        snr (np.ndarray or None): Signal-to-noise ratio values per order.
        w (np.ndarray or None): Predefined weights per order.
    Returns:
        np.ndarray: Computed weights.
    """
    weights = np.array([np.var(x) * len(x) for x in f]) # Partial weight (default): var(f[i]) * len(f[i])
    if snr is not None:
        weights *= (snr * snr)                          # Likelihood weight: snr[i]^2*var(f[i])*len(f[i])
    elif w is not None:
        weights = w                                     # User-defined weights
    return weights


def ccf1d(f, t, m, snr=None, w=None):
    """
    Compute the cross-correlation function, including support for a multi-order spectrum and template.
    The per-order CCFs of multi-order inputs are weighted-averaged using weight[i]=snr[i]^2*var(f[i])*len(f[i]) of each order
    Parameters:
        f (single, or list of, np.ndarray): Observed spectrum - single or multi-order.
        t (single, or list of, np.ndarray): Template - single or multi-order.
        m (int): The maximum lag to consider in both directions.
        snr (np.ndarray or None): Signal-to-noise ratio (same length as f & t) - Used for weighting multi-order CCFs
        w (np.ndarray or None): User defined multi-order weights (same length as f & t).
    Returns:
        np.ndarray: (Combined for multi-order) Cross Correlation Function.
    """
    if isinstance(f, np.ndarray) and isinstance(t, np.ndarray):
        return genNormCorr(f, t, m)

    elif isinstance(f, list) and isinstance(t, list):
        if len(f) != len(t):
            raise ValueError("If f and t are lists (multi-order), they must have the same length.")

        # Compute the CCFs and weights of all orders
        CCFMat = np.array([genNormCorr(f[i], t[i], m) for i in range(len(f))])
        weights = calcWeights(f, snr, w)

        # Combined CCF
        comCCF = np.average(CCFMat, weights=weights, axis=0)
        return comCCF
    else:
        raise TypeError("f and t must both be either numpy arrays or multi-order lists of equal length.")


def todcor(obs=None, t1=None, t2=None, m=None, alpha=None, ccfInput=None, outAll=False):
    """
    The exact TODCOR algorithm (including fixes to the original TODCOR) to find the best radial-velocity shifts for a binary star system.

    Parameters:
    obs (np.ndarray): The observed spectrum of the binary star system.
    t1 (np.ndarray): The template spectrum of the first star.
    t2 (np.ndarray): The template spectrum of the second star.
    m (int): The maximum lag to consider in both directions.
    obs, t1, t2 & m are optional, as they are used only if ccfInput is None.
    alpha (float): The flux ratio of the two components (to be normalized).
    If alpha==None, the optimal positive alpha (highest CCF), per matrix element, is derived and used.
    ccfInput (np.ndarray): An optional structured array with precomputed fields (ccf1, ccf2, ccf12, std12)
                           of shape (2*m+1, 2*m+1). If provided, obs, t1, t2 & m are not needed,
                           as the cross-correlation components are extracted from it, skipping their derivation.
    outAll (bool): If True, also returns a structured numpy array (shape (2*m+1, 2*m+1)) with fields
                   'ccf1', 'ccf2', 'ccf12' and 'std12'.
    
    To derive the Exact TodCor result, inputs should fulfill: len(t1)=len(t2)=len(obs)+2*m
    Otherwise, the regular TodCor result is returned if: len(t1)=len(t2) >= len(obs)

    Returns:
    np.ndarray: A 2D array of cross-correlation values.
    np.ndarray: A 2D array of optimal alpha(s1_index, s2_index)
    Optionally, np.ndarray: A structured numpy array with ccf & std matrices if outAll is True.
    """
    if ccfInput is None:
        l = len(t1); n = len(obs)
        if l != len(t2):
            raise ValueError("The two template arrays must have the same length.")
        if n > l:
            raise ValueError("The obs array cannot be longer than the template arrays.")
        if (alpha is not None) and not (np.isfinite(alpha) and (alpha>=0)):
            alpha = None
            warnings.warn("alpha must be a finite positive. Switched to alpha-fitting mode")
        M = 2*m + 1
        # Calculate the 1D cross-correlation for each template with the observed spectrum
        ccf1V = genNormCorr(obs, t1, m)                     # General Normalized-Correlation array
        ccf2V = genNormCorr(obs, t2, m)                     # General Normalized-Correlation array
        ccf12, std1, std2 = winNormCorr(t1, t2, m, n)       # Windowed Normalized-Correlation matrix & STD arrays
        ccf1 = np.tile(ccf1V, (M,1)).T;  ccf2 = np.tile(ccf2V, (M,1)) # ccf1 & ccf2 matrices
        std12 = std2[None,:] / std1[:,None]                 # std-ratio matrix
    else:                                                   # Use the structured array input
        ccf1,ccf2,ccf12,std12 = ccfInput['ccf1'],ccfInput['ccf2'],ccfInput['ccf12'],ccfInput['std12']

    if alpha is None:                                   # The extreme-point normalized alpha matrix
        alphaM = ( (ccf1 * ccf12 - ccf2) / (ccf2 * ccf12 - ccf1) ).clip(min=0)
    else:
        alphaM = alpha * std12                          # Normalized alpha matrix
    
    # The TodCor matrix
    corrM = ((ccf1 + alphaM * ccf2) / np.sqrt(1.0 + 2.0 * alphaM * ccf12 + alphaM**2))
    
    if alpha is None:
        hiC1 = (ccf1 > corrM)                           # Fix ccf1 > corrM elements
        alphaM[hiC1] = 0;         corrM[hiC1] = ccf1[hiC1]
        hiC2 = (ccf2 > corrM)                           # Fix ccf2 > corrM elements
        alphaM[hiC2] = largeNum;  corrM[hiC2] = ccf2[hiC2]

    alphaM /= std12                                     # Convert back to alpha matrix

    if outAll:                                          # return the CCFs and STDs as a structured array
        ccfOut = np.empty(ccf1.shape, dtype=[('ccf1','f4'),('ccf2','f4'),('ccf12','f4'),('std12','f4')])
        ccfOut['ccf1'],ccfOut['ccf2'],ccfOut['ccf12'],ccfOut['std12'] = ccf1,ccf2,ccf12,std12
        return corrM, alphaM, ccfOut
    else:
        return corrM, alphaM


def todcorVel(M, Ns, dv=1.0, rad=1):
    """
    Estimate sub-pixel TODCOR-peak velocities and uncertainties via 2D quadratic fit.

    Parameters:
    M (np.ndarray): 2D TODCOR matrix.
    Ns (int): Spectrum length (for error estimates)
    dv (float): Velocity step per pixel.
    rad (int): fit-patch radius

    Returns:
    vel (np.ndarray): [v1, v2] Peak sub-pixel velocities coordinates.
    val (float): Interpolated TODCOR peak (CCF) value.
    sig (np.ndarray): [σ_v1, σ_v2] estimated peak velocities uncertainties.
    """
    shape = np.array(M.shape)
    center = shape // 2
    idx = np.array(np.unravel_index(np.argmax(M), shape))

    # Handle edge case: max on border
    if np.any((idx < rad) | (idx >= shape - rad)):
        vel = (idx - center) * dv
        return vel, M[tuple(idx)], np.full(2, np.nan)

    Nv = 2 * rad + 1
    vVec = np.arange(-rad, rad + 1)

    # Allocate data [0] and v1 [1], v2 [2] coordinates grids
    # Grid shape=(kind, v1, v2)
    grid = np.empty((3,Nv, Nv), dtype=np.float64)

    # Fill data and coordinates vectorially
    grid[0] = M[idx[0]-rad:idx[0]+rad+1, idx[1]-rad:idx[1]+rad+1]
    grid[1] = vVec[:,None]    # v1 grid
    grid[2] = vVec[None,:]    # v2 grid

    # Build design matrix for 2nd-degree 2D polynomial and least-squares fitting
    gridFlat = grid.reshape(3,-1)
    powerVec = lambda x, y: np.array([x*x, y*y, x*y,
                                      x, y, np.ones_like(x)])
    A = powerVec(gridFlat[1], gridFlat[2]).T
    cf, *_ = np.linalg.lstsq(A, gridFlat[0], rcond=None)

    # Hessian and gradient
    H = np.array([[2*cf[0], cf[2]],
                  [cf[2], 2*cf[1]]])
    grad = np.array([cf[3], cf[4]])

    # Solve for peak offset (in pixels)
    try:
        delta = -np.linalg.solve(H, grad)
    except np.linalg.LinAlgError:
        delta = np.zeros(2)
    #dv1, dv2 = delta

    # TODCOR value at sub-pixel maximum
    val = powerVec(delta[0],delta[1]) @ cf

    # Not needed: Coefficients covariance using residual variance and A
    #res = gridFlat[0] - A @ cf
    #sigma2 = np.var(res, ddof=A.shape[1])
    #cfCov = sigma2 * np.linalg.inv(A.T @ A)

    try:
        # Peak v1, v2, alpha coordinates covariance matrix and uncertainties sigma
        maxCov = (1 - min(val,0.999999)**2)/Ns/val * -np.linalg.inv(H)
        sig = np.sqrt(np.diag(maxCov)) * dv
    except np.linalg.LinAlgError:
        sig = np.full(2, np.nan)

    # Final result: shifts from center + sub-pixel offsets, scaled to km/s
    vel = (idx - center + delta) * dv
    return vel, val, sig


def todcorVelAlpha(obs=None, t1=None, t2=None, m=None, ccfInput=None, Ns=None, dAlpha=0.01, alphaRad=1, dv=1.0, rad=1):
    """
    Estimate sub-pixel TODCOR-peak coordinates using 3D quadratic fit in (v1, v2, alpha) space.

    Parameters:
    obs (np.ndarray): The observed spectrum of the binary star system.
    t1 (np.ndarray): The template spectrum of the first star.
    t2 (np.ndarray): The template spectrum of the second star.
    m (int): The maximum lag to consider in both directions.
    obs, t1, t2 & m are optional, as they are used only if ccfInput is None.
    alpha (float): The flux ratio of the two components (to be normalized).
    If alpha==None, the optimal positive alpha (highest CCF), per matrix element, is derived and used.
    ccfInput (np.ndarray): An optional structured array with precomputed fields (ccf1, ccf2, ccf12, std12)
                           of shape (2*m+1, 2*m+1). If provided, obs, t1, t2 & m are not needed,
                           as the cross-correlation components are extracted from it, skipping their derivation.
    Ns (int): Spectrum length (for error estimates). Should be provided if ccfInput=ccfs
    dAlpha (float): alpha step size
    alphaRad (int): radius of alpha grid (in pixels)
    dv (float): velocity step per pixel (km/s)
    rad (int): radius of v1,v2 patch (in pixels)

    Returns:
    vel (np.ndarray): [v1, v2] in km/s
    alpha_est (float): sub-pixel alpha estimate
    val (float): sub-pixel TODCOR maximum value
    sig (np.ndarray): [σ_v1, σ_v2, σ_alpha] uncertainties
    ccfs (np.ndarray): A structured numpy array with ccf & std matrices if outAll is True.
    """
    if ccfInput is None:
        Ns = obs.size                           # spectrum length
        M, alphaM, ccfs = todcor(obs, t1, t2, m, outAll=True)
    else:
        M, alphaM = todcor(ccfInput=ccfs)
    shape = np.array(M.shape)
    center = shape // 2
    idx = np.array(np.unravel_index(np.argmax(M), shape))
    bestAlpha = alphaM[tuple(idx)]                     # Optimized alpha at todocor maximum

    # Handle edge case: max on border
    if np.any((idx < rad) | (idx >= shape - rad)):
        vel = (idx - center) * dv
        return vel, bestAlpha, M[tuple(idx)], np.full(3, np.nan)

    # Prepare alpha range and patch size
    Na = 2 * alphaRad + 1
    Nv = 2 * rad + 1
    aVec = np.arange(-alphaRad, alphaRad + 1)
    vVec = np.arange(-rad, rad + 1)

    # Allocate data [0] and v1 [1], v2 [2], alpha [3] coordinates grids
    # Grid shape=(kind, v1, v2, alpha)
    grid = np.empty((4,Nv, Nv, Na), dtype=np.float64)

    # Fill coordinates vectorially
    grid[1] = vVec[:,None,None]    # v1 grid
    grid[2] = vVec[None,:,None]    # v2 grid
    grid[3] = aVec[None,None,:]    # alpha grid

    # Evaluate TODCOR at each alpha and save into the data grid
    for i, alpha in enumerate(aVec):
        M, _ = todcor(ccfInput=ccfs, alpha=bestAlpha + alpha * dAlpha)
        grid[0,:,:,i] = M[idx[0] - rad:idx[0] + rad + 1, idx[1] - rad:idx[1] + rad + 1]

    # Build design matrix for 2nd-degree 3D polynomial and least-squares fitting
    gridFlat = grid.reshape(4,-1)
    powerVec = lambda x, y, z: np.array([x*x, y*y, z*z, x*y, x*z, y*z,
                                         x, y, z, np.ones_like(x)])
    A = powerVec(gridFlat[1], gridFlat[2], gridFlat[3]).T
    cf, *_ = np.linalg.lstsq(A, gridFlat[0], rcond=None)

    # Hessian and gradient
    H = np.array([[2*cf[0], cf[3], cf[4]],
                  [cf[3], 2*cf[1], cf[5]],
                  [cf[4], cf[5], 2*cf[2]]])
    grad = np.array([cf[6], cf[7], cf[8]])

    # Solve for peak offset (in pixels)
    try:
        delta = -np.linalg.solve(H, grad)
    except np.linalg.LinAlgError:
        delta = np.zeros(3)

    # TODCOR value at sub-pixel maximum
    val = powerVec(delta[0], delta[1], delta[2]) @ cf

    # Not needed: Coefficients covariance using residual variance and A
    #res = gridFlat[0] - A @ cf
    #sigma2 = np.var(res, ddof=A.shape[1])
    #cfCov = sigma2 * np.linalg.inv(A.T @ A)

    try:
        # Peak v1, v2, alpha coordinates covariance matrix and uncertainties sigma
        maxCov = (1 - min(val,0.99999)**2)/Ns/val * -np.linalg.inv(H)
        sig = np.sqrt(np.diag(maxCov)) * np.array([dv, dv, dAlpha])
    except np.linalg.LinAlgError:
        sig = np.full(3, np.nan)

    # Final result: shifts from center + sub-pixel offsets, scaled to km/s
    vel = (idx - center + delta[:2]) * dv
    alpha_est = bestAlpha + delta[2] * dAlpha  # Sub-pixel alpha estimate
    return vel, alpha_est, val, sig, ccfs
