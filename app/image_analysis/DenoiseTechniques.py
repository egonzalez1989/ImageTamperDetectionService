from skimage.restoration import (denoise_tv_chambolle, denoise_wavelet)
import numpy as np
import pywt, cv2

# The implementation
'''
    Shrink method can be one of: BayesShrink, VisuShrink, Sure or NeighborVariation
'''
def dwt_denoise(data, wavelet = 'db8', level = 1, mode = 'soft', shrinkage = 'NeighborVariation', color = 0):
    data = data / 255.
    if mode == 'hard':
        denoised = 255 * denoise_wavelet(data, wavelet=wavelet, mode=mode, wavelet_levels=level, multichannel=color)
    if shrinkage in ['BayesShrink', 'VisuShrink']:
        denoised = 255 * denoise_wavelet(data, wavelet=wavelet, mode='soft', method = shrinkage, wavelet_levels=level, multichannel=color)
    elif shrinkage == 'Sure':
        raise ValueError('Not implemented')
    # Local variance denoise
    elif shrinkage == 'NeighborVariation':
        # If a color image is received, denoise for each channel
        if len(data.shape) == 3 and data.shape[-1] == 3:
            data = cv2.split(data)
        # For grayscale images, create a single band
        elif len(data.shape) == 2:
            data = [data]
        # If shape is not (x,y,3), (3,x,y) or (x,y) I don't know what to do
        else:
            raise ValueError('Unknown data format')
        dband = []
        for band in data:
            dwt = pywt.wavedec2(band, wavelet = wavelet, level = level)
            dwt2 = dwt_neighvar_shrink(dwt, W = [3,5,7,9])
            dband.append(pywt.waverec2(dwt2, wavelet = wavelet))

        if len(data) == 3:
            denoised = 255 * cv2.merge(dband)
        else:
            denoised = 255 * dband[0]
    return denoised

'''
'''
def tv_denoise(img, weight=.5):
    if len(img.shape) == 3:
        return (denoise_tv_chambolle(img.astype(float), weight=weight, multichannel=True))
    else:
        return (denoise_tv_chambolle(img.astype(float), weight=weight, multichannel=False))

#def sure_shrink(data):

# Applied to coefficients
def dwt_neighvar_shrink(dwt, W = [3], s2 = .5):
    # For every coefficient matrix
    dwt_est = [dwt[0]]
    for cdwt in dwt[1:]:
        cdwt_est = []
        for X in cdwt:
            S2 = np.ones(X.shape) * np.inf
            for w in W:
                S2 = np.minimum(S2, neighvar(X, w, s2))
                # new estimation
            Xbar = X * S2 / (S2 + s2)
            cdwt_est.append(Xbar)
        dwt_est.append(cdwt_est)
    return dwt_est

# Variance estimation
def neighvar(X, w, s2 = .5):
    # Variance
    Xmean = cv2.blur(X, (w, w), borderType = cv2.BORDER_REFLECT_101)
    Xvar = cv2.blur(X**2, (w, w), borderType = cv2.BORDER_REFLECT_101) - Xmean**2
    Xvarmin = np.maximum(np.zeros(Xvar.shape), Xvar - s2)
    # Estimation
    return Xvarmin

def blockSVDPredict(block):
    U, s, V = np.linalg.svd(block)
    S = np.zeros(np.shape(block))
    S[0, 0] = s[0]
    predict = U @ S @ V
    return predict[s.size // 2, s.size // 2]

def slideSVDPredict(data, Q=3):
    h, w = data.shape
    S = np.zeros((h, w))
    offset = Q//2
    data = cv2.copyMakeBorder(data, offset, offset, offset, offset, borderType=cv2.BORDER_REFLECT101)
    for i in range(h):
        for j in range(w):
            S[i,j] = blockSVDPredict(data[i: i+Q, j:j+Q])
    return S