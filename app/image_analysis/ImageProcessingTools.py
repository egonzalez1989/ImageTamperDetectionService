from matplotlib import pyplot as plt
from scipy.special import erfc
from .DenoiseTechniques import *


'''
    Iterator of a function over every block of size MxN
'''
def block_map(data, function, size):
    h, w = data.shape
    if type(size) is int:
        M, N = size, size
    else:
        M, N = size
    hh, ww = h // M, w // N
    results = []
    for i in range(hh):
        for j in range(ww):
            results.append(function(data[i*M: (i+1)*M, j*N: (j+1)*N]))
    return results


def block_mean(data, B):
    h, w = data.shape
    #X = [plt.hist(data[i: i + B, j: j + B].flatten(), range=(-1, 1), bins=50, histtype='step')[0] for i in
         #range(0, h - B, B) for j in range(0, w - B, B)]  # Data
    X = [data[i: i + B, j: j + B] for i in range(0, h - B, B) for j in range(0, w - B, B)]
    return sum(X) / len(X)

'''shift for slide and pixel similarity
def stride_corr_similarity(data, P, stride = 0):
    h, w = data.shape
    hp, wp = P.shape
    if not stride:
        hs, ws = P.shape
    S = np.zeros(((h - hp) // hs + 1, (w - wp) // ws + 1))
    Pf = P.flatten()
    X = [data[i: i + hp, j: j + wp] for i in range(0, h - hp, hp) for j in range(0, w - wp, wp)]
    for i in range(0, h - stride + 1, stride):
        for j in range(0, w - stride + 1, stride):
            S[i // hs, j // stride] = np.corrcoef(Pf, plt.hist(data[i: i + hs, j: j + ws].flatten(), range=(-1, 1), bins=50, histtype='step')[0])[0, 1]
'''

def block_corr_similarity(data, P):
    h, w = data.shape
    #for i in range(0, h, B):
    hp, wp = P.shape
    Pf = P.flatten()
    X = [data[i: i + hp, j: j + wp].flatten() for i in range(0, h - hp + 1, hp) for j in range(0, w - wp + 1, wp)]
    S = list(map(lambda x: np.cov(x, Pf)[0,1], X))
    S = np.array(S).reshape((h - hp) // hp  + 1, (w - wp) // wp + 1)
    return S

def block_hist_corr_similarity(data, P):
    h, w = data.shape
    hp, wp = P.shape
    Phist = plt.hist(P.flatten(), range=(-1, 1), bins=50, histtype='step')[0]
    S = np.zeros(((h - hp) // hp  + 1, (w - wp) // wp + 1))
    for i in range(0, h - hp + 1, hp):
        for j in range(0, w - wp + 1, wp)   :
            S[i // hp, j // wp] = np.corrcoef(Phist, plt.hist(data[i: i + hp, j: j + wp].flatten(), range=(-1, 1), bins=50, histtype='step')[0])[0, 1]
    return S

def slide_corr_similarity(data, P, roll = 0):
    h, w = data.shape
    hp, wp = P.shape
    S = np.zeros((h - hp + 1, w - wp + 1))
    for i in range(hp):
        for j in range(wp):
            Sij = block_corr_similarity(data[i:, j:], P)
            S[i::hp,j::wp] = Sij
            if roll:
                P = np.roll(P, -1, axis = 1)
        if roll:
            P = np.roll(P, -1, axis=0)
    return S

def pmap_erfc(data, stdev=0, mean=None):
    if stdev == 0:  stdev = np.std(data)
    if mean is None: mean = np.mean(data),
    pmap = erfc(np.abs(data - mean) / (2 ** .5 * stdev))
    return pmap



def weighted_pmap_erfc(data, iters = 0):
    ##
    mean, stdev = np.mean(data), np.std(data)
    pmap = erfc(np.abs(data - mean) / (2 ** .5 * stdev))
    pmap = pmap - pmap.min()
    pmap = pmap / pmap.max()
    for i in range(iters):
        fdata, nw = data * pmap, np.sum(pmap)
        mean = np.sum(fdata) / nw
        stdev = (np.sum(pmap*(data - mean)**2) / nw) ** .5
        pmap = erfc(np.abs(data - mean) / (2 ** .5 * stdev))
    return pmap

def block_pmap_vector(data, B):
    h, w = data.shape
    pmap = pmap_erfc(data)
    X = [pmap[i: i + B, j: j + B].flatten() for i in range(0, h, B) for j in range(0, w, B)]
    return X

''' BXB blocks of noise ordered as unidimensional vector of size B²
'''
def block_noise_vector(data, B):
    h, w = data.shape
    noise = data - dwt_denoise(data)
    X = [noise[i: i + B, j: j + B].flatten() for i in range(0, h, B) for j in range(0, w, B)]
    return X

''' Draws the detected area over an image
'''
def draw_mask(img, mask, color = (0, 0, 255)):
    alpha = .5
    overlay = img.copy()
    output = img.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                    0, output)
    return output

def otsu_components(data, N = 3, A = .01, biggest = False):
    data = data - data.min()
    data = (data / data.max() * 255).astype('uint8')
    blur = cv2.GaussianBlur(data, (N, N), 0)
    # find otsu's threshold value with OpenCV function
    _, otsu = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = 1-otsu
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(otsu)
    idx = np.argsort(stats[:,-1])[::-1]
    out = np.zeros_like(otsu)
    for i in idx[1:]:
        _, _, _, _, a = stats[i]
        if a < A * data.size:
            break
        out[labels == i] = i
    return out

def otsu(data, N=3):
    data = data - data.min()
    data = (data / data.max() * 255).astype('uint8')
    blur = cv2.GaussianBlur(data, (N, N), 0)
    _, otsu = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = 1-otsu
    return otsu

''' 
'''
def classification_to_image(C, size, B, step):
    h, w = size
    # resulting matrix should have size (hB, wB)
    hB, wB = (h - B) // step + 1, (w - B) // step + 1
    Cimg = C.reshape((hB, wB))
    Cimg = cv2.resize(Cimg, None, fx = step, fy = step, interpolation = cv2.INTER_NEAREST)
    return Cimg

#np.corrcoef(Pf, plt.hist(data[i: i + B, j: j + B].flatten(), range=(-1, 1), bins=50, histtype='step')[0])[0, 1]



''' Gives a hXw matrix with blocks of 0's bordered by 1's (a JPEG pattern)'''
def jpeg_pattern(h, w):
    A = np.zeros((h, w))
    for i in range(0, h, 8): A[i, :] = 1
    for j in range(0, w, 8): A[:, j] = 1
    return A


''' Operation to disclose 8X8 borders froma a JPEG compressed image
'''
def jpeg_detection(img):
    img = cv2.equalizeHist(img)
    img = img.astype(float)
    DX = np.abs(cv2.filter2D(img, -1, np.array([1, -1])))
    DY = np.abs(cv2.filter2D(img.T, -1, np.array([1, -1]))).T
    DXY = DX + DY

    # Remove non-JPEG edges
    #DXY2 = cv2.GaussianBlur(DXY, (3,3), 0)
    #plt.imshow(DXY2, cmap='gray')
    #plt.show()

    #DXY = DXY - DXY2
    #plt.imshow(DXY, cmap='gray')
    #plt.show()

    JK = jpeg_pattern(64, 64)
    JK = 1. * JK / np.sum(JK)
    DXY = cv2.filter2D(DXY, -1, JK)
    #img = cv2.GaussianBlur(img, (7, 7), 0)


    '''JK = np.zeros((8,8))
    JK[0,0] = 1
    JK = np.tile(JK, (3,3))
    JK = 1. * JK / np.sum(JK)
    DXY = cv2.filter2D(DXY, -1, JK)'''
    return DXY

def LBP(img):
    img = cv2.copyMakeBorder(img,1,1,1,1,cv2.BORDER_REFLECT101)
    lbp = np.zeros_like(img)
    pow2 = [2**x for x in range(9)]
    h, w = img.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            BP = img[i-1:i+2, j-1:j+2] > img[i, j]
            lbp[i-1, j-1] = np.dot(BP.flatten(), pow2)
    return lbp[1:-1, 1:-1]


''' Extracts values from interpolated values. This only works for green band in xGGx or GxxG patterns.
'''
def get_filter_pattern(cfa_pattern, height, width):
    h, w = cfa_pattern.shape
    cfa_matrix = np.tile(cfa_pattern, ((height + h - 1) // h, (width + w - 1) // w))
    cfa_matrix = cfa_matrix[: height, : width]
    return cfa_matrix

def extract_interpolated(data, cfa_pattern):
    h, w = data.shape
    true_array = get_filter_pattern(np.ones_like(cfa_pattern) - cfa_pattern, h, w)
    return np.extract(true_array, data)

def extract_acquired(data, cfa_pattern):
    h, w = data.shape
    true_array = get_filter_pattern(cfa_pattern, h, w)
    return np.extract(true_array, data)