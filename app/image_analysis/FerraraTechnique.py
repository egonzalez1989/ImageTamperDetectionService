from .ImageProcessingTools import *
from matplotlib import pytplot as plt

def linearDemosaicError(data):
    k = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]) / 5.
    return data - cv2.filter2D(data, -1, k)

def gaussianWindow(n):
        a = np.zeros((6 * n + 1, 6 * n + 1))  # a4.shape[0]
        a[3 * n, 3 * n] = 1
        W = cv2.GaussianBlur(a.astype(float), (2 * n + 1, 2 * n + 1), 1)[2 * n: -2 * n, 2 * n: -2 * n]
        for i in range(2*n+1):
            for j in range((i+1)%2, 2*n+1, 2):
                W[i, j] = 0
        return W / np.sum(W)



def LFeature(block):
    # Green CFA pattern
    G = np.array([[1, 0], [0, 1]])
    GMA = extract_acquired(block, G)
    GMI = extract_interpolated(block, G)
    return np.log(GMA / GMI)


def FerraraAnalysis(img, B, K):
    e = linearDemosaicError(img.astype(float))
    W = gaussianWindow(K)
    c = 1 - np.sum(W ** 2)
    mu = cv2.filter2D(e, W)
    mu2 = cv2.filter2D(e ** 2, W)
    sigma2 = (mu2 - mu ** 2) / c
    L = block_map(sigma2, LFeature, B)
    plt.imshow(L)
    plt.show()

