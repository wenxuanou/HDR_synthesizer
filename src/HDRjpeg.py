import numpy as np
import re
import os
import matplotlib.pyplot as plt
from skimage import (
    color,io,filters,util
)
from readEXR import readEXR

from cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeEXR, read_colorchecker_gm

def downSample(img,N):
    # downsample image by factor of N
    # assume img has 3 color channels
    return img[::N,::N,:]

# Weighting schemes
def weight(imgVal,w_type,Zmin,Zmax,t_expo=None,isReg=False,isLinear=False):

    if w_type == "uniform":
        return W_uniform(imgVal,Zmin,Zmax)
    if w_type == "tent":
        return W_tent(imgVal,Zmin,Zmax)
    if w_type == "gauss":
        return W_gauss(imgVal,Zmin,Zmax,isLinear)
    if w_type == "photon":
        if isReg:
            return np.ones_like(imgVal)
        else:
            return W_photon(imgVal,t_expo,Zmin,Zmax)


def W_uniform(imgVal,Zmin,Zmax):
    w = (imgVal >= Zmin) & (imgVal<=Zmax)
    return w

def W_tent(imgVal,Zmin,Zmax):
    w = (imgVal >= Zmin) & (imgVal<=Zmax)
    w = w * imgVal
    w = np.minimum(w,1-w)
    return w

def W_gauss(imgVal,Zmin,Zmax,isLinear):
    if isLinear:
        w = (imgVal >= Zmin) & (imgVal<=Zmax)
        w = w * imgVal
        w = np.exp(-4 * np.power(w-128,2) / (128**2))
    else:
        w = (imgVal >= Zmin) & (imgVal<=Zmax)
        w = w * imgVal
        w = np.exp(-4 * np.power(w-0.5,2) / (0.5**2))
    return w

def W_photon(imgVal,t_expo,Zmin,Zmax):
    w = (imgVal >= Zmin) & (imgVal<=Zmax)
    w = w * t_expo
    return w

def gammaEncode(img_linear):
    img_gamma = np.where(img_linear <= 0.0031308, 12.92*img_linear,
                         (1+0.055)*np.power(img_linear, 1/2.4)-0.055)
    return img_gamma

def loadImg(imgNames,N):
    imgNum = len(imgNames)

    # initialize parameters
    img = io.imread(imgNames[0])
    [W, H, C] = img.shape  # dimension of image
    W_ = int(W / N)            # get dimension after downsampling
    H_ = int(H / N)
    regex = re.compile(r'\d+')  # regular expression

    img_down_flat = np.zeros((W_ * H_ * 3, imgNum))  # store downsampled flattened image, (W_*H_*3)*imgNum
    img_full_flat = np.zeros((W * H * 3, imgNum))  # store full flattened image, (W*H*3)*imgNum
    log_t_expo = np.zeros((imgNum, 1))  # store exposure time, imgNum*1

    # vectorize all images and combine to a matrix
    for count, file in enumerate(imgNames):
        img = io.imread(file)
        # downsample
        img_down = downSample(img, N)

        # find matching k in filename
        k = int(regex.search(file).group(0))
        t_expo = (1 / 2048) * np.power(2, k - 1)  # exposure time
        log_t_expo[count] = np.log(t_expo)  # log exposure time

        # weighting image pixel value, flatten image
        img_down_flat[:, count] = np.reshape(img_down, (W_ * H_ * 3,))  # flatten to (W_*H_*3)*1
        img_full_flat[:, count] = np.reshape(img, (W * H * 3,))  # flatten to (W*H*3)*1

    print("load finished")
    return img_down_flat, img_full_flat, log_t_expo, W_, H_

def linearization(img_down_flat,img_full_flat,log_t_expo,lamb,weight_type,W_,H_):
    # solve minimization equation
    Zmin = 0    # linearization range
    Zmax = 255
    n = 256     # dimension of g

    A = np.zeros((W_ * H_ * 3 * imgNum + n + 1, W_ * H_ * 3 + n))
    b = np.zeros((W_ * H_ * 3 * imgNum + n + 1, 1))

    if weight_type == "photon":
        k = 1
        for i in range(0, img_down_flat.shape[0]):
            for j in range(0, img_down_flat.shape[1]):
                wij = weight(img_down_flat[i, j], weight_type, Zmin, Zmax, log_t_expo[j],
                             isReg=False)  # weight the pixel intensity
                A[k, int(img_down_flat[i, j])] = wij
                A[k, n + i] = -1 * wij
                b[k] = wij * log_t_expo[j]
                k = k + 1

        A[k, 128] = 1
        k = k + 1
        # regularization term
        for i in range(0, n - 3):
            A[k, i] = lamb * weight(i, weight_type, Zmin, Zmax, isReg=True)
            A[k, i + 1] = -2 * lamb * weight(i, weight_type, Zmin, Zmax, isReg=True)
            A[k, i + 2] = lamb * weight(i, weight_type, Zmin, Zmax, isReg=True)
            k = k + 1
    else:
        k = 1
        for i in range(0, img_down_flat.shape[0]):
            for j in range(0, img_down_flat.shape[1]):
                wij = weight(img_down_flat[i, j], weight_type, Zmin, Zmax,isLinear=True)  # weight the pixel intensity
                A[k, int(img_down_flat[i, j])] = wij
                A[k, n + i] = -1 * wij
                b[k] = wij * log_t_expo[j]
                k = k + 1

        A[k, 128] = 1
        k = k + 1
        # regularization term
        for i in range(0, n - 3):
            A[k, i] = lamb * weight(i, weight_type, Zmin, Zmax,isLinear=True)
            A[k, i + 1] = -2 * lamb * weight(i, weight_type, Zmin, Zmax,isLinear=True)
            A[k, i + 2] = lamb * weight(i, weight_type, Zmin, Zmax,isLinear=True)
            k = k + 1
    print("equation solved")

    x, r, rank, s = np.linalg.lstsq(A, b, rcond=None)
    g = x[0:n]  # obtain g, 256*1
    # plt.figure(1)
    # plt.plot(g)
    # plt.show()

    # retrieve linear intensity
    img_linear = np.exp(g[img_full_flat.astype(int)])

    # normalize to [0,1] range
    img_linear = img_linear / 255
    print("linearization finished")
    return img_linear

def HDR(img_flat, log_t_expo, weight_type, merge_type, W, H):
    # merge LDR to HDR
    img_HDR = np.zeros((img_flat.shape[0], 1))  # (W*H*3)*1
    Zmin = 0.05
    Zmax = 0.95

    temp1 = np.zeros((img_flat.shape[0],1))
    temp2 = np.zeros((img_flat.shape[0],1))


    # print(np.amin(img_flat))

    print("HDR begin")
    if merge_type == "linear":

        for j in range(0,img_flat.shape[1]):
            if weight_type == "photon":
                img_w = weight(img_flat[:, j], weight_type, log_t_expo[j], Zmin, Zmax)
            else:
                img_w = weight(img_flat[:, j], weight_type, Zmin, Zmax)
            temp1 = np.add(temp1, img_w)
            temp2 = np.add(temp2, np.divide(np.multiply(img_w, img_flat[:, j]), np.exp(log_t_expo[j])))
        n = np.where(temp1 > 0, temp1, np.infty).argmin()
        # n = np.where(temp2 > 0, temp2, np.infty).argmin()
        temp1 = np.where(temp1 == 0, n, temp1)
        # temp2 = np.where(temp2 == 0, n, temp2)
        img_HDR = np.divide(temp2,temp1)


    if merge_type == "log":
        epsilon = 0.0001    # add small value avoid nan

        for j in range(0,img_flat.shape[1]):
            if weight_type == "photon":
                img_w = weight(img_flat[:, j], weight_type, log_t_expo[j], Zmin, Zmax)
            else:
                img_w = weight(img_flat[:, j], weight_type, Zmin, Zmax)
            temp1 = np.add(temp1, img_w)
            temp2 = np.add(temp2, np.multiply(img_w, (np.log(img_flat[:, j]+epsilon) - log_t_expo[j])))
        m = np.where(temp1 > 0, temp1, np.infty).argmin()
        n = np.where(temp2 > 0, temp2, np.infty).argmin()
        temp1 = np.where(temp1 == 0, m, temp1)
        temp2 = np.where(temp2 == 0, n, temp2)
        img_HDR = np.exp(np.divide(temp2, temp1))

    # transform HDR image back to matrix form
    img_HDR = np.reshape(img_HDR, (W, H, 3))
    print("merge finished")
    return img_HDR

def loadColorCheker():
    r,g,b = read_colorchecker_gm()
    colorchecker = np.zeros((r.shape[0],r.shape[1],3))
    colorchecker[:,:,0] = r
    colorchecker[:,:,1] = g
    colorchecker[:,:,2] = b
    # patchNum = 6
    patchLoc = np.zeros((6,2))
    patchLoc[:,0] = np.arange(0,6)
    patchLoc[:,1] = 3 * np.ones(6)
    return colorchecker,patchLoc

def getImgPatchLum(img_HDR):
    patchNum = 6
    img_patch_lum = np.zeros((6,1))

    img_lum = lRGB2XYZ(img_HDR)[:, :, 1]

    plt.figure(2)
    io.imshow(gammaEncode(img_HDR)*5)
    for count in range(0, patchNum):
        pt = plt.ginput(1)  # select the center of each patch
        x = np.round(pt[0][0]).astype(int)
        y = np.round(pt[0][1]).astype(int)
        img_patch_lum[count] = np.mean(img_lum[y-1:y+1, x-1:x+1])
    plt.close()
    return img_patch_lum



if __name__ == '__main__':

    # linearlize rendered jpg image
    # read in jpg
    jpgPath = "../data/door_stack"
    imgNames = []
    for file in os.listdir(jpgPath):
        if file.endswith(".jpg"):
            imgNames.append(os.path.join(jpgPath, file))
    imgNum = len(imgNames)

    N = 200  # downsample factor
    weight_type = "gauss"     # weight scheme type: uniform/tent/gauss/photon
    merge_type = "log"    # merging type: linear/log

    lamb = 200  # lambda = 200, large lambda produces smooth g

    # load weighted image into a matrix
    # img_full_flat: (W*H*3)*imgNum
    # img_down_flat: (W_*H_*3)*imgNum
    # log_t_expo: imgNum*1
    # W_, H_: size of downsampled image
    img_down_flat, img_full_flat, log_t_expo, W_, H_ = loadImg(imgNames,N)

    # perfom linearization
    # img_linear: (W*H*3)*imgNum, normalized to [0,1] range
    img_linear = linearization(img_down_flat, img_full_flat, log_t_expo, lamb, weight_type, W_, H_)

    # merge exposure stack to HDR
    # img_HDR: W*H*3
    W = W_*N
    H = H_*N
    img_HDR = HDR(img_linear, log_t_expo, weight_type, merge_type, W, H)
    img_HDR = img_HDR / np.amax(img_HDR)

    io.imshow(gammaEncode(img_HDR))
    plt.title(weight_type+", "+merge_type)
    plt.show()
    # writeEXR("JPG_"+weight_type+"_"+merge_type+".EXR", gammaEncode(img_HDR))
    writeEXR("JPG_"+weight_type+"_"+merge_type+"_noGamma.EXR", img_HDR)


    # load color checker patch
    # select 4,8,12,16,20,24 patch, total 6
    colorchecker, checker_patchLoc = loadColorCheker()  # patchLoc: 6*2
    # select corresponding patch in image
    # img_patch_lum: 6*1
    img_patch_lum = getImgPatchLum(img_HDR)
    m = np.where(img_patch_lum > 0, img_patch_lum, np.infty).argmin()
    img_patch_lum = np.where(img_patch_lum == 0, 0.01 * m, img_patch_lum)

    # get patch luminance
    checker_patch_lum = np.zeros((6, 1))  # checker_patch_lum: 6*1
    checker_lum = lRGB2XYZ(colorchecker)[:, :, 1]
    for count in range(0, 6):
        x = int(checker_patchLoc[count, 0])
        y = int(checker_patchLoc[count, 1])
        checker_patch_lum[count] = checker_lum[y, x]  # extract patch luminance
    # linear regression
    A = np.vstack([np.arange(0, 6), np.ones(6)]).T  # log of luminance map to 0~5
    # solve y = mx + c; m, c are scalar; linear regression on image patch luminance
    m, c = np.linalg.lstsq(A, np.log(img_patch_lum), rcond=None)[0]
    lum_Regress = m * np.arange(0, 6) + c  # map [0:5] to log of luminance

    # matching pixel in image
    LSE = np.linalg.norm(lum_Regress - np.log(checker_patch_lum))

    plt.plot(np.log(checker_patch_lum))
    plt.plot(np.log(img_patch_lum))
    plt.legend(['checker_path', 'img_patch'])
    plt.show()

    print("LSE: ", LSE)
