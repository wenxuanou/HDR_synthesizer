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
def weight(imgVal,w_type,Zmin,Zmax,t_expo=None,isReg=False,isLinear=False,g=None,sigmaGauss=None):

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
    if w_type == "optimal":
        return W_optimal(imgVal,t_expo,Zmin,Zmax,g,sigmaGauss)


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

def W_optimal(imgVal,t_expo,Zmin,Zmax,g,sigmaGauss):
    w = (imgVal >= Zmin) & (imgVal<=Zmax)
    w = w * imgVal
    w = np.where(w == 0, 0, t_expo**2 / (g*w+sigmaGauss))
    return w


def gammaEncode(img_linear):
    img_gamma = np.where(img_linear <= 0.0031308, 12.92*img_linear,
                         (1+0.055)*np.power(img_linear, 1/2.4)-0.055)
    return img_gamma


def loadImg(jpgNames,N,darkFrame=None,isNoiseCalibrate=False,tnc=None):
    imgNum = len(jpgNames)

    # initialize parameters
    img = io.imread(jpgNames[0])
    [W, H, C] = img.shape  # dimension of image
    W_ = int(W / N)            # get dimension after downsampling
    H_ = int(H / N)
    W = W_ * N
    H = H_ * N
    regex = re.compile(r'\d+')  # regular expression

    img_down_flat = np.zeros((W_ * H_ * 3, imgNum))  # store downsampled flattened image, (W_*H_*3)*imgNum
    img_full_flat = np.zeros((W * H * 3, imgNum))  # store full flattened image, (W*H*3)*imgNum
    log_t_expo = np.zeros((imgNum, 1))  # store exposure time, imgNum*1

    # vectorize all images and combine to a matrix
    for count, file in enumerate(jpgNames):
        img = io.imread(file)

        # downsample
        img_down = downSample(img, N)

        # find matching k in filename
        k = int(regex.search(file).group(0))        # look for number k after filename
        t_expo = 1/k                        # exposure time = 1/k
        log_t_expo[count] = np.log(t_expo)  # log exposure time

        # noise calibration, only perform when loading RAW
        if isNoiseCalibrate:
            img = img - (t_expo / tnc) * darkFrame
            img = np.where(img < 0, 0, img)        # make sure no negative

        # weighting image pixel value, flatten image
        img_down_flat[:, count] = np.reshape(img_down[0:W_,0:H_,:], (W_ * H_ * 3,))  # flatten to (W_*H_*3)*1
        img_full_flat[:, count] = np.reshape(img[0:W,0:H,:], (W * H * 3,))  # flatten to (W*H*3)*1

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

    # noise parameters, obtained from NoiseCalibration.py
    # g_noise = 4321.085925183922
    # sigmaGauss = -13924502.744487116
    g_noise = 1000
    sigmaGauss = 0.0023942

    temp1 = np.zeros((img_flat.shape[0],))
    temp2 = np.zeros((img_flat.shape[0],))
    print("merging HDR")
    # print(np.amax(img_flat))
    # print(np.amin(img_flat))

    if merge_type == "linear":

        for j in range(0,img_flat.shape[1]):
            if weight_type == "photon":
                img_w = weight(img_flat[:, j], weight_type, log_t_expo[j], Zmin, Zmax)
            elif weight_type == "optimal":
                img_w = weight(img_flat[:, j], weight_type, Zmin, Zmax, np.exp(log_t_expo[j]),
                               g=g_noise, sigmaGauss=sigmaGauss)
            else:
                img_w = weight(img_flat[:, j], weight_type, Zmin, Zmax)
            temp1 = np.add(temp1, img_w)
            temp2 = np.add(temp2, np.divide(np.multiply(img_w, img_flat[:, j]), np.exp(log_t_expo[j])))
        # n = np.where(temp1 > 0, temp1, np.infty).argmin()
        # temp1 = np.where(temp1 == 0, n, temp1)
        # temp2 = np.where(temp2 == np.NAN, 0, temp2)

        img_HDR = np.divide(temp2,temp1)


    if merge_type == "log":
        # # insert small value to avoid zeros during merging
        # m = np.where(img_flat > 0, img_flat, np.infty).argmin()
        epsilon = 0.00001

        for j in range(0,img_flat.shape[1]):
            if weight_type == "photon":
                img_w = weight(img_flat[:, j], weight_type, log_t_expo[j], Zmin, Zmax)

            elif weight_type == "optimal":
                img_w = weight(img_flat[:, j], weight_type, Zmin, Zmax, np.exp(log_t_expo[j]),
                               g=g_noise, sigmaGauss=sigmaGauss)
            else:
                img_w = weight(img_flat[:, j], weight_type, Zmin, Zmax)
            temp1 = np.add(temp1, img_w)
            temp2 = np.add(temp2, np.multiply(img_w, (np.log(img_flat[:, j]+epsilon) - log_t_expo[j])))

        m = np.where(temp1 > 0, temp1, np.infty).argmin()
        # n = np.where(temp2 > 0, temp2, np.infty).argmin()
        temp1 = np.where(temp1 == 0, m, temp1)
        # temp2 = np.where(temp2 == 0, n, temp2)

        print("temp1: ")
        print(np.amin(temp1))
        print(np.amax(temp1))
        print("temp2: ")
        print(np.amin(temp2))
        print(np.amax(temp2))

        img_HDR = np.exp(np.divide(temp2, temp1))
        print("HDR: ")
        print(np.amin(img_HDR))
        print(np.amax(img_HDR))

    # transform HDR image back to matrix form
    img_HDR = np.reshape(img_HDR, (W, H, 3))
    print("HDR finished")
    return img_HDR




if __name__ == '__main__':
    # load RAW image, no need to linearize
    imgPath = "../data/my_img"
    imgNames = []
    img_type = "RAW"    # img_type: RAW/JPG
    weight_type = "optimal"     # weight scheme type: uniform/tent/gauss/photon/optimal     ## only use "optimal" with RAW
    merge_type = "log"    # merging type: linear/log

    # load dark frame for noise calibration
    darkPath = "../data/darkframe.EXR"
    darkFrame = readEXR(darkPath)
    isNoiseCalibrate = True         # true if want to perform noise calibration, only perform on RAW
    tnc = 1/8                       # dark frame exposure time

    img_linear = np.array([])
    log_t_expo = np.array([])
    W = 0
    H = 0

    if img_type == "RAW":
        for file in os.listdir(imgPath):
            if file.endswith(".tiff"):
                imgNames.append(os.path.join(imgPath, file))

        imgNum = len(imgNames)
        N = 1  # downsample factor

        img_down_flat, img_linear, log_t_expo, W_, H_ = loadImg(imgNames,N,darkFrame,isNoiseCalibrate,tnc)
        img_linear = img_linear / 65535
        W = W_
        H = H_



    if img_type == "JPG":
        for file in os.listdir(imgPath):
            if file.endswith(".JPG"):
                imgNames.append(os.path.join(imgPath, file))

        imgNum = len(imgNames)
        N = 200  # downsample factor, my camera is 5184*3456
        lamb = 200  # lambda = 200, large lambda produces smooth g

        img_down_flat, img_full_flat, log_t_expo, W_, H_ = loadImg(imgNames,N)
        img_linear = linearization(img_down_flat, img_full_flat, log_t_expo, lamb, weight_type, W_, H_)
        W = W_ * N
        H = H_ * N


    img_HDR = HDR(img_linear, log_t_expo, weight_type, merge_type, W, H)
    img_HDR = img_HDR / np.amax(img_HDR)

    io.imshow(gammaEncode(img_HDR))
    plt.title(weight_type+", "+merge_type)
    plt.show()
    writeEXR("myImg_"+img_type+"_"+weight_type+"_"+merge_type+"_noGamma.EXR", img_HDR)