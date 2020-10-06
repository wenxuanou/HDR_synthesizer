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
def weight(imgVal,w_type,Zmin,Zmax,t_expo=None,isReg=False):

    if w_type == "uniform":
        return W_uniform(imgVal,Zmin,Zmax)
    if w_type == "tent":
        return W_tent(imgVal,Zmin,Zmax)
    if w_type == "gauss":
        return W_gauss(imgVal,Zmin,Zmax)
    if w_type == "photon":
        if isReg:
            img_w = np.ones_like(imgVal)
            return img_w
        else:
            img_w = W_photon(imgVal,t_expo,Zmin,Zmax)
            return img_w


def W_uniform(imgVal,Zmin,Zmax):
    w = (imgVal >= Zmin) & (imgVal <= Zmax)
    return w

def W_tent(imgVal,Zmin,Zmax):
    w = (imgVal >= Zmin) & (imgVal <= Zmax)
    w = w * imgVal
    w = np.minimum(w, 1 - w)
    return w

def W_gauss(imgVal,Zmin,Zmax):
    w = (imgVal >= Zmin) & (imgVal <= Zmax)
    w = w * imgVal
    w = np.exp(-4 * np.power(w - 0.5, 2) / (0.5 ** 2))
    return w

def W_photon(imgVal,t_expo,Zmin,Zmax):
    w = (imgVal >= Zmin) & (imgVal <= Zmax)
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
    W = int(W / N)            # get dimension after downsampling
    H = int(H / N)
    regex = re.compile(r'\d+')  # regular expression

    img_flat = np.zeros((W * H * 3, imgNum))  # store flattened image, (W*H*3)*imgNum
    log_t_expo = np.zeros((imgNum, 1))  # store exposure time, imgNum*1

    # vectorize all images and combine to a matrix
    for count, file in enumerate(imgNames):
        img = io.imread(file)
        # downsample
        img = downSample(img, N)
        # regularize pixel intensity to [0,1]
        img = img / 65535

        # find matching k in filename
        k = int(regex.search(file).group(0))
        t_expo = (1 / 2048) * np.power(2, k - 1)  # exposure time
        log_t_expo[count] = np.log(t_expo)  # log exposure time

        # flatten image, raw no need to weight intensity    ###dropped out extra pixels
        img_flat[:, count] = np.reshape(img[0:W,0:H,:], (W * H * 3,))  # flatten to (W*H*3)*1

    print("load image finished")
    return img_flat,log_t_expo,W,H

def HDR(img_flat, log_t_expo, weight_type, merge_type, W, H):
    # merge LDR to HDR
    img_HDR = np.zeros((img_flat.shape[0], 1))  # (W*H*3)*1
    Zmin = 0.05
    Zmax = 0.95

    temp1 = np.zeros((1,img_flat.shape[0]))
    temp2 = np.zeros((1,img_flat.shape[0]))
    print("merging HDR")

    if merge_type == "linear":

        for j in range(0,img_flat.shape[1]):
            if weight_type == "photon":
                img_w = weight(img_flat[:, j], weight_type, log_t_expo[j], Zmin, Zmax)
            else:
                img_w = weight(img_flat[:, j], weight_type, Zmin, Zmax)
            temp1 = np.add(temp1, img_w)
            temp2 = np.add(temp2, np.divide(np.multiply(img_w, img_flat[:, j]), np.exp(log_t_expo[j])))
        n = np.where(temp1 > 0, temp1, np.infty).argmin()
        temp1 = np.where(temp1 == 0, n, temp1)
        # temp2 = np.where(temp2 == np.NAN, 0, temp2)

        img_HDR = np.divide(temp2,temp1)


    if merge_type == "log":
        # # insert small value to avoid zeros during merging
        # m = np.where(img_flat > 0, img_flat, np.infty).argmin()
        epsilon = 0.00001

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

        print("HDR: ")
        print(np.amin(img_HDR))
        print(np.amax(img_HDR))

    # transform HDR image back to matrix form
    img_HDR = np.reshape(img_HDR, (W, H, 3))
    print("HDR finished")
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
    io.imshow(gammaEncode(img_HDR))
    for count in range(0, patchNum):
        pt = plt.ginput(1)  # select the center of each patch
        x = np.round(pt[0][0]).astype(int)
        y = np.round(pt[0][1]).astype(int)
        img_patch_lum[count] = np.mean(img_lum[y-1:y+1, x-1:x+1])
    plt.close()
    return img_patch_lum



if __name__ == '__main__':
    # load RAW image, no need to linearize
    rawPath = "../data/door_stack"
    imgNames = []
    for file in os.listdir(rawPath):
        if file.endswith(".tiff"):
            imgNames.append(os.path.join(rawPath, file))

    imgNum = len(imgNames)
    weight_type = "gauss"  # weight scheme type: uniform/tent/gauss/photon
    merge_type = "log"    # merging type: linear/log
    N = 1  # downsample factor

    # load image into a matrix
    # img_flat: (W*H*3)*imgNum, normalized to [0,1];
    # log_t_expo: imgNum*1
    img_flat, log_t_expo, W, H = loadImg(imgNames,N)

    # merge exposure stack to HDR
    # img_HDR: W*H*3
    img_HDR = HDR(img_flat, log_t_expo, weight_type, merge_type, W, H)
    img_HDR = img_HDR / np.amax(img_HDR)
    io.imshow(gammaEncode(img_HDR))
    # plt.title(weight_type+", "+merge_type)
    # plt.show()
    writeEXR("RAW_"+weight_type+"_"+merge_type+".EXR", gammaEncode(img_HDR))
    # writeEXR("RAW_"+weight_type+"_"+merge_type+"_noGamma.EXR", img_HDR)



    # load color checker patch
    # select 4,8,12,16,20,24 patch, total 6
    colorchecker,checker_patchLoc = loadColorCheker()       # patchLoc: 6*2
    # select corresponding patch in image
    # img_patch_lum: 6*1
    img_patch_lum = getImgPatchLum(img_HDR)
    m = np.where(img_patch_lum > 0, img_patch_lum, np.infty).argmin()
    img_patch_lum = np.where(img_patch_lum == 0, 0.01*m, img_patch_lum)

    # get patch luminance
    checker_patch_lum = np.zeros((6,1))                         # checker_patch_lum: 6*1
    checker_lum = lRGB2XYZ(colorchecker)[:,:,1]
    for count in range(0,6):
        x = int(checker_patchLoc[count,0])
        y = int(checker_patchLoc[count,1])
        checker_patch_lum[count] = checker_lum[y,x]             # extract patch luminance
    # linear regression
    A = np.vstack([np.arange(0,6),np.ones(6)]).T   # log of luminance map to 0~5
    # solve y = mx + c; m, c are scalar; linear regression on image patch luminance
    m,c = np.linalg.lstsq(A, np.log(img_patch_lum), rcond=None)[0]
    lum_Regress = m * np.arange(0,6) + c                        # map [0:5] to log of luminance

    # matching pixel in image
    LSE = np.linalg.norm(lum_Regress - np.log(checker_patch_lum))
    plt.figure(3)
    plt.plot(np.log(checker_patch_lum))
    plt.plot(np.log(img_patch_lum))
    plt.legend(['checker_path','img_patch'])
    plt.show()

    print("LSE: ", LSE)

