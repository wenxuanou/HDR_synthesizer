import numpy as np
import re
import os
import matplotlib.pyplot as plt
from skimage import (
    color,io,filters,util
)
from readEXR import readEXR

from cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeEXR, read_colorchecker_gm

def loadColorCheker():
    r,g,b = read_colorchecker_gm()
    colorchecker = np.zeros((r.shape[0],r.shape[1],3))
    colorchecker[:,:,0] = r
    colorchecker[:,:,1] = g
    colorchecker[:,:,2] = b
    return colorchecker

def gammaEncode(img_linear):
    img_gamma = np.where(img_linear <= 0.0031308, 12.92*img_linear,
                         (1+0.055)*np.power(img_linear, 1/2.4)-0.055)
    return img_gamma

def getImgPatch(img_HDR):
    patchNum = 24
    img_patch = np.ones((patchNum,4))

    plt.figure(2)
    io.imshow(gammaEncode(img_HDR))
    for count in range(0, patchNum):
        pt = plt.ginput(1)  # select the center of each patch
        x = np.round(pt[0][0]).astype(int)
        y = np.round(pt[0][1]).astype(int)
        img_patch[count,0:3] = np.mean(img_HDR[y-1:y+1, x-1:x+1,:],axis=(0,1))
    plt.close()
    return img_patch

def manualBalance(img_rgb, subImg):
    subImg_avg = subImg # average of each channel
    # take green as a reference, dimension 1 from 0,1,2
    r_scale = subImg_avg[1] / subImg_avg[0] # g/r
    g_scale = 1
    b_scale = subImg_avg[1] / subImg_avg[2] # g/b

    img_rgb[:,:,0] = img_rgb[:,:,0] * r_scale
    img_rgb[:,:,2] = img_rgb[:,:,2] * b_scale

    return img_rgb

if __name__ == "__main__":
    colorchecker = loadColorCheker()    # colorchecker: 4*6*3   ## access patch by colorchecker[y,x]

    imgName = "../data/RAW_gauss_log_noGamma.EXR"
    img_HDR = readEXR(imgName)
    img_patch = getImgPatch(img_HDR)    # img_patch: 24*3
    # print(img_patch.shape)


    colorchecker_flat = np.ones((24,3))
    (W,H,C) = colorchecker.shape
    k = 0
    for j in range(0,H):
        for i in range(0, W):
            colorchecker_flat[k,:] = colorchecker[i,j,:]
            k = k+1

    A = np.zeros((3*24,12))
    for count in range(0,24):
        A[count*3,0:4] = img_patch[int(count),:].T
        A[count*3+1,4:8] = img_patch[int(count),:].T
        A[count*3+2,8:12] = img_patch[int(count),:].T


    b = np.reshape(colorchecker_flat,(24*3,1))
    x, r, rank, s = np.linalg.lstsq(A, b, rcond=None)   # x: 12*1

    x = np.reshape(x,(3,4))
    (W,H,C) = img_HDR.shape
    img_corr = np.zeros_like(img_HDR)
    for w in range(0,W):
        for h in range(0,H):
            temp = img_HDR[w,h,:]
            temp = np.append(temp,[1])
            temp = np.reshape(temp,(4,1))
            result = np.dot(x, temp)
            img_corr[w, h, 0] = result[0]
            img_corr[w, h, 1] = result[1]
            img_corr[w, h, 2] = result[2]

    img_corr = img_corr / np.amax(img_corr)
    print("color correction finished!")

    # io.imshow(gammaEncode(img_corr))
    # plt.show()
    # writeEXR("color corrected.EXR", img_corr)

    ####### ps: manual white balance result is worse than that before white balance

    # white balance based on white patch
    subImg = img_patch[4,:]
    img_corr = manualBalance(img_corr, subImg)
    plt.close()

    # io.imshow(gammaEncode(img_corr))
    # plt.show()
    writeEXR("color corrected.EXR", img_corr)