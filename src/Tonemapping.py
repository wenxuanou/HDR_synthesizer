import numpy as np
import re
import os
import matplotlib.pyplot as plt
from skimage import (
    color,io,filters,util
)
from readEXR import readEXR

from cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeEXR, read_colorchecker_gm


def gammaEncode(img_linear):
    img_gamma = np.where(img_linear <= 0.0031308, 12.92*img_linear,
                         (1+0.055)*np.power(img_linear, 1/2.4)-0.055)
    return img_gamma

def tonemapping(img_HDR,tonemap_type,K,B,pixelNum,epsilon):
    img_TM = np.zeros_like(img_HDR)
    if tonemap_type == "all_channel":
        for channel in range(0,3):  # iterate color channel
            im_HDR = np.exp((1/pixelNum) * np.sum(np.log(img_HDR[:,:,channel] + epsilon)))   # im_HDR: scalar
            img_HDR_bar = (K / im_HDR) * img_HDR[:,:,channel]                      # img_HDR_bar: W*H
            i_white = B * np.amax(img_HDR_bar)                                 # i_white: scalar
            img_TM[:,:,channel] = np.divide(np.multiply(img_HDR_bar, (1+np.divide(img_HDR_bar, i_white**2 ))),
                                  1 + img_HDR_bar)

    if tonemap_type == "luminance":
        # convert to XYZ
        img_XYZ = lRGB2XYZ(img_HDR)
        # convert to xyY
        img_xyY = np.zeros_like(img_XYZ)
        img_xyY[:,:,0] = np.divide(img_XYZ[:,:,0], img_XYZ[:,:,0]+img_XYZ[:,:,1]+img_XYZ[:,:,2])
        img_xyY[:,:,1] = np.divide(img_XYZ[:,:,1], img_XYZ[:,:,0]+img_XYZ[:,:,1]+img_XYZ[:,:,2])
        img_xyY[:,:,2] = img_XYZ[:,:,1]

        img_lum = img_xyY[:,:,2]                # extract lumninance Y
        # tone map Y
        im_HDR = np.exp((1 / pixelNum) * np.sum(np.log(img_lum + epsilon)))  # im_HDR: scalar
        img_HDR_bar = (K / im_HDR) * img_lum                                # img_HDR_bar: W*H
        i_white = B * np.amax(img_HDR_bar)                                  # i_white: scalar
        img_lum = np.divide(np.multiply(img_HDR_bar, (1 + np.divide(img_HDR_bar, i_white ** 2))),
                            1 + img_HDR_bar)
        img_xyY[:,:,2] = img_lum

        # convert back to XYZ
        img_XYZ[:,:,0] = np.divide(np.multiply(img_xyY[:,:,0],img_xyY[:,:,2]),img_xyY[:,:,1] ,
                                   out=np.zeros_like(img_XYZ[:,:,0]),
                                   where=img_xyY[:,:,1]!=0)
        img_XYZ[:,:,1] = np.where(img_xyY[:,:,1]==0,0,img_xyY[:,:,2])
        img_XYZ[:,:,2] = np.divide(np.multiply((1-img_xyY[:,:,0]-img_xyY[:,:,1]),img_xyY[:,:,2]),img_xyY[:,:,1],
                                   out=np.zeros_like(img_XYZ[:,:,2]),
                                   where=img_xyY[:,:,1]!=0)
        # nan is where y == 0, make all to 0
        # img_XYZ = np.where(img_XYZ == np.NAN, 0, img_XYZ)

        # conver back to RGB
        img_TM = XYZ2lRGB(img_XYZ)

    return img_TM


if __name__ == "__main__":
    imgName = "../data/myImg_RAW_optimal_log_noGamma.EXR"
    img_HDR = readEXR(imgName)
    [W,H,C] = img_HDR.shape

    # tonemapping
    tonemap_type = "luminance"  # tonemapping type: all_channel/luminance
    K = 0.15  # key
    B = 0.95  # burn
    # K = 0.55
    # B = 0.95
    pixelNum = W * H
    epsilon = 0.0001

    img_TM = tonemapping(img_HDR, tonemap_type, K, B, pixelNum, epsilon)

    io.imshow(gammaEncode(img_TM))
    plt.show()
    writeEXR("noiseOptimal_Tonemap_"+tonemap_type+"_K"+str(K)+"_B"+str(B)+".EXR", img_TM)





