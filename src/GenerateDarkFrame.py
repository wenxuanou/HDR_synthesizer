import numpy as np
import re
import os
import matplotlib.pyplot as plt
from skimage import (
    color,io,filters,util
)
from readEXR import readEXR
from cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeEXR, read_colorchecker_gm


def loadImg(imgNames):
    imgNum = len(imgNames)

    # initialize parameters
    img = io.imread(imgNames[0])
    [W, H, C] = img.shape  # dimension of image
    regex = re.compile(r'\d+')  # regular expression

    img_out = np.zeros_like(img)

    # vectorize all images and combine to a matrix
    for file in imgNames:
        img = io.imread(file)
        img_out = img_out + img

    print("load finished")
    return img_out


def processTile(imgPath,darkFrame):
    for count,file in enumerate(imgPath):
        img = io.imread(file)
        img_out = img - darkFrame
        writeEXR("../data/tile/"+"tile"+str(count)+".EXR",img_out)


if __name__ == "__main__":
    # imgPath = "../data/dark_frame"
    # imgNames = []
    # for file in os.listdir(imgPath):
    #     if file.endswith(".tiff"):
    #         imgNames.append(os.path.join(imgPath, file))
    #
    # img_out= loadImg(imgNames)
    # writeEXR("darkframe.EXR",img_out)

    darkPath = "../data/darkframe.EXR"
    darkFrame = readEXR(darkPath)

    imgPath = "../data/tile"
    imgNames = []
    for file in os.listdir(imgPath):
        if file.endswith(".tiff"):
            imgNames.append(os.path.join(imgPath, file))

    processTile(imgNames,darkFrame)
