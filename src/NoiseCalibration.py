import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from skimage import (
    color,io,filters,util
)
from readEXR import readEXR
from cp_hw2 import lRGB2XYZ, XYZ2lRGB, writeEXR, read_colorchecker_gm


def loadImg(imgNames):
    imgNum = len(imgNames)

    img_out = np.zeros((100,100,3,imgNum))    # for 50 tiles, img_out: 10*10*3*50

    # vectorize all images and combine to a matrix
    for count,file in enumerate(imgNames):
        img = readEXR(file)
        (W,H,C) = img.shape
        subImg = img[int(W/4):int(W/4)+100,int(H/4):int(H/4)+100,:]      # extract 10*10 patch at middle
        img_out[:,:,:,count] = subImg

    print("load finished")
    return img_out


if __name__ == "__main__":
    imgPath = "../data/tile"
    imgNames = []
    for file in os.listdir(imgPath):
        if file.endswith(".EXR"):
            imgNames.append(os.path.join(imgPath, file))

    img_tiles = loadImg(imgNames)           # img_out: 10*10*3*50; pixel values should be 0~65535
    # print(img_tiles.shape)

    # # plot histogram of pixel values
    # plt.figure(1)
    # plt.hist(img_tiles.ravel())
    # plt.show()


    # calculate mean and variance
    img_mean = np.mean(img_tiles,axis=3)    # img_mean: 10*10*3
    img_mean = np.round(img_mean)
    img_var = np.var(img_tiles,axis=3)      # img_var: 10*10*3

    img_mean = img_mean[:,:,1]
    img_mean =  img_mean.flatten()
    img_mean_sort = np.sort(img_mean)
    img_mean_part = img_mean_sort[0:100]

    img_var = img_var[:,:,1]
    img_var = img_var.flatten()
    img_var_sort = np.sort(img_var)
    img_var_part = img_var_sort[0:100]

    A = np.vstack([img_mean_part,np.ones_like(img_mean_part)]).T   # fit into a line
    m,c = np.linalg.lstsq(A, img_var_part, rcond=None)[0]
    lum_Regress = m * img_mean + c

    # # red
    # img_mean_R = img_mean[:,:,0]
    # img_mean_R =  img_mean_R.flatten()
    # img_var_R = img_var[:,:,0]
    # img_var_R = img_var_R.flatten()
    # A = np.vstack([img_mean,np.ones_like(img_mean_R)]).T   # fit into a line
    # m,c = np.linalg.lstsq(A, img_var_R, rcond=None)[0]
    # lum_Regress_R = m * img_mean_R + c
    #
    # # green
    # img_mean_G = img_mean[:,:,0]
    # img_mean_G =  img_mean_G.flatten()
    # img_var_G = img_var[:,:,0]
    # img_var_G = img_var_G.flatten()
    # A = np.vstack([img_mean,np.ones_like(img_mean_R)]).T   # fit into a line
    # m,c = np.linalg.lstsq(A, img_var_G, rcond=None)[0]
    # lum_Regress_G = m * img_mean_G + c
    #
    # # blue
    # img_mean_B = img_mean[:,:,0]
    # img_mean_B =  img_mean_B.flatten()
    # img_var_B = img_var[:,:,0]
    # img_var_B = img_var_B.flatten()
    # A = np.vstack([img_mean,np.ones_like(img_mean_B)]).T   # fit into a line
    # m,c = np.linalg.lstsq(A, img_var_B, rcond=None)[0]
    # lum_Regress_B = m * img_mean_B + c

    plt.figure(2)
    plt.scatter(img_mean,img_var)
    plt.plot(img_mean,lum_Regress,color="red")

    # plt.plot(img_mean_R,lum_Regress_R,color="red")
    # plt.plot(img_mean_G,lum_Regress_G,color="green")
    # plt.plot(img_mean_B,lum_Regress_B,color="blue")
    plt.show()

    print(m)
    print(c)




