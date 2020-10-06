import numpy as np
import OpenEXR
import Imath


def readEXR(name):
    """ Read OpenEXR image (both 16-bit and 32-bit datatypes are supported)
    """

    exrFile = OpenEXR.InputFile(name)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)  # convert any datatype to float32 for numpy

    strR = exrFile.channel('R', pt)
    strG = exrFile.channel('G', pt)
    strB = exrFile.channel('B', pt)

    R = np.frombuffer(strR, dtype=np.float32)
    G = np.frombuffer(strG, dtype=np.float32)
    B = np.frombuffer(strB, dtype=np.float32)

    img = np.dstack((R, G, B))

    dw = exrFile.header()['dataWindow']
    sizeEXR = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1, 3)
    img = np.reshape(img, sizeEXR)

    return img