import numpy
import matplotlib.pyplot as plt
from skimage import (
    color,io,filters,util
)

tile = numpy.tile(numpy.linspace(0, 1, 255), (255,1))
io.imsave("tile.png",tile)
