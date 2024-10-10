# Read in images

from skimage import io, color
from skimage.util import img_as_float
import matplotlib.pyplot as plt

im = color.rgb2gray(io.imread("outdoors.jpg"))
im = img_as_float(im) # big matrix of numbers, which tells you in each pixel how much color there is


# Truncate matrixes correspodning to smalles singular values
# 

plt.imshow(im, cmap="gray")

