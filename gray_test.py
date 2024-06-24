import numpy as np # type: ignore
from skimage.io import imread
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pylab as plt


img = imread('img_gray.jpg')
img = rgb2lab(img)