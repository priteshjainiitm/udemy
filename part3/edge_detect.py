# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:15:00 2017

@author: pj
"""

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img =  mpimg.imread('lena.png')
bw = img.mean(axis =2)

Hx = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1],], dtype = np.float32)
    
Hy = Hx.T

#detect horizontal edges
Gx = convolve2d(bw, Hx)
plt.imshow(Gx, cmap = 'gray')

#vertical
Gy = convolve2d(bw, Hy)
plt.imshow(Gy, cmap = 'gray')

#Gx and Gy are vectors, magnitude is

G = np.sqrt(Gx*Gx + Gy*Gy)
plt.imshow(G, cmap = 'gray')

theta = np.arctan2(Gy, Gx)
plt.imshow(theta, cmap='gray')