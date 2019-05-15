#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:09:20 2019

@author: miguel
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

hdul = fits.open('in.fits')

I = hdul[0].data
I = I[0,0,:,:]

plt.imshow(I)

#F = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(I)))

F = np.fft.fft2(I)

#print(np.abs(F))

plt.imshow(np.abs(F))

plt.show()