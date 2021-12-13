#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 18:35:44 2017

@author: hugo
"""

from pylab import *

from new_libgravimacro import Tiges, methode_Olivier, auto_seuil
import cv2
import pyximport; pyximport.install()
from test_olivier import fast_methode_Olivier
from scipy.stats import circmean
from cmath import rect, phase
from math import radians, degrees
def mean_angle(deg):
    return degrees(phase(sum(rect(1, radians(d)) for d in deg)/len(deg)))

imgp = '/home/hugo/Documents/Boulo/Clinostat_Yasmine/Test_vendredi_31_03/nikon_1/DSC_4625.JPG'

img = cv2.imread(imgp, 0)/255.
#
xystart = ((941,977),(960,1223))
xi, yi= linspace(xystart[0][0], xystart[1][0], dtype='float32'), linspace(xystart[0][1], xystart[1][1],  dtype='float32')
seuil = auto_seuil(img, xi, yi)
#print(seuil)
plot((xystart[0][0],xystart[1][0]),(xystart[0][1],xystart[1][1]),'r-')
imshow(img, 'gray')

data = Tiges(1,1)
#methode_Olivier(img,data,0,0,xi,yi,0.3,100,seuil,rayonfilter=False)

fast_methode_Olivier(img,data,0,0,xi,yi,0.3,100,float(seuil),rayonfilter=False)
plot(data.xc[0,0], data.yc[0,0]);
