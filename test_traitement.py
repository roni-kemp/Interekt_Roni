#!/usr/bin/env python
# coding: utf-8

from new_libgravimacro import ProcessImages
from RootStemExtractor import get_growth_length
from glob import glob

from matplotlib import pyplot as plt 


imgs = sorted(glob('./tests/Test Arabido P30+600 20180627/Nikon1/*.JPG'))
print(imgs)


plot_imgs = False
if plot_imgs:
    im = plt.imread(imgs[0])
    plt.imshow(im, 'gray')
    plt.show()

#Définition des points du trait de base 
XB = [4000, 4000]
YB = [1300, 1600]

PT1 = (XB[0], YB[0])
PT2 = (XB[1], YB[1])

BASE = [(PT1, PT2)]

output, img_names, base = ProcessImages(imgs, num_images=[0, 5],
                                        num_tiges=1, base_points=BASE)

# print(output)
data = output['tiges_data']

get_growth_length(data, 0, imgs=imgs[:10])
plt.show()
