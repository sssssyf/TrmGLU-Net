# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:45:32 2021

@author: liubing
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure,draw 


import torch.nn as nn
import scipy.io as sio
from sklearn.decomposition import PCA
import cv2

input_image_path = './Image/Indian_pines_corrected.mat'
image1 = sio.loadmat(input_image_path)
image1 = image1[list(image1)[-1]]#[:608,:336]
image=image1.reshape(-1,200)
pca=PCA(3)
y=pca.fit(image)
x=y.transform(image)
m,n=x.max(),x.min()
x=np.uint8(255*(x-n)/(m-n))
x=x.reshape(145,145,3)

from skimage.segmentation import slic,mark_boundaries

#segments = slic(x, n_segments=10000, compactness=10, multichannel=True)

#np.save('IP_slic_10000.npy',segments)

i=1100
while(i<=1300):

    n_segments= i

    i+=5

    segments = slic(x, n_segments=n_segments, compactness=10, multichannel=True)

    np.save('.\IP_slic_interval=5\IP_slic_'+str(n_segments)+'.npy',segments)
