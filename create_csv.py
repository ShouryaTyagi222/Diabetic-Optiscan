import os
from tqdm import tqdm
import cv2 
import pywt
import numpy as np


def w2d(img, mode='haar', level=1):
    imArray = img
    
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    imArray =  np.float32(imArray)
    imArray /= 255
    
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)

    imArray_H = cv2.resize(imArray_H, (224, 224))

    return imArray_H

data = []

train_path = '/teamspace/studios/this_studio/diabetic_retinopathy/OCT diabetic images/train'
for i in os.listdir(train_path):
    folder = train_path + '/' + i
    for j in tqdm(os.listdir(folder)):
        img = cv2.imread(folder + '/' + j)
        img = w2d(img, 'rbio1.1',3)
        cv2.imwrite(new+'/'+j, img)
        data.append([folder + '/' + j,i])

import pandas as pd
df = pd.DataFrame(data, columns=['image', 'label'])
df.to_csv('funds_train_data.csv')


data = []

test_path = '/teamspace/studios/this_studio/diabetic_retinopathy/OCT diabetic images/val'
for i in os.listdir(test_path):
    folder = test_path + '/' + i
    for j in tqdm(os.listdir(folder)):
#     img = cv2.imread(folder + '/' + j)
#     img = w2d(img, 'rbio1.1',3)
#     cv2.imwrite(new+'/'+j, img)
        data.append([folder + '/' + j,i])

df = pd.DataFrame(data, columns=['image', 'label'])
df.to_csv('funds_test_data.csv')