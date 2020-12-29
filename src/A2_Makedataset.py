import glob
import os
# os.chdir('/home/dong/Desktop/Placenta/placenta_staining')
import openslide
import numpy as np

import skimage
import skimage.io
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity

from skimage.filters import threshold_local
from scipy import signal
from skimage.color import rgb2gray
from skimage import morphology
np.set_printoptions(threshold=np.inf)
import cv2

def isBG(Img, BG_Thres, BG_Percent):
    if len(Img.shape) > 2:
        Img = rgb2gray(Img)
        Img = np.uint8(Img * 255)
    White_Percent = np.mean((Img > BG_Thres))
    # Black_Percent = np.mean((Img < 255-BG_Thres))
    # if Black_Percent > BG_Percent or White_Percent > BG_Percent or Black_Percent+White_Percent>BG_Percent:
    if White_Percent > BG_Percent:
        return True
    else:
        return False


def PercentBackground(Img,BG_Thres):
    if len(Img.shape) > 2:
        Img = rgb2gray(Img)
        Img = np.uint8(Img * 255)
    White_Percent = np.mean((Img > BG_Thres))
    return White_Percent



import matplotlib.pyplot as plt
import os
import random

wsi_image_abs_path = 'data/WSI_crop_patch/'
dataset_size = 100 #100000
pach_name = 'Train_dataset'

save_dir = os.path.join('data',pach_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    os.mkdir(os.path.join(save_dir,'C4D'))
    os.mkdir(os.path.join(save_dir,'HE'))

file_list=glob.glob(wsi_image_abs_path+'/*C4d*/*.jpg')
rnd_idx = np.random.shuffle(np.arange(len(file_list)))
np.random.shuffle(file_list)
# aaaaaaaaaaaaaaa
idx = 1
cnt = 0
bg = 0
while cnt <= dataset_size: #seegene 53000 , asan 25000
    file_size = os.path.getsize(file_list[idx])
    if file_size<53000 and random.random()>0.95:
        patch_img = plt.imread(file_list[idx])
        resize_patch_img=cv2.resize(patch_img,None,fx=0.5,fy=0.5)
        plt.imsave(os.path.join(save_dir,'C4D',os.path.basename(file_list[idx])),
                   resize_patch_img)
        idx+=1
        cnt+=1
        bg+=1
    elif file_size>=53000:
        patch_img = plt.imread(file_list[idx])
        resize_patch_img=cv2.resize(patch_img,None,fx=0.5,fy=0.5)
        plt.imsave(os.path.join(save_dir,'C4D',os.path.basename(file_list[idx])),
                   resize_patch_img)
        idx+=1
        cnt+=1
    else:
        idx+=1
        pass
    
print('bg count : {}'.format(bg))
print('intr count : {}'.format(cnt))

file_list=glob.glob(os.path.join(wsi_image_abs_path,'*[!C4d]/*.jpg'))
rnd_idx = np.random.shuffle(np.arange(len(file_list)))
np.random.shuffle(file_list)

idx = 0
cnt = 0
while cnt <= dataset_size: #seegene 40000 , asan 25000
    if file_size<40000 and random.random()>0.8:
        patch_img = plt.imread(file_list[idx])
        resize_patch_img=cv2.resize(patch_img,None,fx=0.5,fy=0.5)
        plt.imsave(os.path.join(save_dir,'HE',os.path.basename(file_list[idx])),
                   resize_patch_img)
        idx+=1
        cnt+=1
    elif file_size>=40000:
        patch_img = plt.imread(file_list[idx])
        resize_patch_img=cv2.resize(patch_img,None,fx=0.5,fy=0.5)
        plt.imsave(os.path.join(save_dir,'HE',os.path.basename(file_list[idx])),
                   resize_patch_img)
        idx+=1
        cnt+=1
    else:
        idx+=1
        pass

print('bg count : {}'.format(bg))
print('intr count : {}'.format(cnt))





