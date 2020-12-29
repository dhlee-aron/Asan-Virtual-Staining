import argparse
import os
import glob
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from tifffile import memmap
import matplotlib.pyplot as plt
import openslide
from models import Generator
from skimage.color import rgb2gray
import pandas as pd

def isBG(Img, BG_Thres, BG_Percent):
    Gray_Img = np.uint8(rgb2gray(Img)*255)
    White_Percent = np.mean((Gray_Img > BG_Thres))
    Black_Percent = np.mean((Gray_Img < 255-BG_Thres))

    if Black_Percent > BG_Percent or White_Percent > BG_Percent or Black_Percent+White_Percent>BG_Percent:
        return True
    else:
        return False

def get_region(grid_x, image_w, grid_w, margin_w):
    '''
    Return the base and offset pair to read from the image.
    :param grid_x: grid index on the image
    :param image_w: image width (or height)
    :param grid_w: grid width (or height)
    :param margin: margin width (or height)
    :return: the base index and the width on the image to read
    '''
    image_x = grid_x * grid_w

    image_l = min(image_x, image_w - grid_w)
    image_r = image_l + grid_w - 1

    read_l = max(0, image_l - margin_w)
    read_r = min(image_r + margin_w, image_w - 1)
#    read_l = min(image_x - margin_w, image_w - (grid_w + margin_w))
#    read_r = min(read_l + grid_w + (margin_w << 1), image_w) - 1
#    image_l = max(0,read_l + margin_w)
#    image_r = min(image_l + grid_w , image_w) - 1
    return read_l, image_l, image_r, read_r

def resize_region(im_l, im_r, scale_factor):
    sl = im_l // scale_factor
    sw = (im_r - im_l + 1) // scale_factor
    sr = sl + sw - 1
    return sl, sr

# Testing settings
parser = argparse.ArgumentParser(description='CycleGAN')
parser.add_argument('--input_file_path', default='./data/seegene_rawdata/2019S 031313101010.mrxs', help='input file path')
parser.add_argument('--output_file_path', default='result/', help='output file path')
parser.add_argument('--model_file_path', default='./data/checkpoints/netG_A2B_99.pth', help='result dir')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--nepochs', type=int, default=100, help='saved model of which epochs')
parser.add_argument('--margin', type=int, default=128, help='saved model of which epochs')
parser.add_argument('--local_size', type=int, default=512, help='saved model of which epochs')
parser.add_argument('--img_resize_factor', type=int, default=2, help='saved model of which epochs')
parser.add_argument('--scale_factor', type=int, default=8, help='saved model of which epochs')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")
net_g = Generator(opt.input_nc, opt.output_nc)
if opt.cuda:
    net_g.cuda()
net_g.load_state_dict(torch.load(opt.model_file_path))
net_g.eval()

slide_path = opt.input_file_path
slidename =  os.path.splitext(os.path.basename(opt.input_file_path))[0]
HE_SLIDE = openslide.open_slide(slide_path)
slide_width, slide_height = HE_SLIDE.dimensions

num_w=slide_width//opt.local_size +1
num_h=slide_height//opt.local_size +1

remains_width = slide_width%opt.local_size
remains_height = slide_height%opt.local_size

#ROI_w, ROI_h = ROI_region
result_name = 'pred_{}_{}_{}.tif'.format(slidename,
                    opt.local_size,
                    opt.margin)
result_path = os.path.join(opt.output_file_path,result_name)
# result image memory map
image_file = memmap(result_path,
                    shape=(slide_height//opt.scale_factor,
                           slide_width//opt.scale_factor,3),
                    dtype='uint8',
                    bigtiff=False) #

# get interest tile location
A=np.ones((num_w,num_h)) #
iter_list = [[i[0][0],
              i[0][1]
              ] for i in np.ndenumerate(A)] #
len_itr=len(iter_list)
tsp_map=np.zeros((num_h,num_w))

for itr,[iter_w,iter_h] in enumerate(iter_list):
    l, im_l, im_r, r = get_region(iter_w, slide_width, opt.local_size, opt.margin)
    t, im_t, im_b, b = get_region(iter_h, slide_height, opt.local_size, opt.margin)

    try:
        HE_patch_raw = HE_SLIDE.read_region((l, t), 0, (r-l+1, b-t+1))
        HE_patch_raw = np.array(HE_patch_raw)[:, :,:3]
    except:
        HE_SLIDE = openslide.open_slide(slide_path)
        continue

    if isBG(HE_patch_raw, 240, 0.95) == True:
        continue

    HE_patch_resized = cv2.resize(HE_patch_raw,
                                None,
                                fx=1/opt.img_resize_factor,
                                fy=1/opt.img_resize_factor)

    HE_patch_tensor = transforms.ToTensor()(HE_patch_resized)
    HE_patch_tensor = HE_patch_tensor.view(1,*HE_patch_tensor.shape)

    input_= HE_patch_tensor.to(device).type(torch.float32)
    out = net_g(input_).detach().cpu().numpy().copy()

    out[out>1] = 1
    out[out<0] = 0

    out_t = out.squeeze(0).transpose(1,2,0)
    out_t = (out_t*255).astype('uint8')


    out_t = out_t[(im_t-t)//opt.img_resize_factor:(im_b-t+1)//opt.img_resize_factor,
                    (im_l-l)//opt.img_resize_factor:(im_r-l+1)//opt.img_resize_factor,
                  :]

    HE_re = HE_patch_resized[(im_t-t)//opt.img_resize_factor:(im_b-t+1)//opt.img_resize_factor,
                             (im_l-l)//opt.img_resize_factor:(im_r-l+1)//opt.img_resize_factor,
                            :]

    out_t = cv2.resize(out_t,
                        None,
                        fx=(1/opt.scale_factor * opt.img_resize_factor),
                        fy=(1/opt.scale_factor * opt.img_resize_factor))


    sl, sr = resize_region(im_l, im_r, opt.scale_factor)
    st, sb = resize_region(im_t, im_b, opt.scale_factor)
    image_file[st:sb+1,
               sl:sr+1,:] = out_t

    if itr %100 == 0:
        print('Done {}/{}'.format(itr,len_itr))

image_file.flush()


