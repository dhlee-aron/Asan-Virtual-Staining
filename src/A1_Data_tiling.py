import os
os.chdir('/home/dong/Development/Asan-Virtual-Staining')

import glob
import numpy as np
from tileutils import DeepZoomStaticTiler, CallRemoveFiles
#from WSI_Config import Placenta_Metastasis

WSI_Slide_Dir = 'data/seegene_rawdata'  # Config.WSI_Slide_Dir
Tile_Save_Dir = 'WSI_crop_patch'# Config.WSI_Tiled_Dirs

limit_bounds = True
overlap = 0
tile_size = 512
workers= 4
BG_Thres = 240
BG_Percent = 0.8
Tile_Img_Format= '.jpg'
quality = 90

# calculate value of tile_size and overlap in deepzoom format
if overlap ==0:
    tile_size_deepzoom = tile_size
    overlap_deepzoom = 0
else:
    tile_size_deepzoom = overlap
    overlap_deepzoom = (tile_size-overlap)/2

data_list=glob.glob('{}/*.mrxs'.format(WSI_Slide_Dir))
np.random.shuffle(data_list)
data_list = data_list
for img_path in data_list:
    file_name = os.path.splitext(os.path.split(img_path)[-1])[0]
    save_dir  = os.path.join('data',Tile_Save_Dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    DeepZoomStaticTiler(img_path, save_dir, Tile_Img_Format,
                        tile_size_deepzoom, overlap_deepzoom, limit_bounds, quality,
                        workers,BG_Thres,BG_Percent).run()

