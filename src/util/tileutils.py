
from io import BytesIO
import math
import openslide
from PIL import Image
from xml.etree.ElementTree import ElementTree, Element, SubElement

from openslide import open_slide, ImageSlide
#import json
from multiprocessing import Process, JoinableQueue
import os
import re
#from optparse import OptionParser
#import shutil
#import sys
from unicodedata import normalize
import numpy as np
# import skimage
#from PIL import Image
#import gc
#import pprint
#from time import time



class DeepZoomGenerator(object):
    """Generates Deep Zoom tiles and metadata."""

    BOUNDS_OFFSET_PROPS = (openslide.PROPERTY_NAME_BOUNDS_X,
                openslide.PROPERTY_NAME_BOUNDS_Y)
    BOUNDS_SIZE_PROPS = (openslide.PROPERTY_NAME_BOUNDS_WIDTH,
                openslide.PROPERTY_NAME_BOUNDS_HEIGHT)

    def __init__(self, osr, tile_size=254, overlap=1, limit_bounds=False):
        """Create a DeepZoomGenerator wrapping an OpenSlide object.

        osr:          a slide object.
        tile_size:    the width and height of a single tile.  For best viewer
                      performance, tile_size + 2 * overlap should be a power
                      of two.
        overlap:      the number of extra pixels to add to each interior edge
                      of a tile.
        limit_bounds: True to render only the non-empty slide region."""

        # We have four coordinate planes:
        # - Row and column of the tile within the Deep Zoom level (t_)
        # - Pixel coordinates within the Deep Zoom level (z_)
        # - Pixel coordinates within the slide level (l_)
        # - Pixel coordinates within slide level 0 (l0_)

        self._osr = osr
        self._z_t_downsample = tile_size
        self._z_overlap = overlap
        self._limit_bounds = limit_bounds
        self._magnification = osr.properties['openslide.objective-power']

        # Precompute dimensions
        # Slide level and offset
        if limit_bounds:
            # Level 0 coordinate offset
            self._l0_offset = tuple(int(osr.properties.get(prop, 0))
                        for prop in self.BOUNDS_OFFSET_PROPS)
            # Slide level dimensions scale factor in each axis
            size_scale = tuple(int(osr.properties.get(prop, l0_lim)) / l0_lim
                        for prop, l0_lim in zip(self.BOUNDS_SIZE_PROPS,
                        osr.dimensions))
            # Dimensions of active area
            self._l_dimensions = tuple(tuple(int(math.ceil(l_lim * scale))
                        for l_lim, scale in zip(l_size, size_scale))
                        for l_size in osr.level_dimensions)
        else:
            self._l_dimensions = osr.level_dimensions
            self._l0_offset = (0, 0)
        self._l0_dimensions = self._l_dimensions[0]
        # Deep Zoom level
        z_size = self._l0_dimensions
        z_dimensions = [z_size]
        while z_size[0] > 1 or z_size[1] > 1:
            z_size = tuple(max(1, int(math.ceil(z / 2))) for z in z_size)
            z_dimensions.append(z_size)
        self._z_dimensions = tuple(reversed(z_dimensions))
        # Tile
        tiles = lambda z_lim: int(math.ceil(z_lim / self._z_t_downsample))
        self._t_dimensions = tuple((tiles(z_w), tiles(z_h))
                    for z_w, z_h in self._z_dimensions)

        # Deep Zoom level count
        self._dz_levels = len(self._z_dimensions)

        # Total downsamples for each Deep Zoom level
        l0_z_downsamples = tuple(2 ** (self._dz_levels - dz_level - 1)
                    for dz_level in range(self._dz_levels))

        # Preferred slide levels for each Deep Zoom level
        self._slide_from_dz_level = tuple(
                    self._osr.get_best_level_for_downsample(d)
                    for d in l0_z_downsamples)

        # Piecewise downsamples
        self._l0_l_downsamples = self._osr.level_downsamples
        self._l_z_downsamples = tuple(
                    l0_z_downsamples[dz_level] /
                    self._l0_l_downsamples[self._slide_from_dz_level[dz_level]]
                    for dz_level in range(self._dz_levels))

        # Slide background color
        self._bg_color = '#' + self._osr.properties.get(
                        openslide.PROPERTY_NAME_BACKGROUND_COLOR, 'ffffff')

    def __repr__(self):
        return '%s(%r, tile_size=%r, overlap=%r, limit_bounds=%r)' % (
                self.__class__.__name__, self._osr, self._z_t_downsample,
                self._z_overlap, self._limit_bounds)

    @property
    def level_count(self):
        """The number of Deep Zoom levels in the image."""
        return self._dz_levels

    @property
    def level_tiles(self):
        """A list of (tiles_x, tiles_y) tuples for each Deep Zoom level."""
        return self._t_dimensions

    @property
    def level_dimensions(self):
        """A list of (pixels_x, pixels_y) tuples for each Deep Zoom level."""
        return self._z_dimensions

    @property
    def tile_count(self):
        """The total number of Deep Zoom tiles in the image."""
        return sum(t_cols * t_rows for t_cols, t_rows in self._t_dimensions)

    def get_tile(self, level, address):
        """Return an RGB PIL.Image for a tile.

        level:     the Deep Zoom level.
        address:   the address of the tile within the level as a (col, row)
                   tuple."""

        # Read tile
        args, z_size, I0_location = self._get_tile_info(level, address)
        tile = self._osr.read_region(*args)

        # Apply on solid background
        bg = Image.new('RGB', tile.size, self._bg_color)
        tile = Image.composite(tile, bg, tile)

        # Scale to the correct size
        if tile.size != z_size:
            tile.thumbnail(z_size, Image.ANTIALIAS)

        return tile, I0_location

    def _get_tile_info(self, dz_level, t_location):
        # Check parameters
        if dz_level < 0 or dz_level >= self._dz_levels:
            raise ValueError("Invalid level")
        for t, t_lim in zip(t_location, self._t_dimensions[dz_level]):
            if t < 0 or t >= t_lim:
                raise ValueError("Invalid address")

        # Get preferred slide level
        slide_level = self._slide_from_dz_level[dz_level]

        # Calculate top/left and bottom/right overlap
        z_overlap_tl = tuple(self._z_overlap * int(t != 0)
                    for t in t_location)
        z_overlap_br = tuple(self._z_overlap * int(t != t_lim - 1)
                    for t, t_lim in
                    zip(t_location, self.level_tiles[dz_level]))

        # Get final size of the tile
        z_size = tuple(min(self._z_t_downsample,
                    z_lim - self._z_t_downsample * t) + z_tl + z_br
                    for t, z_lim, z_tl, z_br in
                    zip(t_location, self._z_dimensions[dz_level],
                    z_overlap_tl, z_overlap_br))

        # Obtain the region coordinates
        z_location = [self._z_from_t(t) for t in t_location]
        l_location = [self._l_from_z(dz_level, z - z_tl)
                    for z, z_tl in zip(z_location, z_overlap_tl)]
        # Round location down and size up, and add offset of active area
        l0_location = tuple(int(self._l0_from_l(slide_level, l) + l0_off)  ########### corret location, no limit bound
                    for l, l0_off in zip(l_location, self._l0_offset))
        l_size = tuple(int(min(math.ceil(self._l_from_z(dz_level, dz)),
                    l_lim - math.ceil(l)))
                    for l, dz, l_lim in
                    zip(l_location, z_size, self._l_dimensions[slide_level]))

        # Return read_region() parameters plus tile size for final scaling
        return ((l0_location, slide_level, l_size), z_size, l0_location)

    def _l0_from_l(self, slide_level, l):
        return self._l0_l_downsamples[slide_level] * l

    def _l_from_z(self, dz_level, z):
        return self._l_z_downsamples[dz_level] * z

    def _z_from_t(self, t):
        return self._z_t_downsample * t

    def get_tile_coordinates(self, level, address):
        """Return the OpenSlide.read_region() arguments for the specified tile.

        Most users should call get_tile() rather than calling
        OpenSlide.read_region() directly.

        level:     the Deep Zoom level.
        address:   the address of the tile within the level as a (col, row)
                   tuple."""
        return self._get_tile_info(level, address)[0]

    def get_tile_dimensions(self, level, address):
        """Return a (pixels_x, pixels_y) tuple for the specified tile.

        level:     the Deep Zoom level.
        address:   the address of the tile within the level as a (col, row)
                   tuple."""
        return self._get_tile_info(level, address)[1]

    def get_dzi(self, format):
        """Return a string containing the XML metadata for the .dzi file.

        format:    the format of the individual tiles ('png' or 'jpeg')"""
        image = Element('Image', TileSize=str(self._z_t_downsample),
                        Overlap=str(self._z_overlap), Format=format,
                        xmlns='http://schemas.microsoft.com/deepzoom/2008')
        w, h = self._l0_dimensions
        SubElement(image, 'Size', Width=str(w), Height=str(h))
        tree = ElementTree(element=image)
        buf = BytesIO()
        tree.write(buf, encoding='UTF-8')
        return buf.getvalue().decode('UTF-8')


def isBG(Img, BG_Thres, BG_Percent):
    # Gray_Img = np.uint8(skimage.color.rgb2gray(Img)*255)
    Gray_Img = np.array(Img.convert('L'))
    # plt.imshow(Gray_Img,'gray')
    # plt.show()
    White_Percent = np.mean((Gray_Img > BG_Thres))
    Black_Percent = np.mean((Gray_Img < 255-BG_Thres))

    if Black_Percent > BG_Percent or White_Percent > BG_Percent or Black_Percent+White_Percent>BG_Percent:
        return True
    else:
        return False
#    if Black_Percent > BG_Percent or White_Percent > BG_Percent or Black_Percent+White_Percent>BG_Percent:
#        return False
#    else:
#        return True

class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,
                quality, BG_Thres, BG_Percent):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._slide = None
        self._BG_Thres = BG_Thres
        self._BG_Percent = BG_Percent


    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            try:
                data = self._queue.get()
                if data is None:
                    self._queue.task_done()
                    break
                associated, level, address, tiledir, format = data
                if last_associated != associated:
                    dz = self._get_dz(associated)
                    last_associated = associated
                tile, I0_location = dz.get_tile(level, address)

                image_tile_size = int(self._overlap*2+self._tile_size)
                # tile size not satified
                if tile.size[0] != image_tile_size or tile.size[1] !=image_tile_size:
                    self._queue.task_done()         ############
                    continue
                # if is BackGround
                if isBG(tile, self._BG_Thres, self._BG_Percent):
                    self._queue.task_done()          ##############
                    continue

                tilename = '{}_{}_{}_'.format(I0_location[0],I0_location[1],dz._magnification)
                tilename = tilename + os.path.basename(tiledir) + format
                tile_path = os.path.join(tiledir, tilename)
                tile.save(tile_path, quality=self._quality)           ######################
                tile.close()
                self._queue.task_done()
            except:
                pass
    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, OutputDir, format, associated, queue):
        self._dz = dz
        self._OutputDir = OutputDir
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0

    def run(self):
        self._write_tiles()

    def _write_tiles(self):
        print(self._dz.level_tiles)
        level = self._dz.level_count-1    ################### asan .mrxs ->  -2
        slide_fn = os.path.basename(self._dz._osr._filename)
        slide_fn = slide_fn[: slide_fn.rfind('.')]
        tiledir = os.path.join(self._OutputDir, slide_fn)

        if not os.path.exists(tiledir):
            os.makedirs(tiledir)
        cols, rows = self._dz.level_tiles[level]
        print(rows, cols)
        for row in range(rows):
            # print('row Count',row)
            for col in range(cols):

                self._queue.put((self._associated, level, (col, row), tiledir, self._format))
                self._tile_done()

    def _tile_done(self):
        self._processed += 1
        count = self._processed
        cols, rows = self._dz.level_tiles[self._dz.level_count-1]
        total = cols*rows
        # print('Tiling Count: ',count)

        if count % 10000 == 0 or count == total:
            print("Tiling Count: wrote %d/%d tiles" % (count, total))
            if count == total:
                print('\nFile Done: ',self._dz._osr._filename)


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, OutputDir, format, tile_size, overlap,
                limit_bounds, quality, workers,BG_Thres,BG_Percent):

        self._slide = open_slide(slidepath)
        self._OutputDir = OutputDir
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._BG_Thres = BG_Thres
        self._BG_Percent = BG_Percent
        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                        limit_bounds, quality, BG_Thres, BG_Percent).start()

    def run(self):
        self._run_image()
        # a=1
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            OutputDir = self._OutputDir
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            OutputDir = os.path.join(self._OutputDir, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)
        tiler = DeepZoomImageTiler(dz, OutputDir, self._format, associated,
                    self._queue)
        tiler.run()

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)


def CallRemoveFiles(file_list,thr=200000):

    fs=np.zeros((len(file_list),3),dtype='object')
    count=0

    for itr, file_path in enumerate(file_list):
        file_size=os.stat(file_path).st_size

        fs[itr,0]=file_path
        fs[itr,1]=file_size

        if file_size < thr: #512 -> 60000,  1024 -> 200000
            count+=1
            fs[itr,2]=1

        if itr % 10000 == 0 :
            print(itr)

    return fs, count