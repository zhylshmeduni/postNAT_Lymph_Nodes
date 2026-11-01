import os
from pathlib import Path
import cv2
import openslide
from openslide import OpenSlide
import numpy as np
from skimage import morphology
import scipy as sp
from typing import Union, Tuple, Iterable
import warnings
import xml.etree.ElementTree as ET
import geojson
import h5py

mag_dict={'.ndpi':'hamamatsu.SourceLens',
          '.svs': 'aperio.AppMag',
          '.mrxs':'mirax.GENERAL.OBJECTIVE_MAGNIFICATION',
          '.tif':'ventana.Magnification',
          '.tiff':'aperio.AppMag',
          'openslide': openslide.PROPERTY_NAME_OBJECTIVE_POWER}

mpp_dict={'.ndpi': 'tiff.XResolution',
          '.svs': 'aperio.MPP',
          '.mrxs': 'mirax.LAYER_0_LEVEL_0_SECTION.MICROMETER_PER_PIXEL_X',
          '.tif': 'openslide.mpp-x',
          '.tiff': 'aperio.MPP',
          'openslide': openslide.PROPERTY_NAME_MPP_X}

def get_mag_mpp(wsi_path: Union[str, Path], mag_dict=mag_dict, mpp_dict=mpp_dict, custom_first=True):
    wsiobj = OpenSlide(wsi_path)

    suffix = os.path.splitext(str(wsi_path))[-1]

    mag_info = {}
    mpp_info = {}
    for source in [suffix, 'openslide']:
        if source in mag_dict.keys() and mag_dict[source] in wsiobj.properties.keys():
            mag_info[source] = int(float(wsiobj.properties[mag_dict[source]]) + 0.5)
        else:
            mag_info[source] = None

        if source in mpp_dict.keys() and mpp_dict[source] in wsiobj.properties.keys():
            if source == '.ndpi':
                mpp_info[source] = 10000 / float(wsiobj.properties[mpp_dict[source]])
            else:
                mpp_info[source] = float(wsiobj.properties[mpp_dict[source]])
        else:
            mpp_info[source] = None

    # 如果放大倍率缺失，但 MPP 存在，则通过 MPP 计算放大倍率
    if mag_info['openslide'] is None and mag_info[suffix] is None:
        if mpp_info['openslide'] is not None or mpp_info[suffix] is not None:
            mpp_source = suffix if mpp_info[suffix] is not None else 'openslide'
            mpp = mpp_info[mpp_source]
            # 通过 MPP 计算放大倍率
            mag = int(10 / mpp + 0.5)  # 10x 对应的 MPP 是 1.0 µm/pixel
            mag_info['calculated'] = mag
            mag_source = 'calculated'
        else:
            raise ValueError(f'Both magnification and MPP properties are missing for {wsi_path}.')
    else:
        # 如果放大倍率存在，则直接使用
        if mag_info[suffix] is not None and mag_info['openslide'] is not None:
            mag_source = suffix if custom_first else 'openslide'
            if np.not_equal(mag_info[suffix], mag_info['openslide']):
                warnings.warn(f'Mag from custom properties and openslide properties is different. Use {mag_source} properties first.')
        else:
            mag_source = suffix if mag_info[suffix] is not None else 'openslide'

    # 确定 MPP 的来源
    if mpp_info[suffix] is not None and mpp_info['openslide'] is not None:
        mpp_source = suffix if custom_first else 'openslide'
        if not np.isclose(mpp_info[suffix], mpp_info['openslide']):
            warnings.warn(f'Mpp from custom properties and openslide properties is different. Use {mpp_source} properties first.')
    else:
        mpp_source = suffix if mpp_info[suffix] is not None else 'openslide'

    mag = mag_info[mag_source]
    mpp = mpp_info[mpp_source]

    # 如果放大倍率是通过 MPP 计算的，则跳过 MPP 到放大倍率的映射检查
    if mag_source != 'calculated':
        d_array = np.abs(mpp - 0.125 * 2 ** np.arange(10))
        close_index = np.argmin(d_array)
        mag_frommap = int(10 / (0.125 * 2 ** np.arange(10)[close_index]))
        if np.not_equal(mag, mag_frommap):
            warnings.warn(f'Mag from mpp is different from mag from properties. Use mag from mpp.')
            mag = mag_frommap

    return mag, mpp

def get_mag_scale_factor(wsi_path: Union[str, Path], read_mag, mag_dict=mag_dict, mpp_dict=mpp_dict):
    wsiobj = OpenSlide(wsi_path)
    level_0_magnification, mpp = get_mag_mpp(wsi_path, mag_dict, mpp_dict)
    level_0_magnification = float(level_0_magnification)
    downsample_level = [round(lv) for lv in wsiobj.level_downsamples]
    magnification_level = [level_0_magnification / lv for lv in downsample_level]

    hires_mag = read_mag
    scale_factor = 1
    if read_mag not in magnification_level:
        if read_mag > magnification_level[0]:
            scale_factor = read_mag / magnification_level[0]
            hires_mag = magnification_level[0]
        else:
            mag_list = np.array(magnification_level)
            mag_list = np.sort(mag_list)[::-1]
            hires_mag = mag_list - read_mag
            # only use higher mag as base for loading
            hires_mag = hires_mag[hires_mag > 0]
            # use the immediate higher to save computation
            hires_mag = mag_list[np.argmin(hires_mag)]
            scale_factor = read_mag / hires_mag

    hires_lv = magnification_level.index(hires_mag)
    downsample = downsample_level[hires_lv]
    return downsample, hires_lv, scale_factor

def get_full_img(wsi:OpenSlide, level, scale_factor=1)->np.ndarray:
    read_size = wsi.level_dimensions[level]
    start_point = (0, 0)

    wsi_img = wsi.read_region(start_point, level, read_size)
    wsi_img = np.array(wsi_img)[..., :3]  # remove alpha channel
    if scale_factor != 1:
        # now rescale then return
        if scale_factor > 1.0:
            interp = cv2.INTER_CUBIC
        else:
            interp = cv2.INTER_LINEAR
        resize_shape = (int(read_size[0] * scale_factor), int(read_size[1] * scale_factor))
        wsi_img = cv2.resize(
            wsi_img, resize_shape, interpolation=interp
        )
    return wsi_img
    
def simple_get_mask(thumb_img_rgb:np.ndarray):
    gray = cv2.cvtColor(thumb_img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    mask = morphology.remove_small_objects(
        mask == 0, min_size=8 * 8, connectivity=2
    )
    mask = morphology.remove_small_holes(mask, area_threshold=64 * 64)
    mask = morphology.binary_dilation(mask, morphology.disk(8))
    return mask

def tile_split(tile_size:Union[int, Iterable],
               img_size:Union[int, Iterable],
               overlap_size:Union[int, Iterable],
               drop=False) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    if isinstance(overlap_size, int):
        overlap_size = (overlap_size, overlap_size)

    tile_left = np.arange(0, img_size[0]-overlap_size[0], tile_size[0]-overlap_size[0])
    tile_top = np.arange(0, img_size[1]-overlap_size[1], tile_size[1]-overlap_size[0])
    tile_x = np.arange(len(tile_left))
    tile_y = np.arange(len(tile_top))

    tile_left, tile_top= np.meshgrid(tile_left, tile_top, indexing='xy')
    tile_x, tile_y = np.meshgrid(tile_x, tile_y, indexing='xy')

    tile_left, tile_top = tile_left.flatten(), tile_top.flatten()
    tile_x, tile_y = tile_x.flatten(), tile_y.flatten()
    
    tile_right = tile_left+tile_size[0]
    tile_bottom = tile_top+tile_size[1]

    tile_bbox_list = np.array([[tile_left, tile_top], [tile_right, tile_bottom]]).transpose(2, 0, 1)
    tile_position_list = np.array([tile_x, tile_y]).transpose(1, 0)
    if drop:
        drop_index = np.logical_or(tile_right>img_size[0], tile_bottom>img_size[1])
        tile_bbox_list = tile_bbox_list[~drop_index]
        tile_position_list = tile_position_list[~drop_index]
    else:
        tile_bbox_list[:, :, 0] = np.minimum(tile_bbox_list[:, :, 0], img_size[0])
        tile_bbox_list[:, :, 1] = np.minimum(tile_bbox_list[:, :, 1], img_size[1])

    tile_bbox_list = tile_bbox_list.astype(int)
    tile_position_list = tile_position_list.astype(int)
    
    return tile_bbox_list, tile_position_list

### utils from FUSCC
def get_mag_from_mpp(mpp, mpp_array=np.array([0.125, 0.250, 0.500, 1.000])):
    d_array = np.abs(mpp-mpp_array)
    close_index = np.argmin(d_array)
    mag = int(10/mpp_array[close_index])
    return mag

def def_slide_info(wsi_path, thumb_mag, till_mag, mag_dict=mag_dict, mpp_dict=mpp_dict):
    #print('slide_mag:', slide_mag, '\n slide mpp:', slide_mpp, '\n slide_mag_from_mpp:', slide_mag_from_mpp)

    # low_level_value, low_level_index, final_low_mag = find_low_level(slide, slide_mag=slide_mag_from_mpp, low_mag=low_mag)
    # tiling_level_value, tiling_level_index, final_tiling_mag = find_save_level(slide, slide_mag=slide_mag_from_mpp, tiling_mag=tiling_mag)
    mag, mpp = get_mag_mpp(wsi_path, mag_dict, mpp_dict)
    thumb_downsample, thumb_level, thumb_factor = get_mag_scale_factor(wsi_path, thumb_mag, mag_dict, mpp_dict)
    till_downsample, till_level, till_factor = get_mag_scale_factor(wsi_path, till_mag, mag_dict, mpp_dict)

    return (thumb_downsample, thumb_level, thumb_factor), (till_downsample, till_level, till_factor), (mag, mpp)

def filter_background(thumb_img_rgb:np.ndarray)->np.ndarray:
    def imfill(img):
        return sp.ndimage.binary_fill_holes(img)

    def imbinarize(img, threshold):
        new_img = np.zeros(img.shape)
        new_img[img >= threshold] = 1
        return new_img

    def invert(img):
        return np.logical_not(img).astype(int)

    def imdilate(x, radius=5):
        mask = morphology.disk(radius)
        x = morphology.dilation(x, footprint=mask)
        return x

    def imerode(x, radius=3):
        mask = morphology.disk(radius)
        x = morphology.binary_erosion(x, footprint=mask)
        return x

    def bwareaopen(bw, p, conn):
        return morphology.remove_small_objects(bw, min_size=p, connectivity=conn)

    radius = 5
    gray = cv2.cvtColor(thumb_img_rgb, cv2.COLOR_BGR2GRAY) #灰度化处理：rgb2bgr then gray
    thresh, tissue_mask = cv2.threshold(gray, 0, 1, cv2.THRESH_OTSU)

    # imfill
    tissue_mask = imfill(invert(tissue_mask))
    # disk --> imdilate
    tissue_mask = imdilate(tissue_mask, 5)
    # disk --> imerode
    tissue_mask = imerode(tissue_mask, 3)
    # imfill
    tissue_mask = imfill(tissue_mask)
    # bwareaopen
    tissue_mask = bwareaopen(tissue_mask, radius * radius, 8)
    return tissue_mask

def get_windows_ImageScope_xml(xml_path):
    polygons = []
    assert os.path.exists(xml_path), f"Can't find {xml_path}"
    tree = ET.parse(xml_path)
    annos = tree.getroot()
    if not annos.tag == 'Annotations':
        raise FileNotFoundError
    for region in annos.iter('Region'):
        if region.get('Type')=='0' or region.get('Type')=='1':
            polygon = []
            for v in region.iter('Vertex'):
                polygon.append([float(v.get('X')), float(v.get('Y'))])
            polygons.append(polygon)
    return polygons

def get_windows_ASAP_xml(xml_path):
    polygons = []
    assert os.path.exists(xml_path), f"Can't find {xml_path}"
    tree = ET.parse(xml_path)
    annos = tree.getroot()
    if not annos.tag == 'ASAP_Annotations':
        raise FileNotFoundError
    for anno in annos.iter('Annotation'):
        if anno.get('Type')=='Rectangle' or anno.get('Type')=='Polygon':
            polygon = []
            for v in anno.iter('Coordinate'):
                polygon.append([float(v.get('X')), float(v.get('Y'))])
            polygons.append(polygon)
    return polygons

def get_windows_QuPath_geojson(geojson_path):
    polygons = []
    with open(geojson_path, 'r') as f:
        data = geojson.load(f)
    for anno in data['features']:
        if anno['geometry']['type'] == 'Polygon':
            polygon = anno['geometry']['coordinates'][0]
            polygons.append(polygon)
    return polygons

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a', chunk_size=32):
    with h5py.File(output_path, mode) as file:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (chunk_size, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val
    return output_path
if __name__ == '__main__':
    wsi = 'C:/Users/Administrator.DESKTOP-JADQ414/Desktop/data_example/FUSCC_Project2/VLM/20241118wsi tile and annotate test/data/2024-13037TIF.tif'
    def_slide_info(wsi, 1.25, 20)