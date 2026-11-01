import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path, WindowsPath
from typing import Tuple, Dict, Union, Set, Optional, Iterable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import openslide
from openslide import OpenSlide
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision import transforms
import cv2

from transformers import AutoModel
import timm

from utils.wsi_utils_strict import (def_slide_info, get_full_img, filter_background, simple_get_mask,
                                    get_windows_ImageScope_xml, get_windows_ASAP_xml, get_windows_QuPath_geojson,
                                    mag_dict, mpp_dict, get_mag_mpp,
                                    tile_split, save_hdf5)

try:
    from timm.layers import SwiGLUPack as SwiGLU
except ImportError:
    try:
        from timm.layers import SwiGLUPacked as SwiGLU
    except ImportError:
        try:
            from timm.layers import SwiGLU
        except ImportError:
            class SwiGLU(nn.Module):
                def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                             bias=True, **kwargs):
                    super().__init__()
                    out_features = out_features or in_features
                    hidden_features = hidden_features or in_features
                    self.fc1 = nn.Linear(in_features, hidden_features * 2, bias=bias)
                    self.act = act_layer()
                    self.drop = nn.Dropout(drop)
                    self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

                def forward(self, x):
                    x = self.fc1(x)
                    x1, x2 = x.chunk(2, dim=-1)
                    x = self.act(x1) * x2
                    x = self.drop(x)
                    x = self.fc2(x)
                    x = self.drop(x)
                    return x


class MultiWorkersWsi_dataset(Dataset):
    def __init__(self, wsi_path: Union[str, Path], patch_size=256, overlap=0,
                 tile_mag=10, thumb_mag=1.25, xml_path=None,
                 dropR=0.75, target_mpp: Optional[float] = None,
                 transform=None, thumb_dir: Optional[Union[str, Path]] = None,
                 save_thumb_png: bool = True,
                 cache: Optional[str] = None):
        '''
        Important: It is forbidden to build an instance of the OpenSlide type as an attribute in the __init__ of a class,
        otherwise the same C pointer will be copied during multi-process initialisation resulting in an error!
        '''
        if isinstance(wsi_path, str):
            wsi_path = Path(wsi_path)
        self.name = wsi_path.stem
        self.wsi_path = wsi_path
        self.wsi = OpenSlide(wsi_path)
        self.patch_size = patch_size
        self.raw_patch_size = self.patch_size
        self.overlap = overlap
        self.tile_mag = tile_mag
        self.thumb_mag = thumb_mag
        self.xml_path = xml_path
        self.dropR = dropR
        self.transform = transform
        self.save_thumb_png = save_thumb_png
        if isinstance(thumb_dir, str):
            thumb_dir = Path(thumb_dir)
            if self.save_thumb_png:
                thumb_dir.mkdir(parents=True, exist_ok=True)
        self.thumb_dir = thumb_dir
        self.cache = cache

        thumb_info, tile_info, slide_info = def_slide_info(wsi_path, thumb_mag, tile_mag)
        self.tile_downsample, self.tile_level, self.tile_factor = tile_info
        self.thumb_downsample, self.thumb_level, self.thumb_factor = thumb_info
        self.slide_mag, self.slide_mpp = slide_info

        if target_mpp is not None:
            self.target_mpp = target_mpp
        else:
            self.target_mpp = self.slide_mpp

        if self.tile_factor == 1.0:
            self.target_mpp = self.target_mpp / self.tile_factor
            self.patch_size = int(self.raw_patch_size * self.target_mpp / self.slide_mpp)
        else:
            self.target_mpp = self.target_mpp / self.tile_factor
            self.patch_size = int(self.raw_patch_size * self.target_mpp / 0.25)

        if self.transform is None:
            self.transform = v2.Compose([v2.ToImage(),
                                         v2.ToDtype(torch.float32, scale=True)])
            self.add_resize_transform()

        image_shape = self.wsi.level_dimensions[self.tile_level]
        self.image_shape = np.array(image_shape)

        self.get_windows()

        self.get_thumb()
        self.get_mask()

        self.get_patch_info()
        self.filter_patch()
        self.n = len(self.patch_bbox_list)

        if self.save_thumb_png and (self.thumb_dir is not None):
            self.save_thumb()

    def add_resize_transform(self):
        if any(isinstance(t, (v2.Resize, transforms.Resize)) for t in self.transform.transforms):
            pass
        else:
            self.transform = v2.Compose([v2.Resize(self.raw_patch_size),
                                         *self.transform.transforms])

    def get_windows(self):
        self.polygons = None
        self.polygon_mask = None
        if self.xml_path is not None:
            get_windows_func = [get_windows_ImageScope_xml,
                                get_windows_ASAP_xml,
                                get_windows_QuPath_geojson]
            for get_windows in get_windows_func:
                try:
                    self.polygons = get_windows(self.xml_path)
                except AssertionError as E:
                    raise OSError(f"Can't find {self.xml_path}")
                except:
                    pass
                else:
                    print(f'Get {len(self.polygons)} windows from \'{Path(self.xml_path).name}\'')
                    break
            else:
                print(f'Failed to get windows but get xml_path \'{Path(self.xml_path).name}\', ',
                      'please check the `get_windows_func` and annotation file')

    def get_patch_info(self):
        self.patch_bbox_list, self.patch_coord_list = tile_split(self.patch_size, self.image_shape, self.overlap,
                                                                 drop=True)
        self.raw_patch_bbox_list, self.raw_patch_coord_list = tile_split(self.raw_patch_size, self.image_shape,
                                                                         self.overlap, drop=True)

    def get_mask(self):
        tissue_seg_func = filter_background
        self.tissue_mask = tissue_seg_func(self.thumb)

    def get_thumb(self):
        self.thumb = get_full_img(self.wsi, self.thumb_level, self.thumb_factor)

    def filter_patch(self):
        if not hasattr(self, 'tissue_mask'):
            raise ValueError('There is no mask, please get mask first')
        if not hasattr(self, 'patch_bbox_list'):
            raise ValueError('There is no patch_bbox_list, please get patch_bbox_list first')

        if self.polygons is not None:
            self.polygon_mask = np.zeros_like(self.tissue_mask, dtype=np.uint8)
            ratio_s2t = self.slide_mag / self.thumb_mag
            polygons = [(np.array(polygon) / ratio_s2t).astype(int) for polygon in self.polygons]
            cv2.drawContours(self.polygon_mask, polygons, -1, 255, -1)
            self.polygon_mask = self.polygon_mask > 0
            self.mask = self.tissue_mask * self.polygon_mask
        else:
            self.mask = self.tissue_mask

        patch_filter_index = []
        ratio_p2t = (self.tile_mag / self.tile_factor) / self.thumb_mag
        patch_bbox_list = self.patch_bbox_list.copy()
        patch_bbox_list = np.round(patch_bbox_list / ratio_p2t).astype(int)
        for i, patch in enumerate(patch_bbox_list):
            patch_mask = self.mask[patch[0, 1]:patch[1, 1], patch[0, 0]:patch[1, 0]]
            patch_tissue_ratio = np.sum(patch_mask) / patch_mask.size
            if patch_tissue_ratio > self.dropR:
                patch_filter_index.append(i)

        self.patch_bbox_list = self.patch_bbox_list[patch_filter_index]
        self.patch_coord_list = self.patch_coord_list[patch_filter_index]

    def save_thumb(self):
        thumb_with_mask = self.thumb.copy().astype(np.uint8)

        contours_tissue, _ = cv2.findContours(self.tissue_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(thumb_with_mask, contours_tissue, -1, (0, 255, 0), 3)

        thumb_with_mask[~self.mask] = 0.5 * thumb_with_mask[~self.mask]

        if self.polygons is not None:
            ratio_s2t = self.slide_mag / self.thumb_mag
            polygons = [(np.array(polygon) / ratio_s2t).astype(int) for polygon in self.polygons]
            cv2.drawContours(thumb_with_mask, polygons, -1, (255, 0, 0), 5)

        contours_tissue, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(thumb_with_mask, contours_tissue, -1, (0, 255, 0), 3)

        Image.fromarray(self.thumb).save(self.thumb_dir / (self.wsi_path.stem + '_original.png'))
        Image.fromarray(thumb_with_mask).save(self.thumb_dir / (self.wsi_path.stem + '_segmented.png'))

    def __len__(self):
        return self.n

    def __restart_wsi(self):
        self.wsi = OpenSlide(self.wsi_path)

    def read_region(self, bbox, level) -> torch.Tensor:
        if not hasattr(self, 'wsi'):
            self.__restart_wsi()
        img = self.wsi.read_region(bbox[0] * int(self.tile_downsample + 0.5), level, (bbox[1] - bbox[0])).convert('RGB')
        return img

    def __getitem__(self, idx):
        img = self.read_region(self.patch_bbox_list[idx], self.tile_level)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.patch_coord_list[idx]

    def save_cache(self):
        cache = self.__dict__.copy()
        cache_path = Path(self.cache)
        unused_keys = ['wsi', 'mask', 'tissue_mask', 'thumb', 'polygon_mask']
        for k in unused_keys:
            if k in cache:
                cache.pop(k)

        cache_path.unlink(missing_ok=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)

        dataset_keys = [i for i in cache.keys()]
        for key in dataset_keys:
            if key not in ['transform', 'wsi_path', 'tile_level', 'n', 'cache']:
                cache.pop(key)
        return cache

    def load_cache(self):
        cache_path = Path(self.cache)
        cache = torch.load(cache_path, weights_only=False)
        return cache

    def __getstate__(self):
        if 'wsi' in self.__dict__:
            self.__dict__.pop('wsi')

        if self.cache is not None:
            state = self.save_cache()
        else:
            state = self.__dict__.copy()

        return state

    def __setstate__(self, state):
        state['wsi'] = OpenSlide(state['wsi_path'])
        self.__dict__.update(state)
        if state['cache'] is not None:
            cache = self.load_cache()
            self.__dict__.update(cache)


if __name__ == '__main__':
    patch_size = 256
    target_mpp = 0.23
    tile_mag = 10
    model_use = 'Virchow2'
    save_thumb_png = False
    world_size = 1
    rank = 0

    data_dir = Path(r'D:\New_Nodes')
    output_dir = Path(r'D:\Digital_Pathology_Resources\Feats\Macrosection_Project\Lymph_Feats\Virchow2')
    thumb_dir = Path(r'D:\Digital_Pathology_Resources\Masks\Macrosection_Project\Lymph_Masks')

    file = data_dir.rglob('*')
    file_wsi = [f for f in file if f.suffix in ['.svs', '.ndpi', '.tiff', '.tif']]

    ## Load model
    assert model_use in ['CONCH1.5', 'UNI', 'Virchow2', 'UNI2-h']
    match model_use:
        case 'UNI':
            model = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0,
                dynamic_img_size=True,
                checkpoint_path='./UNI/pytorch_model.bin'
            )
            model.load_state_dict(
                torch.load(os.path.join('./UNI', "pytorch_model.bin"), map_location="cpu", weights_only=True),
                strict=True)
        case 'CONCH1.5':
            titan = AutoModel.from_pretrained('./TITAN', trust_remote_code=True, local_files_only=True)
            model, _ = titan.return_conch()
        case 'Virchow2':
            model_dir = "D:\\Python_project\\FM_feature_v3\\Virchow2"
            try:
                model = timm.create_model(
                    f"local-dir:{model_dir}",
                    pretrained=True,
                    mlp_layer=SwiGLU,
                    act_layer=torch.nn.SiLU,
                    num_classes=0,
                )
            except Exception:
                model = timm.create_model(
                    "vit_huge_patch14_224",
                    img_size=224,
                    patch_size=14,
                    embed_dim=1280,
                    depth=32,
                    num_heads=16,
                    mlp_ratio=3416 / 1280,
                    reg_tokens=4,
                    mlp_layer=SwiGLU,
                    act_layer=torch.nn.SiLU,
                    num_classes=0,
                )
                bin_path = os.path.join(model_dir, "pytorch_model.bin")
                safe_path = os.path.join(model_dir, "model.safetensors")

                if os.path.exists(safe_path):
                    from safetensors.torch import load_file
                    state_dict = load_file(safe_path)
                elif os.path.exists(bin_path):
                    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
                else:
                    raise FileNotFoundError(f"Missing weight files in {model_dir}")

                model.load_state_dict(state_dict, strict=True)
        case 'UNI2-h':
            model_dir = "D:\\Python_project\\FM_feature_v3\\UNI2-h"

            try:
                model = timm.create_model(
                    f"local-dir:{model_dir}",
                    pretrained=True,
                    num_classes=0,
                    dynamic_img_size=True,
                )
            except Exception:
                model = timm.create_model(
                    "vit_giant_patch14_224",
                    img_size=224,
                    patch_size=14,
                    embed_dim=1536,
                    depth=24,
                    num_heads=24,
                    init_values=1e-5,
                    mlp_ratio=8192 / 1536,
                    reg_tokens=8,
                    no_embed_class=True,
                    mlp_layer=SwiGLU,
                    act_layer=torch.nn.SiLU,
                    num_classes=0,
                    dynamic_img_size=True,
                )
                bin_path = os.path.join(model_dir, "pytorch_model.bin")
                safe_path = os.path.join(model_dir, "model.safetensors")

                if os.path.exists(safe_path):
                    from safetensors.torch import load_file

                    state_dict = load_file(safe_path)
                elif os.path.exists(bin_path):
                    state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
                else:
                    raise FileNotFoundError(f"Missing weight files in {model_dir}")

                if "model" in state_dict:
                    state_dict = state_dict["model"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

                model.load_state_dict(state_dict, strict=True)

    UNI2_TRANSFORM = v2.Compose(
        [
            v2.Resize(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    for wsi_path in tqdm(file_wsi, position=0):
        xml_path = None
        output_path = output_dir / (wsi_path.relative_to(data_dir).with_suffix('.h5'))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        thumb_path = thumb_dir / (wsi_path.relative_to(data_dir).parent)
        if save_thumb_png:
            thumb_path.mkdir(parents=True, exist_ok=True)

        dataset = MultiWorkersWsi_dataset(wsi_path, xml_path=xml_path, patch_size=patch_size, tile_mag=tile_mag,
                                          thumb_mag=1.25,
                                          target_mpp=target_mpp,
                                          thumb_dir=thumb_path if save_thumb_png else None,
                                          save_thumb_png=save_thumb_png,
                                          transform=UNI2_TRANSFORM,
                                          dropR=0.1875,
                                          cache=f'./.cache/.cache_{rank}.pth')
        loader = DataLoader(dataset, batch_size=256, num_workers=8)

        model = model.eval().cuda()
        mode = 'w'
        assert_dict = {'features': [], 'coords': []}
        for img, coord in tqdm(loader, position=1, leave=False):
            with torch.inference_mode():
                if model_use == 'Virchow2':
                    output = model.forward_features(img.float().cuda())
                    class_token = output[:, 0]
                    patch_tokens = output[:, 5:]
                    feature = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
                elif model_use == 'UNI2-h':
                    feature = model(img.float().cuda())
                else:
                    feature = model(img.float().cuda())

                feature = feature.cpu().numpy().astype(np.float32)
                assert_dict['features'].append(feature)
                assert_dict['coords'].append(coord.numpy().astype(np.int32))
        assert_dict['features'] = np.concatenate(assert_dict['features'], axis=0)
        assert_dict['coords'] = np.concatenate(assert_dict['coords'], axis=0)
        save_hdf5(output_path, assert_dict, attr_dict=None, mode=mode)