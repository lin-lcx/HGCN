import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import openslide
from openslide.deepzoom import DeepZoomGenerator
import imageio

from torch.autograd import Variable
import torchvision.models as models
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
from random import sample
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import joblib
# from scipy.spatial.distance import pdist
import json
from torch_geometric.data import Data
import random

import torchvision.models as models
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(1)

class fully_connected(nn.Module):
    """docstring for BottleNeck"""
    def __init__(self, model, num_ftrs, num_classes):
        super(fully_connected, self).__init__()
        self.model = model
        self.fc_4 = nn.Linear(num_ftrs,num_classes)

    def forward(self, x):
        x = self.model(x) 
        x = torch.flatten(x, 1)
        out_1 = x
        out_3 = self.fc_4(x)
        return  out_1, out_3


import cv2
import json
import argparse
import openslide
import numpy as np
import gc
from pathlib import Path
from utils.filters import adaptive, otsu, RGB_filter
from utils.general import get_three_points, keep_patch, out_of_bound
import joblib

def tiling(slide_filepath, magnification, patch_size, scale_factor=32, tissue_thresh=0.35, method='adaptive', overview_level=5,
           coord_dir=None, overview_dir=None, mask_dir=None, patch_dir=None, filename=None, model_final=None,thumbnail_dir=None):
    feature={}
    transform=transforms.Compose([transforms.ToTensor()])
    slide = openslide.open_slide(str(slide_filepath))
    # for k in slide.properties.keys():
    #     print(f"{k}: {slide.properties[k]}")
    # print(f"properties keys: \n{slide.properties.keys()}")
    # print(f"properties: \n{slide.properties}")
    if 'aperio.AppMag' in slide.properties.keys():
        if slide.properties['aperio.AppMag'] == '40.000000':
            level0_magnification = int(float(slide.properties['aperio.AppMag']))
        else:
            level0_magnification = int(slide.properties['aperio.AppMag'])
    elif 'openslide.mpp-x' in slide.properties.keys():
        level0_magnification = 40 if int(np.floor(float(slide.properties['openslide.mpp-x']) * 10)) == 2 else 20
    else:
        level0_magnification = 40

    if level0_magnification < magnification:
        print(f"{level0_magnification}<{magnification}? magnification should <= level0_magnification.")
        return
    patch_size_level0 = int(patch_size * (level0_magnification / magnification))

    
    downsamples = slide.level_downsamples
    overview_level = len(downsamples)-1
     
        
        
    if overview_dir is not None:
        thumbnail = slide.get_thumbnail(slide.level_dimensions[overview_level]).convert('RGB')
        thumbnail = cv2.cvtColor(np.asarray(thumbnail), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(thumbnail_dir / f'{filename}.png'), thumbnail)
    else:
        thumbnail = None

    if patch_dir is not None:
        patch_dir = patch_dir / filename
        patch_dir.mkdir(parents=True, exist_ok=True)

    # Mask
    mask_filepath = str(mask_dir / f'{filename}.png') if mask_dir is not None else None
    if method == 'adaptive':
        mask, color_bg = adaptive(slide, mask_downsample=scale_factor, mask_filepath=mask_filepath)
    elif method == 'otsu':
        mask, color_bg = otsu(slide, mask_downsample=scale_factor, mask_filepath=mask_filepath)
    elif method == 'rgb':
        mask, color_bg = RGB_filter(slide, mask_downsample=scale_factor, mask_filepath=mask_filepath)
    else:
        raise ValueError(f"filter method is wrong, {method}. ")
    mask_w, mask_h = mask.size
    mask = cv2.cvtColor(np.asarray(mask), cv2.COLOR_GRAY2BGR)
    mask_patch_size = int(((patch_size_level0 // scale_factor) * 2 + 1) // 2)
    num_step_x = int(mask_w // mask_patch_size)
    num_step_y = int(mask_h // mask_patch_size)

    coord_list = []
    print(f"Processing {filename}...")
    i=0
    for row in range(num_step_y):
        for col in range(num_step_x):
            points_mask = get_three_points(col, row, mask_patch_size)
            row_start, row_end = points_mask[0][1], points_mask[1][1]
            col_start, col_end = points_mask[0][0], points_mask[1][0]
            patch_mask = mask[row_start:row_end, col_start:col_end]
            if keep_patch(patch_mask, tissue_thresh, color_bg):
                points_level0 = get_three_points(col, row, patch_size_level0)
                if out_of_bound(slide.dimensions[0], slide.dimensions[1], points_level0[1][0], points_level0[1][1]):
                    continue
                coord_list.append({'row': row, 'col': col, 'x': points_level0[0][0], 'y': points_level0[0][1]})
                if overview_dir is not None:
                    points_thumbnail = get_three_points(col, row, patch_size_level0 / slide.level_downsamples[overview_level])
                    cv2.rectangle(thumbnail, points_thumbnail[0], points_thumbnail[1], color=(255, 255, 255), thickness=3)
                if patch_dir is not None:
                    patch_level0 = slide.read_region(location=points_level0[0], level=0,
                                                     size=(patch_size_level0, patch_size_level0)).convert('RGB')
                    patch = patch_level0.resize(size=(patch_size, patch_size))
                    
#                     print(f'{row}_{col}.png')
                    patch.save(str(patch_dir / f'{row}_{col}.png'))

                    image = transform(patch).unsqueeze(0)
                    inputs = Variable(image).to(device)
                    x,_= model_final(inputs)
                    feature[str(row)+'_'+str(col)]=x.cpu().detach().numpy().squeeze() 
                    i+=1
                    print('\r' + str(i), end='', flush=True)
                    
    # Save
    coord_dict = {
        'slide_filepath': str(slide_filepath),
        'magnification': magnification,
        'magnification_level0': level0_magnification,
        'num_row': num_step_y,
        'num_col': num_step_x,
        'patch_size': patch_size,
        'patch_size_level0': patch_size_level0,
        'num_patches': len(coord_list),
        'coords': coord_list
    }
    with open(coord_dir / f'{filename}.json', 'w', encoding='utf-8') as fp:
        json.dump(coord_dict, fp)
    if thumbnail is not None:
        cv2.imwrite(str(overview_dir / f'{filename}.png'), thumbnail)
    print(f"{filename} | mag0: {level0_magnification} | (rows, cols): {num_step_y}, {num_step_x} | "
          f"patch_size: {patch_size} | num_patches: {len(coord_list)}")

    return feature

def run(args):
    # Directories
    
    
    
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))
    num_ftrs = model.classifier.in_features
    model_final = fully_connected(model.features, num_ftrs, 30)
    model = model.to(device)
    model_final.eval()
    model_final = model_final.to(device)
    model_final = nn.DataParallel(model_final)

    model_final.load_state_dict(torch.load('KimiaNetPyTorchWeights.pth'))


    
    all_feature = {}

    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    coord_dir = save_dir / 'coord'
    coord_dir.mkdir(parents=True, exist_ok=True)
    if args.overview:
        overview_dir = save_dir / 'overview'
        overview_dir.mkdir(parents=True, exist_ok=True)
    else:
        overview_dir = None
    if args.save_mask:
        mask_dir = save_dir / 'mask'
        mask_dir.mkdir(parents=True, exist_ok=True)
    else:
        mask_dir = None
    if args.save_patch:
        patch_dir = save_dir / 'patch'
        patch_dir.mkdir(parents=True, exist_ok=True)
    else:
        patch_dir = None
    if args.overview:
        thumbnail_dir = save_dir / 'thumbnail'
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
    else:
        thumbnail_dir = None
        
    np_dir = save_dir / 'np'
    np_dir.mkdir(parents=True, exist_ok=True)
        
    slide_filepath_list = sorted(list(Path(args.slide_dir).rglob(f'*{args.wsi_format}')))
    # if need filter files, add code here. example:
    # slide_filepath_list = [x for x in slide_filepath_list if str(x).find('DX1') != -1]
    num_slide = len(slide_filepath_list)
    
    all_slide_name = joblib.load('use_slide.pkl')
    
    print(f"Slide number: {num_slide}.\n")
    print(f"Start tiling ...")
    
    for slide_idx, slide_filepath in enumerate(slide_filepath_list):
        
        print(str(slide_filepath))
        if str(slide_filepath).split('/')[-1].split('.')[0] in all_slide_name:

            
            if args.specify_filename:
                filename = slide_filepath.stem[args.filename_l:]
            else:
                filename = slide_filepath.stem
            if (coord_dir / f'{filename}.json').exists() and not args.exist_ok:
                print(f"{str(coord_dir / f'{filename}.json')} exists, skip!")
                continue

            print('patch_dir',patch_dir)

            print(f"{slide_idx + 1:3}/{num_slide}, Processing {filename}...")
            one_fea = tiling(
                slide_filepath=slide_filepath,
                magnification=args.magnification,
                patch_size=args.patch_size,
                scale_factor=args.scale_factor,
                tissue_thresh=args.tissue_thresh,
                method=args.method,
                coord_dir=coord_dir,
                overview_dir=overview_dir,
                mask_dir=mask_dir,
                patch_dir=patch_dir,
                filename=filename,
                model_final=model_final,
                thumbnail_dir = thumbnail_dir
            )
            all_feature[filename] = one_fea
            print(f"{filename} Done!\n")
            gc.collect()
            joblib.dump(one_fea,str(np_dir)+'/'+filename+".pkl")
            
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_dir', type=str, default='/slide_dir')
    parser.add_argument('--save_dir', type=str, default='/save_dir')
    parser.add_argument('--exist_ok', action='store_true', default=False)
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--magnification', type=int, default=10, choices=[40, 20, 10, 5])
    parser.add_argument('--scale_factor', type=int, default=32,
                        help="scale wsi to down-sampled image for judging tissue percent of each patch.")
    parser.add_argument('--tissue_thresh', type=float, default=0.35) #0.35
    parser.add_argument('--overview', action='store_true', default=True)
    parser.add_argument('--save_mask', action='store_true', default=True)
    parser.add_argument('--save_patch', action='store_true', default=True)
    parser.add_argument('--wsi_format', type=str, default='.svs', choices=['.svs', '.tif'])
    parser.add_argument('--specify_filename', action='store_true', default=False)
    parser.add_argument('--filename_l', type=int, default=0)
    parser.add_argument('--filename_r', type=int, default=12)
    parser.add_argument('--method', type=str, default='rgb', choices=['otsu', 'adaptive', 'rgb'])
    args = parser.parse_args()
    print(f"args:\n{args}")
    run(args)


if __name__ == '__main__':
    main()
