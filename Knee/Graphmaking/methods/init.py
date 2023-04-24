import os
from os.path import join

import Knee.Graphmaking.config as cfg
import numpy as np
import SimpleITK as sitk
import torch
from SimpleITK import GetArrayFromImage as GAFI
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode as im

select_bone = [1, 2, 3]
select_cart = [4, 5, 6]


def split_seg(seg):
    n = 6 + 1
    _seg = torch.zeros([n, *seg.shape], dtype=torch.int)
    for i in range(n):
        temp = seg.clone()
        temp[temp != i] = 0
        _seg[i] = temp
    _seg[_seg > 0] = 1
    return _seg


def extract_mask(les_file, shape):
    if os.path.exists(les_file):
        les_data = sitk.ReadImage(les_file)
        les_img = torch.tensor(sitk.GetArrayFromImage(les_data).astype(int), dtype=int)
    else:
        les_img = torch.zeros([*shape], dtype=int)
    return les_img


def initData(data):
    mri_path = join(data.path)
    seg_path = join(data.path.replace(".nii.gz", "_seg.nii.gz"))
    mri_data = sitk.ReadImage(mri_path)
    seg_data = sitk.ReadImage(seg_path)
    mri_img = torch.tensor(GAFI(mri_data), dtype=torch.float32)
    seg_img = torch.tensor(GAFI(seg_data).astype(np.int32), dtype=torch.int32)
    data.rawshape = mri_img.shape

    # Normalize spacing
    spacing = list(mri_data.GetSpacing())
    scale = spacing[0] / cfg.STD_SPACING
    size = (seg_img.shape[0], int(seg_img.shape[1] * scale), int(seg_img.shape[2] * scale))
    mri_img = Resize(size=size[1:], interpolation=im.BILINEAR)(mri_img)
    seg_img = Resize(size=size[1:], interpolation=im.NEAREST)(seg_img)

    # Devide segment labels into different slices
    seg = split_seg(seg_img)
    bone = seg[select_bone]
    cart = seg[select_cart]

    # Add infomation
    data.shape = mri_img.shape[1:]
    data.slice = mri_img.shape[0]
    data.space = [spacing[2], cfg.STD_SPACING, cfg.STD_SPACING]
    data.num = 3

    data.mri = mri_img
    data.bone = bone
    data.cart = cart

    return data
