import argparse
import glob
import math
import os
import copy
import os.path
import torch
import torch.nn as nn
import pickle
# import config as cfg
from Knee.Segmentation.net import UNet
import numpy as np
import SimpleITK as sitk
import torch
from Knee.nets.csnet import CSNet
import torchvision
from SimpleITK import GetArrayFromImage as GAFI
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
from Knee.Graphmaking.methods.edge import extractEdges
from Knee.Graphmaking.methods.FOV import adjustFOV
from Knee.Graphmaking.methods.init import initData
from Knee.Graphmaking.methods.patch import extractPatch
from Knee.Graphmaking.methods.save import saveData
# from Graphmaking.methods.test import test
from Knee.Graphmaking.methods.vertex import extractVertex

def knee_prompt(idx:int, prob: list):
    # grading=["no defects","mild defects","severe defects"]
    grading=["无缺陷","轻微缺陷","严重缺陷"]
    report_prompt="诊断结果为："
    report_prompt+=grading[idx]
    report_prompt+=f"置信度为{prob[idx]:.2f}"
    
    return report_prompt


class MRIData(object):
    def __init__(self, file_path,save_path) -> None:
        super().__init__()
        self.path = file_path
        file_name=os.path.basename(file_path).replace(".nii.gz", "_graph")
        # self.save = file_path.replace(".nii.gz", "_graph")
        self.save = os.path.join(save_path,file_name)


def run(data,save: bool=False):
    data = copy.deepcopy(data)
    initData(data)
    adjustFOV(data)
    extractVertex(data)
    extractEdges(data)
    extractPatch(data)
    if save:
        saveData(data)
        del data
        return
    else:
        return data
    
    


def normalize(image):
    image = (image - image.min()) / (image.max() - image.min())
    img_max = 0.995
    img_min = 0.005
    image[image > img_max] = img_max
    image[image < img_min] = img_min
    image = (image - img_min) / (img_max - img_min)
    return image


def save_label(mri_data, preds, save_path):
    seg_data = sitk.GetImageFromArray(preds.cpu().numpy().astype(np.uint8))
    seg_data.SetOrigin(mri_data.GetOrigin())
    seg_data.SetDirection(mri_data.GetDirection())
    seg_data.SetSpacing(mri_data.GetSpacing())
    sitk.WriteImage(seg_data, save_path)


class Data(object):
    def __init__(self, in_path: str) -> None:
        super().__init__()
        vertex = np.load(in_path.replace(".nii.gz", "_graph.npz"))
        with open(in_path.replace(".nii.gz", "_graph.pkl"), "rb") as f:
            self.graph = pickle.load(f)
            self.vertex = pickle.load(f)
            self.vertex = self.vertex.astype(np.int32)
        seg = sitk.ReadImage(in_path.replace(".nii.gz", "_seg.nii.gz"))
        self.patch = torch.tensor(vertex["patch"][:, None, ...], dtype=torch.float32)
        self.pos = torch.tensor(vertex["pos"], dtype=torch.float32)
        self.seg = seg
        self.name = in_path.replace(".nii.gz", "")
        return

    def to(self, dev):
        self.patch = self.patch.to(dev)
        self.pos = self.pos.to(dev)
        self.graph = self.graph.to(dev)
        return


def graphCAM(x, y, classify, using_sigmoid=True):
    w = classify.weight.detach()[y]
    w = torch.transpose(w, 0, 1)
    cam = torch.mm(x, w)
    cam_min = cam.min(dim=0, keepdim=True)[0]
    cam_max = cam.max(dim=0, keepdim=True)[0]
    norm = cam_max - cam_min
    norm[norm == 0] = 1e-5
    cam = (cam - cam_min) / norm
    if using_sigmoid:
        cam = torch.sigmoid(100 * (cam - 0.5))
    cam_mean = cam.mean(dim=0)[0]
    cam = nn.ReLU()(cam / cam_mean - 1.5)
    cam_min = cam.min(dim=0, keepdim=True)[0]
    cam_max = cam.max(dim=0, keepdim=True)[0]
    norm = cam_max - cam_min
    norm[norm == 0] = 1e-5
    cam = (cam - cam_min) / norm
    cam = cam.squeeze()
    cam = cam.detach().cpu().numpy()
    return cam.reshape(-1, 1)


def make_lesion(data, gcls, vcls,save_path):
    seg_img = sitk.GetArrayFromImage(data.seg).astype(np.int16)
    cart_idx = [4, 5, 6]
    lesion = 8 + gcls
    for idx, v in enumerate(data.vertex):
        if vcls[idx]:
            min_w = max(0, v[1] - 16)
            max_w = min(seg_img.shape[1], v[1] + 16)
            min_h = max(0, v[2] - 16)
            max_h = min(seg_img.shape[2], v[2] + 16)
            patch = seg_img[v[0], min_w:max_w, min_h:max_h]
            for c in cart_idx:
                patch[patch == c] = lesion
            seg_img[v[0], min_w:max_w, min_h:max_h] = patch
    seg_img = seg_img - 8
    seg_img[seg_img < 0] = 0
    # os.path.join("./knee_data/",file_name.replace(".nii.gz", "_seg.nii.gz"))
    target_path=os.path.join(save_path,f"{os.path.basename(data.name)}_lesion.nii.gz")
    # save_label(data.seg, torch.tensor(seg_img), f"{data.name}_lesion.nii.gz")
    save_label(data.seg, torch.tensor(seg_img), target_path)


def knee_forward(knee_mri: str):
    device = torch.device("cpu")
    knee_unet = UNet()
    knee_unet.load_state_dict(torch.load('./weights/unet_weight.pth'))
    knee_unet.eval()
    torch.set_grad_enabled(False)

    mri_path = os.path.join(knee_mri)
    image_data = sitk.ReadImage(mri_path)
    image = torch.tensor(GAFI(image_data).astype(np.float32), dtype=torch.float32).to(device)
    image = normalize(image)
    shape = image.shape[1:]
    _image = Resize((256, 256), interpolation=InterpolationMode.BILINEAR)(image)
    _preds = torch.argmax(knee_unet(_image[:, None, ...]), axis=1)
    preds = Resize(shape, interpolation=InterpolationMode.NEAREST)(_preds)
    save_label(image_data, preds, knee_mri.replace(".nii.gz", "_seg.nii.gz"))

    data = MRIData(knee_mri)
    run(data,save=True)

    data = Data(knee_mri)
    data.to(device)
    csnet=CSNet().to(device)
    csnet.load_state_dict(torch.load('./weights/csnet_weight.pth'))
    g_cls, v_cls, x = csnet(data)
    vcls = graphCAM(x, torch.argmax(g_cls, dim=1), csnet.gclassifier)
    vcls[vcls < 0.5] = 0
    vcls[vcls > 0] = 1
    vcls = vcls.astype(np.int32)
    gcls = torch.argmax(g_cls, dim=1).item()
    # gprob = np.round(torch.nn.Softmax(dim=1)(g_cls).cpu().detach().numpy(), decimals=3)
    gprob = torch.nn.Softmax(dim=1)(g_cls).cpu().detach().numpy()
    make_lesion(data, gcls, vcls)
    
    print(gcls)
    print(gprob)
    return gcls,gprob


class KneeCAD:
    def __init__(self,save_path: str) -> None:
        self.save_path=save_path
        self.device = torch.device("cpu")
        knee_unet = UNet().to(self.device)
        knee_unet.load_state_dict(torch.load('./weights/unet_weight.pth'))
        self.knee_unet=knee_unet
        csnet=CSNet().to(self.device)
        csnet.load_state_dict(torch.load('./weights/csnet_weight.pth'))
        self.csnet=csnet
    
    def forward(self, knee_mri: str):
        file_name=os.path.basename(knee_mri)
        self.knee_unet.eval()
        torch.set_grad_enabled(False)

        mri_path = os.path.join(knee_mri)
        image_data = sitk.ReadImage(mri_path)
        image = torch.tensor(GAFI(image_data).astype(np.float32), dtype=torch.float32).to(self.device)
        image = normalize(image)
        shape = image.shape[1:]
        _image = Resize((256, 256), interpolation=InterpolationMode.BILINEAR)(image)
        _preds = torch.argmax(self.knee_unet(_image[:, None, ...]), axis=1)
        preds = Resize(shape, interpolation=InterpolationMode.NEAREST)(_preds)
        
        # save_label(image_data, preds, knee_mri.replace(".nii.gz", "_seg.nii.gz"))
        save_label(image_data, preds, os.path.join(self.save_path,file_name.replace(".nii.gz", "_seg.nii.gz")))

        data = MRIData(knee_mri,save_path=self.save_path)
        run(data,save=True)

        data = Data(knee_mri)
        data.to(self.device)
        g_cls, v_cls, x = self.csnet(data)
        vcls = graphCAM(x, torch.argmax(g_cls, dim=1), self.csnet.gclassifier)
        vcls[vcls < 0.5] = 0
        vcls[vcls > 0] = 1
        vcls = vcls.astype(np.int32)
        gcls = torch.argmax(g_cls, dim=1).item()
        # gprob = np.round(torch.nn.Softmax(dim=1)(g_cls).cpu().detach().numpy(), decimals=3)
        gprob = torch.nn.Softmax(dim=1)(g_cls).cpu().detach().numpy()
        make_lesion(data, gcls, vcls,save_path=self.save_path)
        
        print(gcls)
        print(gprob)
        return gcls,gprob



if __name__ == "__main__":
    kneecad=KneeCAD("./knee_data/")
    label,prob=kneecad.forward("../CSNet_knee/demo/00002_2.nii.gz")
    # label,prob=knee_forward(knee_mri="../CSNet_knee/demo/00002_2.nii.gz")
    report_prompt=knee_prompt(label,prob[0].tolist())
    print(report_prompt)

