import argparse
from tracemalloc import start
from torch import Tensor
from torchvision import transforms
from PIL import Image
import time
import torch
from prompt import prob2text
from models.models import BaseCMNModel
from modules.tokenizers import Tokenizer
from modules.generator import Generator
from models.classifier import Classifier
from easydict import EasyDict as edict
import json
from utils import transform
import numpy as np
import torch.nn.functional as F


def getImg(imgPath: str,idx: int=None):
    # idx should be 1 or 2
    img= Image.open(imgPath).convert('RGB')
    reportTrans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
    imgTrans=transforms.Compose([
        transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
    img1,img2=reportTrans(img),imgTrans(img)
    res=[img1.unsqueeze(dim=0), img2.unsqueeze(dim=0)]
    # img0 for text generation, img1 for inference
    if idx is not None:
        return res[idx-1]
    return res

def getJFImg(imgPath: str,imgcfg,idx: int=None):
    # idx should be 1 or 2
    img1= Image.open(imgPath).convert("RGB")
    img2= Image.open(imgPath)
    reportTrans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
    img1=reportTrans(img1)
    img2 = np.array(img2)
    img2 = transform(img2,imgcfg)
    img2=torch.tensor(img2,dtype=torch.float32)
    res=[img1.unsqueeze(dim=0), img2.unsqueeze(dim=0)]
    # img0 for text generation, img1 for inference
    if idx is not None:
        return res[idx-1]
    return res

def get_pred(output, cfg):
    if cfg.criterion == 'BCE' or cfg.criterion == "FL":
        # for num_class in cfg.num_classes:
        #     assert num_class == 1
        # pred = torch.sigmoid(output.view(-1)).cpu().detach().numpy()
        pred = torch.sigmoid(output.view(-1))
    elif cfg.criterion == 'CE':
        for num_class in cfg.num_classes:
            assert num_class >= 2
        prob = F.softmax(output)
        # pred = prob[:, 1].cpu().detach().numpy()
        pred = prob[:, 1]
    else:
        raise Exception('Unknown criterion : {}'.format(cfg.criterion))
    return pred

def JFinfer(img_model, img, imgcfg):
    img_model.eval()
    outs_classes, _ = img_model(img)
    assert isinstance(outs_classes,list)
    # outs_classes= model.infer(img)
    prob = np.zeros((len(imgcfg.Data_CLASSES), 1))
    prob= get_pred(torch.Tensor(outs_classes), imgcfg)
    return prob

def JFinit(cfg_path,weight_path):
    imgcfg = edict(json.load(open(cfg_path)))
    img_model = Classifier(imgcfg)
    # model.to(torch.cuda())
    img_model.load_state_dict(torch.load(weight_path))
    return img_model, imgcfg

# def reportGen(cfg: dict=None):
def reportGen():
    cfg={
        "visual_extractor":"resnet101",
        "ann_path":"annotation.json",
        "threshold":10,
        "cmm_dim":512,
        "cmm_size":2048,
        'logit_layers':1,
        "d_model":512,
        "d_ff":512,
        "d_vf":2048,
        "num_layers":3,
        "num_heads":8,
        "drop_prob_lm":0.5,
        "dropout":0.1,
        "max_seq_length":100,
        "bos_idx":0,
        "eos_idx":0,
        "pad_idx":0,
        "use_bn":0,
        "n_gpu":1,
        "topk":32,
        "sample_method":"beam_search",
        "sample_n":1,
        "beam_size":3,
        "temperature":1.0,
        "load":'./weights/r2gcmn_mimic-cxr.pth',
        "group_size":1,
        "output_logsoftmax":1,
        "decoding_constraint":0, 
        "block_trigrams":1, 
        }
    tokenizer = Tokenizer(cfg)
    model = BaseCMNModel(cfg, tokenizer)
    generator= Generator(cfg, model)
    return generator




if __name__=="__main__":
    
    loracfg={
            "rank":4,
            "num_classes":14
            }
    disease={
        "Atelectasis":0,
        "Cardiomegaly":1,
        "Effusion":2,
        "Infiltration":3,
        "Mass":4,
        "Nodule":5,
        "Pneumonia":6,
        "Pneumothorax":7,
        "Consolidation":8,
        "Edema":9,
        "Emphysema":10,
        "Fibrosis":11,
        "Pleural_Thickening":12,
        "Hernia":13,
        }
    fivedisease={
        "Cardiomegaly":0,
        "Edema":1,
        "Consolidation":2,
        "Atelectasis":3,
        "Pleural Effusion":4,
           }



    # img_model,imgcfg=JFinit('./CXR/config/JF.json','./weights/JFchexpert.pth')

    # imgPath='../data/mimic-cxr/p10/p10005866/s55665483/6039e5db-d35aed7c-106102aa-126d200e-a262c646.jpg'
    # img1,img2=getJFImg(imgPath,imgcfg) 

    generator=reportGen()
    start=time.time()
    # report=generator.report(img1)
    end=time.time()
    print(f"Inference time {end-start:.3f}s")
    # print(report)
    start=time.time()
    
    # logits=lora_model(img2)
    # prob=torch.sigmoid(logits)
    # prob=JFinfer(img_model,img2,imgcfg)
    print(f"Image Inference {time.time()-start:.3f}s")
    converter=prob2text(prob,fivedisease)
    # converter=prob2text(prob,disease)
    res=converter.promptA()
    print(converter.promptA())
    print("\n")
    print(converter.promptB())
    print("\n")
    print(converter.promptC())
    print("\n")
    print("hold")
    # print(prob)
    