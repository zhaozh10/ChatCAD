import os 
import cv2 
import numpy as np 
import torch 
import albumentations as A 
from dental.models.detection import get_hourglass
from dental.models.segmentation import ToothInstanceSegmentation
from dental.models.classification import DenseNet, densenet161
from dental.utils import ctdet_decode, gaussian_map, exp_tem, crop
import torch.nn.functional as F
from PIL import Image

class PAN():
    
    def __init__(self, config):
        self._c = config 

    def load_model(self):
        
        detection_model = get_hourglass['large_hourglass']
        detection_model.load_state_dict(torch.load(self._c['detection'])['model'])
        detection_model.cuda()
        detection_model.eval()
        
        segmentation_model = ToothInstanceSegmentation(1, 2)
        segmentation_model.load_state_dict(torch.load(self._c['segmentation'])['model'])
        segmentation_model.cuda()
        segmentation_model.eval()
        
        classification_model = densenet161(pretrained=False)
        classification_model.load_state_dict(torch.load(self._c['classification'])['model'])
        classification_model.cuda()
        classification_model.eval()
        
        return detection_model, segmentation_model, classification_model
    
    def apply_detection(self, img, model):
        
        augmentation = A.Compose([
            A.Resize(width=1024, height=512),
            A.Equalize(p=1)
        ])
        
        h, w = img.shape
        aug_img = augmentation(image=img)['image']
        aug_img = aug_img.astype(np.float32) / 255.
        img_tensor = torch.from_numpy(aug_img).unsqueeze(0).unsqueeze(0).float().cuda()

        with torch.no_grad():
            out = model(img_tensor)[-1]
        dets = ctdet_decode(*out, K=32)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]
        top_preds = {}
        dets[:, :4] *= 4 
        clses = dets[:, -1]
        for j in range(32):
            inds = (clses == j)
            top_preds[j] = dets[inds, :].astype(np.float32)

        detection_results = {}

        for k, v in top_preds.items():
            tid = ((int(k)) // 8 + 1) * 10 + (int(k)) % 8 + 1
            if v.shape[0]:
                max_score_v = v[np.argmax(v[:, -2])]
                if max_score_v[-2] > self._c['detection_threshold']:
                    detection_results[tid] = {}
                    detection_results[tid]['xmin'] = max_score_v[0] / 1024. * w
                    detection_results[tid]['ymin'] = max_score_v[1] / 512.  * h
                    detection_results[tid]['xmax'] = max_score_v[2] / 1024. * w
                    detection_results[tid]['ymax'] = max_score_v[3] / 512.  * h
        
        return detection_results

    def apply_segmentation(self, patch, model):
        augmentation = A.Compose([
                A.Resize(width=256, height=384)
        ])
        
        aug_img = augmentation(image=patch)['image'] / 255.
        img_tensor = torch.from_numpy(aug_img).unsqueeze(0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            out = F.softmax(model(img_tensor), dim=1)[:, 1:2]
        
        return out
    
    def apply_classification(self, patch, mask, model):
        augmentation = A.Compose([
            A.Equalize(p=1)
        ])
        
        aug_img = augmentation(image=patch)['image'] / 255.
        
        img_tensor = torch.from_numpy(np.stack([aug_img, mask])).unsqueeze(0).float()
        with torch.no_grad():
            score = float(exp_tem(model(img_tensor.cuda()).squeeze().cpu().numpy())[1])
    
        return score

    def excution(self, img_path):
        img=Image.open(img_path).convert('L')
        img = np.asarray(img, dtype=np.uint8)
        # img = cv2.imread(img_path, 0)
        
        # detection of pan
        detection_model, segmentation_model, classification_model = self.load_model()
        detection_results = self.apply_detection(img, detection_model)
        
        res = {}
        for k,v in detection_results.items():
            x, y = (v['xmin'] + v['xmax']) / 2, (v['ymin'] + v['ymax']) / 2
            patch = img[int(v['ymin']):int(v['ymax']), int(v['xmin']):int(v['xmax'])]
            mask = self.apply_segmentation(patch, segmentation_model)
            mask = F.interpolate(mask, size=(int(v['ymax'])-int(v['ymin']), int(v['xmax'])-int(v['xmin'])), mode='bilinear', align_corners=True).cpu().squeeze().numpy()
            mask = np.where(mask > 0.5, 1., 0.)
            crop_size = [int(y)-self._c['crop_area'][0]//2, int(y)+self._c['crop_area'][0] - self._c['crop_area'][0]//2, int(x)-self._c['crop_area'][1]//2, int(x)+self._c['crop_area'][1]-self._c['crop_area'][1]//2]
                
            crop_img = crop(img, crop_size)
            pad_size = [self._c['crop_area'][0] - mask.shape[0], self._c['crop_area'][1] - mask.shape[1]]
            if pad_size[0] < 0:
                mask = mask[-pad_size[0]//2:-pad_size[0]//2+pad_size[0], :]
                pad_size[0] = 0
            if pad_size[1] < 0:
                mask = mask[:, -pad_size[1]//2:-pad_size[1]//2+pad_size[1]]
                pad_size[1] = 0
            pad_mask = np.pad(mask, ((pad_size[0]//2, pad_size[0]-pad_size[0]//2), (pad_size[1]//2, pad_size[1]-pad_size[1]//2)), 'constant', constant_values=0)

            res[k] = self.apply_classification(crop_img, pad_mask, classification_model)
        # print(res)
        return res
            
class Score2txt:
    def __init__(self, disease_prediction:dict):

        self.disease=disease_prediction
        self.levelprompt={
            "0":"未发现",
            "1":"可能患有",
            "2":"疑似患有",
        }
        self.grade=self.grading()

    def grading(self):

        diseased_teeth = []
        for k in self.disease.keys():
            
            score=self.disease[k]
            if score > 0.7 and k % 10 != 8:
                diseased_teeth.append(k)
        
        if len(diseased_teeth) == 0:
            level = 0
        elif len(diseased_teeth) < 2:
            level = 1
        else:
            level = 2
            
        return {
            'level':level, 
            'teeth':diseased_teeth
        }


    
    def promptGeneration(self):
        prompt = ''
        promptHead = '牙周炎诊断：'
        prompt += promptHead
        prompt_level = "患者" + self.levelprompt[str(self.grade['level'])] + "牙周炎。"
        prompt += prompt_level
        
        if self.grade['level'] == 0:
            return prompt + '\n'
        else:
            prompt += "通过检查每颗牙齿，"
            for idx, t in enumerate(self.grade['teeth']):
                prompt += f"牙齿{t}，" if idx != len(self.grade['teeth']) - 1 else f"牙齿{t}"
            prompt += "周围" + self.levelprompt[str(self.grade['level'])] + "炎症。\n"
        
        return prompt
            
if __name__=="__main__":
    img_path = './dental/periodontals/Subject No.338.jpg'
    config = {
        'detection': './weights/detection.pt',
        'segmentation': './weights/segmentation.pt',
        'classification': './weights/classification.pt',
        'detection_threshold':0.2,
        'crop_area': (384, 256)
    }
    pan = PAN(config)
    txt_prompt = Score2txt(pan.excution(img_path)).promptGeneration()
    print(txt_prompt)
