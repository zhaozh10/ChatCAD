import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as si
import numpy as np 

def exp_tem(x, tau=5.):
    return np.exp(x/tau) / sum(np.exp(x/tau))

def crop(img, bbox):
    pad_img = img.copy()
    pad_size = [[0, 0], [0, 0]]
    if bbox[0] < 0:
        pad_size[0][0] = -bbox[0]
    if bbox[1] > img.shape[0]:
        pad_size[0][1] = bbox[1]-img.shape[0]
    if bbox[2] < 0:
        pad_size[1][0] = -bbox[2]
    if bbox[3] > img.shape[1]:
        pad_size[1][1] = bbox[3]-img.shape[1]
    
    pad_img = np.pad(pad_img, pad_size, 'constant', constant_values=0)
    crop_img = pad_img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    return crop_img


def gaussian_map(img, center, sigma):

    img[center[0], center[1]] = 1
    img = si.gaussian_filter(img, sigma=(sigma, sigma))
    return img

def gather_feature(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def transpose_gather_feature(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feature(feat, ind)
    return feat

def nms(heat, kernel=3):
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep


def topk(scores, K=32):
  batch, cat, height, width = scores.size()

  topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

  topk_inds = topk_inds % (height * width)
  topk_ys = (topk_inds / width).int().float()
  topk_xs = (topk_inds % width).int().float()
  
  topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
  topk_clses = (topk_ind / K).int()
  
  topk_ys = gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_xs = gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
  topk_inds = gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)

  return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def ctdet_decode(hmap, regs, w_h=None, K=32):
    
    batch, cat, height, width = hmap.shape
    hmap=torch.sigmoid(hmap)

    hmap = nms(hmap)  # perform nms on heatmaps

    scores, inds, clses, ys, xs = topk(hmap, K=K) # get top K scores, indices, classes, x and y coordinates

    regs = transpose_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    

    if w_h is not None:
        w_h = transpose_gather_feature(w_h, inds)
        w_h = w_h.view(batch, K, 2)
        bboxes = torch.cat([xs - w_h[..., 0:1] / 2, ys - w_h[..., 1:2] / 2, xs + w_h[..., 0:1] / 2, ys + w_h[..., 1:2] / 2], dim=2)
        
        return torch.cat([bboxes, scores, clses], dim=2)
    
    else:

        return torch.cat([xs, ys, scores, clses], dim=2) # B, K, 2

