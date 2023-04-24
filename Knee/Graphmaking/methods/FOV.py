import numpy as np
import skimage.feature
import skimage.measure
import skimage.morphology
import torch
from csaps import csaps


def smooth_surface(XY, num=1000):
    [X, Y] = XY
    n = X.shape[0]
    t = np.linspace(0, n, n)
    ti = np.linspace(0, n, num)
    new_X = csaps(t, X, ti, smooth=0.8)
    new_Y = csaps(t, Y, ti, smooth=0.8)
    return np.array([new_X, new_Y])


def adjustFOV(data):
    # adjust FOV

    # s dircection
    s_index = torch.zeros([data.slice], dtype=torch.int32)
    for s in range(data.slice):
        if torch.sum(data.bone[:, s] + data.cart[:, s]) > 0:
            s_index[s] = 1
    s_idx = s_index.nonzero().reshape(-1)

    # h dircection
    petalla = data.bone[2]
    h_min = max(petalla.nonzero()[:, 1].min().item() - 30, 0)
    h_max = h_min + 330

    # w dircection
    w_min = max(petalla.nonzero()[:, 2].min().item() + 10, 0)
    w_max = w_min + 330

    # # extract surface
    data.surface = torch.zeros(data.bone.shape, dtype=torch.int32)
    for b in range(data.num):
        for s in range(data.slice):
            slice = data.bone[b, s].clone().numpy()
            for _ in range(3):
                slice = skimage.morphology.binary_dilation(slice)
            slice = skimage.morphology.binary_dilation(slice).astype(int) - slice.astype(int)
            data.surface[b, s] = torch.tensor(slice)

    # update bone segmentation

    data.surface[..., :h_min, :] = 0
    data.surface[..., h_max:, :] = 0
    data.surface[..., :, :w_min] = 0
    data.surface[..., :, w_max:] = 0

    # Normalize intensity
    data.mri = (data.mri - data.mri.min()) / (data.mri.max() - data.mri.min())

    return data
