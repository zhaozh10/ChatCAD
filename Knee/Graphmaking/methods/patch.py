
import dgl
import Knee.Graphmaking.config as cfg
import torch
from torchvision.transforms import CenterCrop, Pad


def paintingPatch(slice, vertex):
    p = cfg.PATCH_SIZE
    p1, p2 = p - p // 2, p + p // 2
    shape = slice.shape
    slice = Pad([p], fill=0.0)(slice)
    mask = torch.zeros(slice.shape, dtype=bool)
    for xy in vertex:
        x, y = xy
        x, y = int(x), int(y)
        mask[x + p1 : x + p2, y + p1 : y + p2] = 1

    mask = CenterCrop(shape)(mask)
    return mask


def extract_patch(slice, vertex):
    p = cfg.PATCH_SIZE
    p1, p2 = p - p // 2, p + p // 2
    slice = Pad([p], fill=0)(slice)
    patch = []
    for xy in vertex:
        x, y = xy
        x, y = int(x), int(y)
        patch.append(slice[x + p1 : x + p2, y + p1 : y + p2])
    return torch.stack(patch)


def extractPatch(data):
    ptchs, bones = [], []
    mask = torch.zeros(data.mri.shape, dtype=bool)

    # for b in [0]:
    for b in range(data.num):
        for s in range(data.slice):
            if len(data.v_2d[b][s]) > 0:
                _ptchs = extract_patch(data.mri[s], data.v_2d[b][s])
                _bones = extract_patch(data.bone[b][s], data.v_2d[b][s])
                ptchs.append(_ptchs)
                bones.append(_bones)
                mask[s] += paintingPatch(data.mri[s], data.v_2d[b][s])
    data.patch = torch.cat(ptchs, dim=0).type(torch.float32)
    data.bones = torch.cat(bones, dim=0).type(torch.long)
    data.pos = torch.tensor(data.v_3d, dtype=torch.float32)
    data.pos[:, 0] = data.pos[:, 0] / ((data.slice - 1) * data.thick)
    data.pos[:, 1] = data.pos[:, 1] / data.shape[0]
    data.pos[:, 2] = data.pos[:, 2] / data.shape[1]
    graph = dgl.graph((data.edges[0], data.edges[1]))
    data.graph = dgl.add_self_loop(graph)
    data.mask = mask
