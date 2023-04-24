
import torch
import torch.nn as nn



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


if __name__ == "__main__":
    x = torch.randn([5, 512])
    y = torch.tensor([1])
    classify = nn.Linear(512, 3)
    cam = graphCAM(x, y, classify)
    print(cam)
