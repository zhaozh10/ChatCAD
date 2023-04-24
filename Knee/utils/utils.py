import csv
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision



def initLogging(logFilename):
    """Init for logging"""
    logger = logging.getLogger("")

    if not logger.handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            format="[%(asctime)s-%(levelname)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            filename=logFilename,
            filemode="w",
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s-%(levelname)s] %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)


def get_lr(epoch, cfg):
    lr = cfg.lr * (cfg.lr_decay ** (epoch // cfg.lr_freq + 1))
    return lr


def init_train(cfg):

    cfg.device = torch.device("cuda" if cfg.gpu != "cpu" else "cpu")
    if not os.path.exists(cfg.path_log):
        os.makedirs(cfg.path_log)
    initLogging(cfg.log_dir)
    logging.debug("Input: " + cfg.path)
    logging.info("Log: " + cfg.log_dir)

    forma = ("\n" + "|{:^9}" * 4 + "|") * 2
    title = ["NET", "BS", "FOLD", "LR"]
    items = [cfg.net, cfg.bs, cfg.fold, cfg.lr]
    logging.info(forma.format(*title, *items))


def save_model(epoch, result, net, cfg):
    result = result["test"]
    if not os.path.exists(cfg.path_ckpt):
        os.makedirs(cfg.path_ckpt)

    # Save latest model
    # torch.save(net, cfg.last_ckpt)

    # Save best model
    if result.rec > result.best_recall:
        result.best_recall = result.rec
        result.best_epoch = epoch + 1
        torch.save(net, cfg.best_ckpt)
    logging.info("BEST RECALL: {:.3f}, EPOCH: {:3}\n\n".format(result.best_recall, result.best_epoch))
    return


def save_model_new(epoch, result, net, cfg):
    if not os.path.exists(cfg.path_ckpt):
        os.makedirs(cfg.path_ckpt)

    # Save latest model
    # torch.save(net, cfg.last_ckpt)

    # Save best model
    if result.rec > result.best_recall:
        result.best_recall = result.rec
        result.best_epoch = epoch + 1
        torch.save(net, cfg.best_ckpt)
    logging.info("BEST RECALL: {:.3f}, EPOCH: {:3}\n\n".format(result.best_recall, result.best_epoch))
    return


def csv_writer(filename, data):
    with open(filename, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        for rows in data:
            # csvwriter.writerow(fields)
            if isinstance(rows, list):
                csvwriter.writerow(rows)
            else:
                csvwriter.writerow([rows])
    return


def readTrainIndex(csv_name):
    with open(csv_name) as f:
        reader = csv.reader(f)
        file_list = list(reader)
        file_list = [_[0] for _ in file_list]
    return file_list


# def get_stage_size(net, cfg):
#     x = torch.ones([4, 1, cfg.input_size, cfg.input_size], dtype=torch.float32).to(next(net.parameters()).device)
#     dims_list, size_list = [x.shape[1]], [x.shape[2]]
#     with torch.no_grad():
#         for layer in net:
#             x = layer(x)
#             dims_list.append(x.shape[1])
#             size_list.append(x.shape[2])
#     return dims_list, size_list
def get_stage_size(net, input_size):
    x = torch.ones([4, 1, input_size, input_size], dtype=torch.float32).to(next(net.parameters()).device)
    dims_list, size_list = [x.shape[1]], [x.shape[2]]
    with torch.no_grad():
        for layer in net:
            x = layer(x)
            dims_list.append(x.shape[1])
            size_list.append(x.shape[2])
    return dims_list, size_list

# def make_cnn(cfg):
#     if cfg.pretrain is not None:
#         cnn = torch.load(cfg.pretrain[cfg.fold])
#         return cnn.cnn, cnn.pooling
#     else:
#         from nets.cnn import CNN

#         cnn = CNN(cfg)
#         return cnn.cnn, cnn.pooling

def make_cnn():
    from Knee.nets.cnn import CNN
    cnn = CNN()
    return cnn.cnn, cnn.pooling


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


def unfreeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = True


def add_in_log(r):
    rv, rt = r["val"], r["test"]
    title_items = ["SET", "ACC", "REC", "AUC", "PRE", "SPE", "F1"]
    val = ["TRAI"] + [rv.acc, rv.rec, rv.auc, rv.pre, rv.spe, rv.f1]
    test = ["TEST"] + [rt.acc, rt.rec, rt.auc, rt.pre, rt.spe, rt.f1]
    forma_1 = "\n|{:^8}" + "|{:^5}" * (len(title_items) - 1) + "|"
    forma_2 = ("\n|{:^8}" + "|{:^.3f}" * (len(title_items) - 1) + "|") * 2
    logging.info("TRAINRECALL: {:.3f}, TIME: {:.1f}s".format(rv.rec, rv.ft - rv.st))
    logging.info("TEST RECALL: {:.3f}, TIME: {:.1f}s".format(rt.rec, rt.ft - rt.st))
    logging.debug((forma_1 + forma_2).format(*title_items, *val, *test))
    logging.debug("\nG VAL CM:\n{}\nG TEST CM:\n{}".format(rv.cm, rt.cm))


def add_in_log_new(r):
    title_items = ["SET", "ACC", "REC", "AUC", "PRE", "SPE", "F1"]
    test = ["TEST"] + [r.acc, r.rec, r.auc, r.pre, r.spe, r.f1]
    forma_1 = "\n|{:^8}" + "|{:^5}" * (len(title_items) - 1) + "|"
    forma_2 = "\n|{:^8}" + "|{:^.3f}" * (len(title_items) - 1) + "|"
    logging.info("TEST RECALL: {:.3f}, TIME: {:.1f}s".format(r.rec, r.ft - r.st))
    logging.debug((forma_1 + forma_2).format(*title_items, *test))
    logging.debug("\nG TEST CM:\n{}".format(r.cm))


def eval_log(r):
    rt = r["test"]
    title_items = ["SET", "ACC", "REC", "AUC", "PRE", "SPE", "F1"]
    test = ["TEST"] + [rt.acc, rt.rec, rt.auc, rt.pre, rt.spe, rt.f1]
    forma_1 = "\n|{:^8}" + "|{:^5}" * (len(title_items) - 1) + "|"
    forma_2 = "\n|{:^8}" + "|{:^.3f}" * (len(title_items) - 1) + "|"
    logging.info("TEST RECALL: {:.3f}, TIME: {:.1f}s".format(rt.rec, rt.ft - rt.st))
    logging.debug((forma_1 + forma_2).format(*title_items, *test))
    logging.debug("TEST CM:\n{}".format(rt.cm))


def eval_args(parser):
    parser.add_argument("-f", type=int, default=0)
    parser.add_argument("-g", type=str, default="0")
    parser.add_argument("-t", type=bool, default=False)
    parser.add_argument("-c", type=str, default="tasks/CNN/config/Patch_R18_V2.yaml")
    parser.add_argument("-net", type=str, default="")
    args = parser.parse_args()
    return args


def get_args(parser):
    parser.add_argument("-f", type=int, default=0)
    parser.add_argument("-g", type=str, default="0")
    parser.add_argument("-c", type=str)
    parser.add_argument("-t", type=bool, default=False)
    parser.add_argument("-d1", type=str, default="")
    parser.add_argument("-d2", type=str, default="")

    args = parser.parse_args()
    return args


def show_slices(img, nrow=5, path="test_slice.png"):
    assert type(img) is not torch.Tensor or np.ndarray
    if type(img) is np.ndarray:
        img = torch.tensor(img)
    if len(img.shape) == 3:
        img = img[:, None, ...]
    img = normalize(img)
    x = torchvision.utils.make_grid(img, padding=1, nrow=nrow).transpose(0, 1).transpose(1, 2).cpu().numpy()
    plt.imsave(path, x, cmap="gray")


def normalize(image):
    image = (image - image.min()) / (image.max() - image.min())
    img_max = 0.995
    img_min = 0.005
    image[image > img_max] = img_max
    image[image < img_min] = img_min
    image = (image - img_min) / (img_max - img_min)
    return image
