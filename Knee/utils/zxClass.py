import os
import time

import numpy as np
import torch
import yaml
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, roc_auc_score)

from Knee.utils.utils import readTrainIndex


def get_one_hot(label, num_cls):
    label = label.reshape(-1)
    label = np.eye(num_cls)[label]
    return label


class Config(object):
    def __init__(self, args) -> None:
        super().__init__()
        cfg = yaml.load(open(args.c), Loader=yaml.FullLoader)

        # Training Settings
        self.gpu = args.g
        # self.seed = cfg["seed"]
        self.num_workers = cfg["num_workers"]
        self.bs = cfg["bs"]
        self.fold = args.f
        self.test = args.t
        self.device = None

        # Data Settings
        self.path = cfg["path"]
        self.csv = cfg["csv"]
        self.result = cfg["result"]
        self.num_cls = cfg["num_cls"]
        self.trainfile = readTrainIndex(os.path.join(self.csv, f"train_set_{self.fold}.csv"))
        self.testfile = readTrainIndex(os.path.join(self.csv, f"test_set_{self.fold}.csv".format()))
        self.valfile = readTrainIndex(os.path.join(self.csv, f"test_set_{(self.fold + 1) % 5}.csv"))

        # Optimizer settings
        self.lr = cfg["lr"]
        self.momentum = cfg["momentum"]
        self.weight_decay = cfg["weight_decay"]
        self.lr_freq = cfg["lr_freq"]
        self.lr_decay = cfg["lr_decay"]

        # Model Settings
        self.task = cfg["task"]
        self.net = cfg["net"]
        self.input_size = cfg["input_size"]
        if os.path.exists(cfg["pretrain"][0]):
            self.pretrain = cfg["pretrain"]
        else:
            self.pretrain = None

        self.num_epoch = cfg["num_epoch"]

        self.TIME = time.strftime("%Y-%m-%d-%H-%M")  # time of we run the script

        if self.gpu == "cpu" or self.test:
            self.path_log = os.path.join("results", "temp")
            self.path_ckpt = self.path_log
            self.log_dir = os.path.join(self.path_log, "test.log")
            self.best_ckpt = os.path.join(self.path_ckpt, "best.pth")
            self.last_ckpt = os.path.join(self.path_ckpt, "last.pth")
        else:
            self.path_log = os.path.join(self.result, "logs", self.task, self.net)
            self.path_ckpt = os.path.join(self.result, "checkpoints", self.task, self.net)
            self.log_dir = os.path.join(self.path_log, "{}-{}.log".format(self.fold, self.TIME))
            self.best_ckpt = os.path.join(self.path_ckpt, "{}-{}-{}".format(self.fold, self.TIME, "best.pth"))
            self.last_ckpt = os.path.join(self.path_ckpt, "{}-{}-{}".format(self.fold, self.TIME, "last.pth"))


class Config_eval(object):
    def __init__(self, args) -> None:
        super().__init__()
        cfg = yaml.load(open(args.c), Loader=yaml.FullLoader)

        # Training Settings
        self.gpu = args.g
        # self.seed = cfg["seed"]
        self.num_workers = cfg["num_workers"]
        self.bs = cfg["bs"]
        self.fold = args.f
        self.test = args.t
        self.device = None

        # Data Settings
        self.path = cfg["path"]
        csv = cfg["csv"]
        self.result = cfg["result"]
        self.num_cls = cfg["num_cls"]
        if os.path.exists(cfg["pretrain"][0]):
            self.pretrain = cfg["pretrain"]
        else:
            self.pretrain = None
        self.trainfile = readTrainIndex(f"{csv}/train_set_{self.fold}.csv")
        self.testfile = readTrainIndex(f"{csv}/train_set_{self.fold}.csv")
        self.valfile = readTrainIndex(f"{csv}/train_set_{self.fold}.csv")

        # Optimizer settings
        self.lr = cfg["lr"]
        self.momentum = cfg["momentum"]
        self.weight_decay = cfg["weight_decay"]
        self.lr_freq = cfg["lr_freq"]
        self.lr_decay = cfg["lr_decay"]

        # Model Settings
        self.task = cfg["task"]
        self.net = cfg["net"]
        self.input_size = cfg["input_size"]

        self.TIME = time.strftime("%Y-%m-%d-%H-%M")  # time of we run the script

        if self.gpu == "cpu" or self.test:
            self.path_log = os.path.join("results", "temp")
            self.log_dir = os.path.join(self.path_log, "eval_test.log")
        else:
            self.path_log = os.path.join(self.result, "logs", self.task, self.net)
            self.log_dir = os.path.join(self.path_log, "eval_{}-{}.log".format(self.fold, self.TIME))

class Result(object):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.best_epoch = 0
        self.best_recall = 0.0
        return

    def add_output(self, pred, true):
        self.pred.append(pred)
        self.true.append(true)
        return

    def init_output(self):
        self.st = time.time()
        self.pred = []
        self.true = []
        return

    def cal_output(self):

        self.pred = torch.cat(self.pred, dim=0)
        self.true = torch.cat(self.true, dim=0)

        probe = torch.softmax(self.pred, dim=1)
        true = self.true.cpu().detach().numpy()
        pred = probe.cpu().detach().numpy()
        preds = np.argmax(pred, axis=1)
        true_one_hot = get_one_hot(true, self.cfg.num_cls)

        self.acc = accuracy_score(true, preds)
        self.rec = sensitivity_score(true, preds, average="macro")
        self.auc = roc_auc_score(true_one_hot, pred, average="macro")
        self.pre = precision_score(true, preds, average="macro", zero_division=0)
        self.spe = specificity_score(true, preds, average="macro")
        self.f1 = f1_score(true, preds, average="macro")
        self.cm = confusion_matrix(true, preds)

        self.ft = time.time()
        return


class Data(object):
    def __init__(self) -> None:
        super().__init__()
        self.patch = None
        self.label = None
        self.bones = None
        return

    def to(self, device):
        self.patch = self.patch.to(device)
        self.label = self.label.to(device)
        self.bones = self.bones.to(device)
        return

class CSNData(object):
    def __init__(self) -> None:
        super().__init__()
        self.patch = None
        self.vlabel = None
        self.glabel = None
        self.pos = None
        self.graph = None

        return

    def to(self, device):
        self.patch = self.patch.to(device)
        self.glabel = self.glabel.to(device)
        self.vlabel = self.vlabel.to(device)
        self.pos = self.pos.to(device)
        self.graph = self.graph.to(device)
        return
