from catalyst.core import State
import torch
# from dataset import ALASKAData
import numpy as np
import albumentations as albu
from typing import Tuple, Dict, List
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from torch.nn import functional as F

from sklearn import metrics
import numpy as np

from catalyst.dl.callbacks import (
    MetricManagerCallback, MetricCallback, Callback, CallbackOrder,
    CallbackNode)
from typing import Union, List, Dict


def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization


class AlaskaAUCCallback(Callback):
    def __init__(
        self,
        prefix: str = "wauc",
        input_key: Union[str, List[str], Dict[str, str]] = "targets",
        output_key: Union[str, List[str], Dict[str, str]] = "logits",
    ):
        super().__init__(
            order=CallbackOrder.Metric, node=CallbackNode.All
        )
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.y_true = []
        self.y_pred = []

    def on_epoch_start(self, state: "State"):
        self.y_true = []
        self.y_pred = []

    def on_batch_end(self, state: State):
        logits = state.output[self.output_key].detach().float()
        targets = state.input[self.input_key].detach().cpu().numpy()

        probabilities = F.softmax(logits, dim=1)
        probabilities = probabilities.cpu().numpy()
        preds = probabilities.argmax(axis=1)
        new_preds = np.zeros((len(preds), 1))
        new_preds[preds != 0, 0] = probabilities[preds != 0, 1:].sum(1)
        new_preds[preds == 0, 0] = 1 - probabilities[preds == 0, 0]

        self.y_pred.append(new_preds)
        self.y_true.append(targets)

    def on_epoch_end(self, state: "State"):
        self.y_true = np.concatenate(self.y_true, axis=0)
        self.y_pred = np.concatenate(self.y_pred, axis=0)

        metric = alaska_weighted_auc(self.y_true, self.y_pred)
        state.epoch_metrics[f"{state.loader_name}_{self.prefix}"] = metric


def to_tensor(x):
    return np.transpose(x, (2, 0, 1))


# def get_train_augm(size=Tuple[int, int],
#                    p=0.5):
#     return albu.Compose([
#         albu.Flip(p=p),
#         albu.ToFloat(255),
#         ToTensorV2()  # albu.Lambda(image=to_tensor)
#     ])


# def get_valid_augm(size=Tuple[int, int],
#                    p=0.5):
#     return albu.Compose([
#         albu.ToFloat(255),
#         ToTensorV2()  # albu.Lambda(image=to_tensor)
#     ])


def get_loader(dataframe, augm_func, size,
               bs, shuffle, stage, num_workers=2):
    return DataLoader(
        ALASKAData(
            dataframe, augm_func(
                size
            ), stage=stage
        ),
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers
    )


def replace(network, original, replacement, name):
    def replace_(m, name, original=original, replacement=replacement):
        for attr_str in dir(m):
            target_attr = getattr(m, attr_str)
            if type(target_attr) == original:
                setattr(m, attr_str, replacement())
    for m in network.modules():
        replace_(m, name)


def display_random_images(size: int,
                          labels: Dict[str, int]):
    paths = list(labels.keys())
    idx = np.random.randint(0, len(labels)-1, size*size)
    fig, ax = plt.subplots(size, size, figsize=(18, 13))
    k = 0
    for i in range(size):
        for j in range(size):
            img = cv2.imread(paths[idx[k]])[..., ::-1]
            ax[i, j].imshow(img)
            ax[i, j].axis(False)
            ax[i, j].set_title(
                f"Label: {labels[paths[idx[k]]]}, ID: {paths[idx[k]].split('/')[-1]}")
            k += 1
    plt.tight_layout()
    return
