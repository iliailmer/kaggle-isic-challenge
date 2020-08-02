from sklearn.preprocessing import normalize
from skimage.exposure import equalize_adapthist
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import albumentations as albu
import cv2
import numpy as np
# from jpegio import jpegio as jio
from catalyst.utils import get_one_hot

import torch


class DataFromImages(Dataset):
    def __init__(self, images,
                 targets,
                 tfms: albu.Compose,
                 meta_features: np.ndarray,
                 stage='train'
                 ):
        super().__init__()
        self.images_targets = images
        # self.targets = targets
        self.tfms = tfms
        self.meta_features = normalize(meta_features)
        self.stage = stage

        if stage == 'train':
            targets = np.array([get_one_hot(label, 2) for label in targets])
            targets[np.where(targets == 0)] = -1
            self.targets_pm = np.array(targets)

    def __getitem__(self, idx):
        if self.stage == 'train':
            return {"features": (self.tfms(image=self.images_targets[idx][0])['image']),
                    "meta_features": torch.from_numpy(self.meta_features[idx]),
                    "targets": self.images_targets[idx][1],
                    "targets_pm": self.targets_pm[idx],
                    "targets_one_hot": get_one_hot(self.images_targets[idx][1], 2)}
        else:
            if len(self.images_targets[idx]) == 2:
                return {"features": (self.tfms(image=self.images_targets[idx][0])['image']),
                        "meta_features": (torch.from_numpy(self.meta_features[idx])),
                        "name": self.images_targets[idx][1]}
            else:
                return {"features": (self.tfms(image=self.images_targets[idx])['image']),
                        "meta_features": (torch.from_numpy(self.meta_features[idx])),
                        "name": self.images_targets[idx][1]}

    def __len__(self):
        return len(self.images_targets)


class SkinData(Dataset):
    def __init__(self, metadata_df,
                 path: str,
                 tfms: albu.Compose,
                 meta_features: np.ndarray,
                 stage='train'
                 ):
        super().__init__()
        self.images_targets = metadata_df
        self.path = path
        self.tfms = tfms
        self.meta_features = normalize(meta_features)
        self.stage = stage

    def __getitem__(self, idx):
        if self.stage == 'train':
            image_name, target = self.images_targets[[
                'image_name', 'target']].values[idx]
            image_name = f"{self.path}/{image_name}.jpg"
            image = cv2.cvtColor(cv2.imread(
                image_name), cv2.COLOR_BGR2RGB).astype(np.uint8)
            return {"features": (self.tfms(image=image)['image']),
                    "meta_features": torch.from_numpy(self.meta_features[idx]),
                    "targets": target,
                    "targets_one_hot": get_one_hot(target, 2)}
        else:
            image_name = self.images_targets[
                'image_name'].values[idx]
            image_name = f"{self.path}/{image_name}.jpg"
            image = cv2.cvtColor(cv2.imread(
                image_name), cv2.COLOR_BGR2RGB).astype(np.uint8)
            return {"features": (self.tfms(image=image)['image']),
                    "meta_features": (torch.from_numpy(self.meta_features[idx])),
                    "name": image_name}

    def __len__(self):
        return len(self.images_targets)


def to_tensor(x):
    return np.transpose(x, (2, 0, 1))


def get_train_augm(size=Tuple[int, int],
                   p=0.5):
    return albu.Compose([
        albu.OneOf([
            albu.CLAHE(6, (4, 4), always_apply=True),
            albu.Equalize(always_apply=True)], p=0.99),
        albu.HorizontalFlip(p=p),
        albu.VerticalFlip(p=p),
        albu.HueSaturationValue(p=p),
        albu.RandomBrightnessContrast(p=p),
        albu.Rotate(p=p),
        albu.ToFloat(255),
        ToTensorV2()  # albu.Lambda(image=to_tensor)
    ])


def get_valid_augm(size=Tuple[int, int],
                   p=0.5):
    return albu.Compose([
        albu.ToFloat(255),
        ToTensorV2()  # albu.Lambda(image=to_tensor)
    ])
