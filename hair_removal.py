import pandas as pd
from skimage import color, morphology
from skimage.filters import threshold_otsu
import numpy as np
import glob
from tqdm.auto import tqdm
from skimage.exposure import rescale_intensity
from torch import Tensor
import cv2
import os


def rescale(image: np.ndarray, mn: int = 0, mx: int = 1):
    return rescale_intensity(image, out_range=(mn, mx))


def rescale_torch(image: Tensor) -> Tensor:
    return (255 * (image - image.min()) / (image.max() - image.min())).type(uint8)


def hair_removal(image, radius=3):
    luv = color.rgb2luv(image)
    L = luv[:, :, 0]
    # u = luv[:, :, 1]
    # v = luv[:, :, 2]
    morphed = np.zeros_like(luv)
    for i in range(3):
        morphed[:, :, i] = morphology.closing(
            luv[:, :, i], selem=morphology.disk(radius))

    diff = np.abs(L - morphed[:, :, 0])
    threshold = diff > threshold_otsu(diff)

    notted = np.zeros_like(luv)
    for i in range(3):
        notted[:, :, i] = luv[:, :, i]*(~threshold)

    multiplied = np.zeros_like(luv)
    for i in range(3):
        multiplied[:, :, i] = morphed[:, :, i]*threshold

    result = rescale(color.luv2rgb(multiplied+notted), 0, 255)
    return result.astype(np.uint8)


if __name__ == '__main__':
    metadata_train = pd.read_csv("metadata_train.csv")
    metadata_test = pd.read_csv("metadata_test.csv")
    train_image_names = glob.glob("../input/isic/train_256/*.jpg")
    test_image_names = glob.glob("../input/isic/test_256/*.jpg")
    os.system("mkdir -p ../input/isic/train_256_nohair")
    os.system("mkdir -p ../input/isic/test_256_nohair")
    for img in tqdm(train_image_names):
        image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        image = hair_removal(image)
        image = cv2.imwrite(
            f"../input/isic/train_256_nohair/{img.split('/')[-1]}",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    images = np.array([(cv2.cvtColor(cv2.imread(
        f'../input/isic/train_256_nohair/{x}.jpg'), cv2.COLOR_BGR2RGB).astype(np.uint8), t) for (x, t) in tqdm(metadata_train[['image_name', 'target']].values)])
    with open("images_targets_256_nohair.npy", "wb") as f:
        np.save(f, images)

    for img in tqdm(test_image_names):
        image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        image = hair_removal(image)
        image = cv2.imwrite(
            f"../input/isic/test_256_nohair/{img.split('/')[-1]}",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(metadata_test[['image_name']].head())
    print(metadata_test['image_name'].values[0])
    images = np.array([(cv2.cvtColor(cv2.imread(
        f'../input/isic/test_256_nohair/{x}.jpg'), cv2.COLOR_BGR2RGB).astype(np.uint8)) for x in tqdm(metadata_test['image_name'].values)])
    with open("images_test_256_nohair.npy", "wb") as f:
        np.save(f, images)
