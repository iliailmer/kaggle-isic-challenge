import argparse
import os

import cv2
import kornia
import numpy as np
import pandas as pd
import torch
from catalyst.contrib.nn import (
    CrossEntropyLoss,
    FocalLossBinary,
    Lookahead,
    RAdam,
)
from catalyst.data.sampler import BalanceClassSampler
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import (
    AccuracyCallback,
    AUCCallback,
    CriterionCallback,
    CutmixCallback,
    EarlyStoppingCallback,
    F1ScoreCallback,
    MetricAggregationCallback,
    MixupCallback,
    OptimizerCallback,
    PrecisionRecallF1ScoreCallback,
)
from catalyst.utils import prepare_cudnn, set_global_seed
from pytorch_toolbelt.losses import FocalLoss
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchsampler import ImbalancedDatasetSampler
from tqdm import auto as tqdm
from ttach import ClassificationTTAWrapper
from ttach.aliases import d4_transform, flip_transform
from dataset import DataFromImages, SkinData, get_train_augm, get_valid_augm
from models import ENet, Model

set_global_seed(2020)
prepare_cudnn(True)
parser = argparse.ArgumentParser()


# size of batch
parser.add_argument("-bs", type=int, default=8)
# learning rate
parser.add_argument("-lr", type=float, default=1e-3)
# weight decay rate
parser.add_argument("-wd", type=float, default=0)
# number of cores to load data
parser.add_argument("-nw", type=int, default=4)
# number of epochs
parser.add_argument("-e", type=int, default=1)
# number of accumulation steps
parser.add_argument("-acc", type=int, default=1)

parser.add_argument("-test", type=int, default=0)
parser.add_argument("-train", type=int, default=1)
parser.add_argument("-pseudo", type=int, default=0)
parser.add_argument("-name", type=str, default="efficientnet-b1")
args = parser.parse_args()


train_path = "../input/isic/stratified/train"
test_path = "../input/isic/stratified/test"

metadata_train = pd.read_csv(f"../input/isic/stratified/train.csv").drop(
    "patient_id", axis=1
)
metadata_train = metadata_train[metadata_train["tfrecord"] != -1]
metadata_train["sex"] = metadata_train["sex"].map(
    {np.nan: -1, "male": 1, "female": 0}
)
metadata_train["anatom_site_general_challenge"] = metadata_train[
    "anatom_site_general_challenge"
].map(
    {
        np.nan: -1,
        "torso": 0,
        "lower extremity": 1,
        "upper extremity": 2,
        "head/neck": 3,
        "palms/soles": 4,
        "oral/genital": 5,
    }
)
metadata_train = metadata_train.fillna(metadata_train.mode().iloc[0])

metadata_test = pd.read_csv(f"../input/isic/stratified/test.csv")
metadata_test["sex"] = metadata_test["sex"].map(
    {np.nan: -1, "male": 1, "female": 0}
)
metadata_test["anatom_site_general_challenge"] = metadata_test[
    "anatom_site_general_challenge"
].map(
    {
        np.nan: -1,
        "torso": 0,
        "lower extremity": 1,
        "upper extremity": 2,
        "head/neck": 3,
        "palms/soles": 4,
        "oral/genital": 5,
    }
)
metadata_test = metadata_test.fillna(metadata_test.mode().iloc[0])


targets = metadata_train.target.values.astype(np.int32)

sample_submission = pd.read_csv(
    "../input/isic/stratified/sample_submission.csv"
)
meta_features = metadata_train[
    ["sex", "age_approx", "anatom_site_general_challenge"]
].values.astype(np.float32)

meta_features_test = metadata_test[
    ["sex", "age_approx", "anatom_site_general_challenge"]
].values.astype(np.float32)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

runner = SupervisedRunner()
logdir = f"./logs/{args.name}"

test_data = DataLoader(
    SkinData(
        metadata_df=metadata_test,
        path=test_path,
        tfms=get_valid_augm(None),
        meta_features=meta_features_test,
        stage="test",
    ),
    shuffle=False,
    batch_size=1,
    num_workers=args.nw,
)
oof_predictions = sample_submission.copy().drop(["target"], axis=1)
for i in range(kfold.n_splits):
    oof_predictions[f"fold_{i+1}"] = 0


def callback_get_label(dataset: DataFromImages, idx):
    return dataset.__getitem__(idx)["targets"]


sizes = [(224, 224), (192, 192), (128, 128), (256, 256), (160, 160)]
# from imblearn.over_sampling import
if __name__ == "__main__":
    for fold, (idxT, idxV) in enumerate(kfold.split(np.arange(15))):
        X_train = metadata_train.loc[metadata_train.tfrecord.isin(idxT)]
        X_val = metadata_train.loc[metadata_train.tfrecord.isin(idxV)]

        print(f"Fold: {fold}, {len(X_train)}")

        # ENet('efficientnet-b3')  #
        model = ENet("efficientnet-b4")

        criterion = FocalLossBinary()  # FocalLoss()
        optimizer = Lookahead(
            RAdam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )
        runner = SupervisedRunner(
            model, input_key=["features", "meta_features"]
        )
        train_dataset = SkinData(
            metadata_df=X_train,
            path=train_path,
            tfms=get_train_augm(size=sizes[fold], p=0.67),
            meta_features=X_train[
                ["sex", "age_approx", "anatom_site_general_challenge"]
            ].values.astype(np.float32),
        )
        val_dataset = SkinData(
            metadata_df=X_val,
            path=train_path,
            tfms=get_valid_augm(None),
            meta_features=X_val[
                ["sex", "age_approx", "anatom_site_general_challenge"]
            ].values.astype(np.float32),
        )
        loaders = {
            "train": DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=args.bs,
                num_workers=args.nw,
            ),
            "valid": DataLoader(
                val_dataset,
                shuffle=False,
                drop_last=True,
                batch_size=args.bs,
                num_workers=args.nw,
            ),
        }

        # print(next(iter(loaders['train'])))
        # break
        callbacks = [
            # CutmixCallback(),
            CriterionCallback(input_key="targets_one_hot"),
            OptimizerCallback(accumulation_steps=args.acc, metric_key="loss"),
            AUCCallback(input_key="targets_one_hot"),
            AccuracyCallback(num_classes=2),
            EarlyStoppingCallback(metric="loss", patience=15),
        ]
        kwargs = dict(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            callbacks=callbacks,
            logdir=logdir,
            num_epochs=args.e,
            verbose=True,
            #   fp16=dict(opt_level="O1"),
            main_metric="auc/_mean",
            minimize_metric=False,
        )
        runner.train(**kwargs)
        model.load_state_dict(
            torch.load(f"{logdir}/checkpoints/best.pth")["model_state_dict"]
        )
        model.eval()
        progress_bar_test = tqdm.tqdm(test_data)

        # pseudo_labeled = {'Id': [], 'Label': []}
        for step, images in enumerate(progress_bar_test):
            # print(step, end='\r')
            im, meta = (
                images["features"].cuda(),
                images["meta_features"].cuda(),
            )

            im_h = kornia.augmentation.F.hflip(im)

            im_v = kornia.augmentation.F.vflip(im)

            # im_ = kornia.augmentation.F.rotate(im, torch.tensor([90]))

            y_pred = (
                0.8 * model(im, meta)
                + 0.1 * model(im_v, meta)
                + 0.1 * model(im_h, meta)
            )

            oof_predictions.loc[
                oof_predictions["image_name"]
                == metadata_test.image_name.values[step],
                f"fold_{fold+1}",
            ] = (
                torch.nn.functional.softmax(y_pred, dim=1)
                .detach()
                .cpu()
                .numpy()[:, 1]
            )
        oof_predictions.to_csv("oof_predictions.csv", index=False)
        print(oof_predictions.iloc[:, 1])
    sample_submission["target"] = np.median(
        oof_predictions[[f"fold_{i+1}" for i in range(kfold.n_splits)]].values,
        axis=1,
    )
    sample_submission.to_csv(f"submission_{args.name}_median.csv", index=False)
    sample_submission["target"] = np.mean(
        oof_predictions[[f"fold_{i+1}" for i in range(kfold.n_splits)]].values,
        axis=1,
    )
    sample_submission.to_csv(f"submission_{args.name}_mean.csv", index=False)
    sample_submission["target"] = oof_predictions.iloc[
        :, 1 : kfold.n_splits
    ].values.prod(axis=1) ** (1 / kfold.n_splits)
    sample_submission.to_csv(
        f"submission_{args.name}_geom_mean.csv", index=False
    )
    os.system(
        f"kaggle competitions submit -c siim-isic-melanoma-classification -f"
        + f" submission_{args.name}_median.csv "
        + f"-m '{args.name}'"
    )
    os.system(
        f"kaggle competitions submit -c siim-isic-melanoma-classification -f"
        + f" submission_{args.name}_mean.csv "
        + f"-m '{args.name}'"
    )
    os.system(
        f"kaggle competitions submit -c siim-isic-melanoma-classification -f"
        + f" submission_{args.name}_geom_mean.csv "
        + f"-m '{args.name}'"
    )
