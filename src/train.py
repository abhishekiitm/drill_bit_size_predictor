"""
This module is used for training the model on the given task 
"""

import os

import pandas as pd
import numpy as np
import albumentations
import torch
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit

import dataset
import engine
import utils
from model import get_model


def train_val_test_split(X, y, stratify, val_size=0.2, test_size=0.2, random_state=42):
    """
    function to split data into train, val and test set
    train : val : test -> (1-val_size-test_size) : val_size : test_size
    """
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size + test_size, random_state=random_state
    )
    for train_index, test_val_index in split.split(X, stratify):
        X_train = X[train_index]
        y_train = y[train_index]
        X_val_test = X[test_val_index]
        y_val_test = y[test_val_index]

    if test_size == 0:
        return X_train, y_train, X_val_test, y_val_test, None, None

    modified_size = test_size / (val_size + test_size)
    split2 = StratifiedShuffleSplit(
        n_splits=1, test_size=modified_size, random_state=random_state
    )
    for val_index, test_index in split2.split(X_val_test, stratify[test_val_index]):
        X_val = X_val_test[val_index]
        y_val = y_val_test[val_index]
        X_test = X_val_test[test_index]
        y_test = y_val_test[test_index]

    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    # location of data.csv and generated images folder
    # with all the png images
    data_dir = "data_generated_images"

    # location where model will be saved to / loaded from
    model_dir = "models"

    # set True to load model from the previously save checkpoint
    WARM_START = True

    # initial learning rate
    LR_START = 6.25e-5

    # no of epochs with no improvement in val_loss to wait before reducing learning rate
    PATIENCE_LR = 3

    # no of epochs with no improvement in val_loss to wait before stopping the training
    PATIENCE_EARLY_STOPPING = 15

    # size of each training/validation batch that will be sent to GPU/CPU
    BATCH_SIZE = 32

    # let's train for 10 epochs
    EPOCHS = 100

    # ratio of validation and test sets
    # train_size will become 1 - val_size - test_size
    VAL_SIZE = 0.3
    TEST_SIZE = 0.0  # set to 0 for training the final model

    # cuda/cpu device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    # load the dataframe
    df = pd.read_csv(os.path.join(data_dir, "data.csv"))

    # fetch all image locations
    image_paths = df.image_path.values

    # targets numpy array
    widths = df.width.values.astype(float)
    heights = df.height.values.astype(float)
    targets = np.c_[widths, heights]

    # get column for stratification (column looks like 20_26)
    stratify = df.width.apply(str) + df.height.apply(str) + df.is_dark.apply(str)

    # fetch out model, we will try both pretrained
    # and non-pretrained weights
    model = get_model()
    if WARM_START:
        model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))

    # move model to device
    model.to(device)

    # train, validation, test split with a fixed random state
    (
        train_images,
        train_targets,
        val_images,
        val_targets,
        test_images,
        test_targets,
    ) = train_val_test_split(
        image_paths,
        targets,
        stratify,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_state=42,
    )

    # save test_df to disk for plotting confusion matrix later
    if test_images is not None:
        test_widths = [int(x[0]) for x in test_targets]
        test_heights = [int(x[1]) for x in test_targets]
        test_df = pd.DataFrame(
            list(zip(test_images, test_widths, test_heights)),
            columns=["image_path", "test_widths", "test_heights"],
        )
        test_df.to_csv(os.path.join(data_dir, "test.csv"))

    # albumentations is an image augmentation library
    # that allows you to do many different types of image
    # augmentations.
    # these augmentations will be applied while training
    aug = albumentations.Compose(
        [
            albumentations.RandomBrightness(),
            albumentations.RandomContrast(),
            albumentations.GaussianBlur(p=0.2),
        ]
    )
    resize = None

    # fetch the ClassificationDataset class

    train_dataset = dataset.CustomDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=resize,
        augmentations=aug,
    )

    # torch dataloader creates batches of data
    # from classification dataset class
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    # same for validation data
    valid_dataset = dataset.CustomDataset(
        image_paths=val_images,
        targets=val_targets,
        resize=resize,
        augmentations=None,
    )

    # test_dataset = dataset.ClassificationDataset(
    #     image_paths=test_images,
    #     targets=test_targets,
    #     resize=resize,
    #     augmentations=None,
    # )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    # simple Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)

    # learning rate scheduler
    lr_scheduler = utils.LRScheduler(optimizer, patience=PATIENCE_LR)

    # early stopping
    early_stopping = utils.EarlyStopping(patience=PATIENCE_EARLY_STOPPING)

    min_val_mse = float("inf")
    # train and print auc score for all epochs
    for epoch in range(EPOCHS):
        engine.train(train_loader, model, optimizer, device=device)
        predictions, val_targets = engine.evaluate(valid_loader, model, device=device)
        train_predictions, train_targets = engine.evaluate(
            train_loader, model, device=device
        )
        val_mse = metrics.mean_squared_error(val_targets, predictions)
        train_mse = metrics.mean_squared_error(train_targets, train_predictions)

        if val_mse < min_val_mse:
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
            min_val_mse = val_mse

        print(f"Epoch={epoch}, Valid MSE={val_mse}")
        print(f"Epoch={epoch}, Train MSE={train_mse}")
        lr_scheduler(val_mse)
        early_stopping(val_mse)
        if early_stopping.early_stop:
            break
