"""
This module contains utils required for the training scripts

LRScheduler
EarlyStopping
get_mean_std
classify_one
classify_all
"""

import numpy as np
import cv2
import torch


class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(self, optimizer, patience=3, min_lr=1e-6, factor=0.5):
        """
        new_lr = old_lr * factor

        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
        )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


def get_mean_std(img_paths):
    """
    calculates the mean and standard deviation of the all the images in the list img_paths

    :img_paths: list of image paths for which the mean and standard deviation are to be calculated
    """
    mean = np.array([0.0, 0.0, 0.0])
    stdTemp = np.array([0.0, 0.0, 0.0])
    std = np.array([0.0, 0.0, 0.0])
    numSamples = len(img_paths)

    for i in range(numSamples):
        im = cv2.imread(str(img_paths[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.0

        for j in range(3):
            mean[j] += np.mean(im[:, :, j])

    mean = mean / numSamples

    for i in range(numSamples):
        im = cv2.imread(str(img_paths[i]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255.0
        for j in range(3):
            stdTemp[j] += ((im[:, :, j] - mean[j]) ** 2).sum() / (
                im.shape[0] * im.shape[1]
            )

    std = np.sqrt(stdTemp / numSamples)
    return mean, std


def classify_one(prediction):
    """
    classifies one regression prediction to the available classes

    :prediction: an array of length 2 whose elements denote [predicted width, predicted height]
    """
    target_widths = np.array([20, 28, 35, 42])
    idx_width = np.argmin(np.abs(target_widths - prediction[0]))

    pred_width = int(target_widths[idx_width])

    target_heights_dict = {
        20: [26, 28],
        28: [22],
        35: [19, 22, 28, 30],
        42: [22, 30],
    }

    target_heights = np.array(target_heights_dict[pred_width])
    idx_height = np.argmin(np.abs(target_heights - prediction[1]))
    pred_height = int(target_heights[idx_height])

    return [pred_width, pred_height]


def classify_all(predictions):
    """
    converts the regression predictions to the available classes

    :predictions: an array of predictions. Each prediction is an array of length 2
                    whose elements denote [predicted width, predicted height]
    """
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classify_one(prediction))
    return predicted_classes
