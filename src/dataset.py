"""
This module implements a custom dataset which is suited for our usecase.
This has logic related to how we will fetch the image data and the targets for training / evaluation
"""
import torch
import numpy as np
from PIL import Image


class CustomDataset:
    def __init__(self, image_paths, targets, resize=None, augmentations=None):
        """
        :param image_paths: list of path to images
        :param targets: numpy array
        :param resize: tuple, e.g. (256, 256), resizes image if not None
        :param augmentations: albumentation augmentations
        """
        self.image_paths = image_paths
        self.targets = targets
        self.resize = resize
        self.augmentations = augmentations

    def __len__(self):
        """
        Return the total number of samples in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, item):
        """
        For a given "item" index, return everything we need
        to train a given model
        """
        # use PIL to open the image
        image = Image.open(self.image_paths[item])
        # grab correct targets
        targets = self.targets[item]
        # resize if needed
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        # convert image to numpy array
        image = np.array(image)
        # if we have albumentation augmentations
        # add them to the image
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        # add channel if image has no channels (grayscale)
        if len(image.shape) == 2:
            image = np.resize(image, (image.shape[0], image.shape[1], 1))

        # pytorch expects CHW instead of HWC
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # return tensors of image and targets
        # take a look at the types!
        # for regression tasks,
        # dtype of targets will change to torch.float
        return {
            "image": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }
