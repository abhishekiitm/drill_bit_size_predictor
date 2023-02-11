"""
This script takes in a folder as input and performs inference on all the png files within that folder

pass in the folder argument while calling this script:
$ python folder_inference.py <folder_with_images>
"""

import os
import sys

import pandas as pd
import numpy as np
import torch
import cv2

import utils
from model import get_model


if __name__ == "__main__":

    model_dir = "models"
    device = "cpu"

    X = 390
    Y = 475
    W = 140
    H = 40

    print(len(sys.argv))
    print(sys.argv)

    if len(sys.argv) == 1:
        print(
            "No folder name given as input from the command line. Finding images in current working directory.\n"
        )
        input_dir = "C:\\Users\\saini\\Documents\\drill_bit_size_predictor\\test_folder"
        # input_dir = os.getcwd()
    else:
        input_dir = sys.argv[1]

    # validation for whether the folder exists
    if not os.path.exists(input_dir):
        print("Folder {input_dir} does not exist. Exiting.")
        sys.exit(1)

    image_filenames = [x for x in os.listdir(input_dir) if x.endswith(".png")]

    # validation for whether there are png files within the folder
    if len(image_filenames) == 0:
        print("No png files found in the folder {input_dir}. Exiting.")
        sys.exit(1)

    # load the trained model and set it in eval mode
    model = get_model()
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
    model.eval()

    # perform inference over each image
    pred_labels = []
    for image_filename in image_filenames:
        im = cv2.imread(os.path.join(input_dir, image_filename))
        cropped_frame = im[Y : Y + H, X : X + W]
        # cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

        img = np.resize(img, (img.shape[0], img.shape[1], 1, 1))

        # pytorch expects BCHW instead of HWCB
        img = np.transpose(img, (3, 2, 0, 1)).astype(np.float32)
        img = torch.tensor(img, dtype=torch.float)
        img = img.to(device, dtype=torch.float)

        prediction = model(img)
        prediction = prediction.detach().cpu().numpy().tolist()

        pred_width, pred_height = utils.classify_one(prediction[0])

        print(
            f"Image: {image_filename}, predicted width = {pred_width} , predicted height = {pred_height}"
        )

        pred_labels.append(str(pred_width) + "x" + str(pred_height))

    df = pd.DataFrame(pred_labels, columns=["pred_label"])
    grouped_df = pd.DataFrame(
        df.groupby("pred_label").apply(np.size), columns=["count"]
    )
    print(grouped_df)
