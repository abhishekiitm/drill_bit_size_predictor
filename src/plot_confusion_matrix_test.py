"""
This module plots the confusion matrix on the test set
I have used this as a sanity check to ensure regression is able to classify correctly
"""

import os

import pandas as pd
import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from model import get_model
import dataset
import engine
import utils


if __name__ == "__main__":

    BATCH_SIZE = 32

    data_dir = "data_generated_images"
    model_dir = "models"

    device = "cpu"

    model = get_model()
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))

    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    test_images = test_df.image_path.values
    test_widths = test_df.test_widths.values.astype(float)
    test_heights = test_df.test_heights.values.astype(float)
    test_targets = np.c_[test_widths, test_heights]

    test_dataset = dataset.CustomDataset(
        image_paths=test_images,
        targets=test_targets,
        resize=None,
        augmentations=None,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
    )
    predictions, test_targets = engine.evaluate(test_loader, model, device=device)

    predicted_classes = utils.classify_all(predictions)

    pred_labels = [str(x[0]) + "_" + str(x[1]) for x in predicted_classes]
    test_labels = [str(int(x[0])) + "_" + str(int(x[1])) for x in test_targets]

    cf_matrix = confusion_matrix(test_labels, pred_labels)

    print(cf_matrix)

    ax = sns.heatmap(cf_matrix, annot=True, cmap="Blues")

    ax.set_title("Seaborn Confusion Matrix with labels\n\n")
    ax.set_xlabel("\nPredicted Values")
    ax.set_ylabel("Actual Values ")

    ## Ticket labels - List must be in alphabetical order
    # ticket_labels = sorted(list(set(test_labels)))
    ax.xaxis.set_ticklabels(sorted(list(set(pred_labels))))
    ax.yaxis.set_ticklabels(sorted(list(set(pred_labels))))

    ## Display the visualization of the Confusion Matrix.
    print(test_labels)
    plt.show()
