"""
This module contains the model that will be trained
"""
import torch.nn as nn


class CustomNet(nn.Module):
    def __init__(self):
        """
        Simple CNN with the following architecture
        conv -> bn -> relu -> pool -> fc

        Final layer has two heads for predicting width and height respectively
        """
        super(CustomNet, self).__init__()

        # six filters used
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1, stride=2)
        self.bn = nn.BatchNorm2d(6)
        self.relu = nn.LeakyReLU()

        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(2100, 2)

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def get_model():
    model = CustomNet()

    return model


# m = get_model()
