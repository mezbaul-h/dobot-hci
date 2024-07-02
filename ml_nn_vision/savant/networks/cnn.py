import numpy as np
import torch.nn as nn
import torch.optim as optim
from savant.settings import TORCH_DEVICE
from torch.functional import F

from .common import ModelBase, RunnerBase


class PrintShape(nn.Module):
    def forward(self, x):
        print("Shape:", x.shape)
        print("---")
        return x


class CNNModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        conv_layers = []

        # feature_dimension = (21, 2)
        feature_dimension = (33, 2)

        num_conv_layers = 2
        conv_in_channels = 1
        conv_out_channels = 16

        for _ in range(num_conv_layers):
            conv_layer = [
                nn.Conv2d(
                    in_channels=conv_in_channels, out_channels=conv_out_channels, kernel_size=3, stride=1, padding=1
                ),
                nn.ReLU(),
                # NOTE: No max-pooling, input dimension is already low.
                # nn.MaxPool2d(kernel_size=2, stride=2),
            ]

            conv_layers.extend(conv_layer)

            # Update in and out channels.
            conv_in_channels = conv_out_channels
            conv_out_channels = conv_in_channels * 2

        self.net = nn.Sequential(
            *conv_layers,
            nn.Flatten(),
            # conv_in_channels is the number of output channels from the final conv layer.
            nn.Linear(in_features=conv_in_channels * feature_dimension[0] * feature_dimension[1], out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.out_features),
        )

    def forward(self, x):
        # print(x.unsqueeze(1))
        # print(x.unsqueeze(1).shape)
        x = self.net(x.unsqueeze(1))

        return x


class CNNRunner(RunnerBase):
    def initialize(self):
        super().initialize()

        self.model = CNNModel(in_features=self.in_features, out_features=self.out_features).to(TORCH_DEVICE)

        # Loss function
        self.criterion = nn.CrossEntropyLoss().to(TORCH_DEVICE)

        # Optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def predict(self, features):
        logits = super().predict(features)
        probabilities = F.softmax(logits, dim=1)

        # This dict is sorted.
        label_probabilities = self.generate_label_probabilities(probabilities)

        return label_probabilities

    def train_step(self, features, target_outputs):
        calculated_outputs = self.model(features)
        loss = self.criterion(calculated_outputs, target_outputs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
