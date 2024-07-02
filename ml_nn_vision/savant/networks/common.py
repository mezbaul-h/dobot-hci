import abc
import csv
import time
import typing

import matplotlib.pyplot as plt
import numpy
import pandas
import torch
import torch.nn as nn
from savant.settings import TORCH_DEVICE
from savant.utils import to_tensors


class ModelBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        print("Initializing Model:", self.__class__.__name__)

        self.in_features = kwargs["in_features"]
        self.out_features = kwargs["out_features"]


class RunnerBase(abc.ABC):
    def __init__(self, **kwargs):
        print("Initializing Runner:", self.__class__.__name__)

        self._initialized = False
        self.criterion: typing.Optional[nn.Module] = None
        self.current_epoch = 0
        self.epoch_losses = {
            "training": [],
            "validation": [],
        }
        self.in_features = kwargs.get("in_features")
        self.label_mappings = kwargs.get("label_mappings")
        self.learning_rate = kwargs.get("learning_rate")
        self.model: typing.Optional[ModelBase] = None
        self.momentum = kwargs.get("momentum")
        self.num_epochs = kwargs.get("num_epochs")
        self.num_gates = kwargs.get("num_gates")
        self.optimizer: typing.Optional[nn.Module] = None
        self.out_features = kwargs.get("out_features")

    def evaluate(self, dataloader):
        ...

    @staticmethod
    def get_best_epoch(epoch_losses, default):
        try:
            return epoch_losses.index(min(epoch_losses))
        except ValueError:
            return default

    def initialize(self):
        if not self._initialized:
            self._initialized = True

    def load_state(self, state_filename):
        checkpoint = torch.load(state_filename)

        # NOTE: These parameters must be initialized first before calling `initialize()`.
        self.in_features = checkpoint["in_features"]
        self.learning_rate = checkpoint["learning_rate"]
        self.momentum = checkpoint["momentum"]
        self.num_gates = checkpoint["num_gates"]
        self.out_features = checkpoint["out_features"]

        self.initialize()

        self.criterion.load_state_dict(checkpoint["criterion_state"])
        self.current_epoch = checkpoint["current_epoch"]
        self.epoch_losses = checkpoint["epoch_losses"]
        self.label_mappings = checkpoint["label_mappings"]
        self.model.load_state_dict(checkpoint["model_state"])
        self.num_epochs = checkpoint["num_epochs"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

    def save_loss_plot(self, target_filename="loss_plot.png"):
        # Extract losses for training and validation.
        training_losses = self.epoch_losses["training"]
        best_training_epoch_index = self.get_best_epoch(training_losses, self.current_epoch)
        validation_losses = self.epoch_losses["validation"]
        best_validation_epoch_index = self.get_best_epoch(validation_losses, self.current_epoch)

        # Set the figure size to 1000 x 600 pixels.
        plt.figure(figsize=(10, 6))

        # Create training line plot.
        plt.plot(range(1, len(training_losses) + 1), training_losses, color="blue", label="Training Loss")
        plt.axvline(
            x=best_training_epoch_index + 1,
            color="blue",
            label=f"Best Training Epoch: {best_training_epoch_index + 1}",
        )

        # Create validation line plot.
        plt.plot(range(1, len(validation_losses) + 1), validation_losses, color="orange", label="Validation Loss")
        plt.axvline(
            x=best_validation_epoch_index + 1,
            color="orange",
            label=f"Best Validation Epoch: {best_validation_epoch_index + 1}",
        )

        # Set labels and title.
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs\n" f"Learning Rate: {self.learning_rate}")

        # Show the legend.
        plt.legend()

        # Save the plot to an image file.
        plt.savefig(target_filename)

    def predict(self, features):
        self.model.eval()

        with torch.no_grad():
            calculated_outputs = self.model(to_tensors(features)[0])

            return calculated_outputs

    def save_state(self, state_filename):
        torch.save(
            {
                "criterion_state": self.criterion.state_dict(),
                "current_epoch": self.current_epoch,
                "epoch_losses": self.epoch_losses,
                "in_features": self.in_features,
                "label_mappings": self.label_mappings,
                "learning_rate": self.learning_rate,
                "model_state": self.model.state_dict(),
                "momentum": self.momentum,
                "num_epochs": self.num_epochs,
                "num_gates": self.num_gates,
                "optimizer_state": self.optimizer.state_dict(),
                "out_features": self.out_features,
            },
            state_filename,
        )

    @abc.abstractmethod
    def train_step(self, features, target_outputs):
        ...

    def validate(self, dataloader):
        # Calculate validation loss.
        self.model.eval()

        total_validation_loss = 0.0
        num_samples = len(dataloader.dataset)

        with torch.no_grad():
            for features, target_outputs in dataloader:
                calculated_outputs = self.model(features)

                total_validation_loss += self.criterion(calculated_outputs, target_outputs).item()

        validation_loss_avg = total_validation_loss / num_samples

        return validation_loss_avg

    def train(self, train_loader, validation_loader):
        self.initialize()

        num_samples = len(train_loader.dataset)

        while self.current_epoch < self.num_epochs:
            epoch_start_time = time.time()

            total_training_loss = 0.0

            self.model.train()

            for features, target_outputs in train_loader:
                loss = self.train_step(features, target_outputs)

                total_training_loss += loss.item()

            training_loss_avg = total_training_loss / num_samples

            validation_loss_avg = self.validate(validation_loader)

            seconds_elapsed = time.time() - epoch_start_time

            self.epoch_losses["training"].append(training_loss_avg)
            self.epoch_losses["validation"].append(validation_loss_avg)

            try:
                best_epoch_index = self.epoch_losses["training"].index(min(self.epoch_losses["training"]))
            except ValueError:
                best_epoch_index = self.current_epoch

            print(
                f"[{self.current_epoch+1}/{self.num_epochs}] "
                f"Training loss: {training_loss_avg:.10f} | "
                f"Validation loss: {validation_loss_avg:.10f} | "
                f"Best training epoch: {best_epoch_index+1} "
                f"({seconds_elapsed:.2f}s)"
            )

            self.current_epoch += 1

    def generate_label_probabilities(self, probabilities):
        # Convert probabilities tensor to a list
        probabilities_list = probabilities.squeeze().tolist()

        # Zip class labels with their corresponding probabilities
        label_probabilities = zip(self.label_mappings.keys(), probabilities_list)

        # Convert to dictionary
        label_probabilities_dict = dict(label_probabilities)

        # Sort the label_probabilities_dict based on probabilities in descending order
        sorted_label_probabilities = sorted(label_probabilities_dict.items(), key=lambda item: item[1], reverse=True)

        # Convert the sorted list of tuples back to a dictionary
        sorted_label_probabilities_dict = dict(sorted_label_probabilities)

        return sorted_label_probabilities_dict
