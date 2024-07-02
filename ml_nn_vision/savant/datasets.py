import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from wisdom.settings import BODY_POSE_LANDMARK_COUNT, TRAINING_DATA_DIR

from .utils import one_hot_encode_classes, pad_numpy_array, to_tensors


class Dataset:
    def __init__(self, **kwargs):
        self.out_features = None
        self.in_features = None
        self.label_mappings = None

    def get_label_mappings(self):
        return self.label_mappings

    def _make_dataset(self):
        X = []
        y = []

        for recorder_paradigm in TRAINING_DATA_DIR.glob("*"):
            if recorder_paradigm.is_dir():
                for gesture in recorder_paradigm.glob("*"):
                    for data_file in gesture.glob("*.npy"):
                        flattened_features = np.load(data_file)

                        # Reshape the flat array into a 2D array.
                        features = flattened_features.reshape(-1, 2)

                        if features.shape[0] < BODY_POSE_LANDMARK_COUNT:
                            features = pad_numpy_array(features, (BODY_POSE_LANDMARK_COUNT, features.shape[1]))

                        X.append(features)
                        y.append(gesture.name)

        class_names = list(set(y))

        self.out_features = len(class_names)

        encoded_classes = one_hot_encode_classes(class_names)

        self.label_mappings = encoded_classes

        for i in range(len(y)):
            y[i] = encoded_classes[y[i]]

        return np.array(X), np.array(y)

    def _lazy_initialization(self):
        self._make_dataset()

    def process(self):
        # self._lazy_initialization()

        X, y = self._make_dataset()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.4)

        X_train, X_validation, X_test = to_tensors(X_train, X_validation, X_test)

        y_train, y_validation, y_test = to_tensors(
            y_train,
            y_validation,
            y_test,
            dtype=torch.float,
        )

        train_loader, validation_loader, test_loader = self.make_dataloaders(
            (X_train, y_train, {"batch_size": 64, "shuffle": True}),
            (X_validation, y_validation, {"batch_size": 64, "shuffle": True}),
            (X_test, y_test, {"batch_size": 64, "shuffle": False}),
        )

        return train_loader, validation_loader, test_loader

    @staticmethod
    def make_dataloaders(*Xy_pairs):
        # create data loaders
        dataloaders = []

        for x, y, loader_params in Xy_pairs:
            dataset = TensorDataset(x, y)
            dataloader = DataLoader(dataset=dataset, **loader_params)

            dataloaders.append(dataloader)

        return dataloaders
