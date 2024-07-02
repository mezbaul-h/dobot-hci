import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer

from .settings import TORCH_DEVICE


def to_tensors(*args, dtype=torch.float):
    # Convert data to PyTorch tensors
    return [torch.tensor(item).to(device=TORCH_DEVICE, dtype=dtype) for item in args]


def one_hot_encode_classes(class_names):
    """
    Encode a list of class names into one-hot encoded representations.

    Parameters:
    - class_names (list): List of class names

    Returns:
    - classes (dict): Dictionary mapping class names to their one-hot encoded representations
    """
    # Create dictionary to store one-hot encoded representations
    classes = {}

    # Iterate over class names
    for idx, class_name in enumerate(class_names):
        # Create one-hot encoded array
        one_hot_encoded = np.zeros(len(class_names))
        one_hot_encoded[idx] = 1

        # Store one-hot encoded array in dictionary
        classes[class_name] = one_hot_encoded.tolist()
    return classes


def pad_numpy_array(arr, desired_shape, padding_value=-1):
    """
    Pad a numpy array to the desired shape.

    Parameters:
    - arr (numpy.ndarray): Input array to be padded.
    - desired_shape (tuple): Desired shape of the padded array.
    - padding_value (scalar, optional): Value to use for padding. Defaults to -1.

    Returns:
    - numpy.ndarray: Padded array.

    Example:
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> desired_shape = (3, 2)
    >>> padded_data = pad_numpy_array(arr, desired_shape)
    >>> print(padded_data)
    [[ 1  2]
     [ 3  4]
     [-1 -1]]
    """
    # Calculate the amount of padding needed for each dimension
    pad_width = ((0, desired_shape[0] - arr.shape[0]), (0, 0))  # Padding only along the first dimension

    # Pad the array with the specified padding value
    padded_data = np.pad(arr, pad_width, mode="constant", constant_values=padding_value)

    return padded_data
