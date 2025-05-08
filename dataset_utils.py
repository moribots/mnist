"""
File: dataset_utils.py
Description: Utilities for loading and preparing the MNIST dataset.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import multiprocessing


# Get MNIST dataloaders.
def get_mnist_dataloaders(batch_size=64, data_dir='./data'):
    """
    Prepares and returns the MNIST dataloaders for training and testing.
    Args:
            batch_size (int): The batch size for the dataloaders.
            data_dir (str): Directory where the MNIST dataset will be stored.

    Returns:
            tuple (DataLoader, DataLoader): The training and testing dataloaders.
    """
    # --------------------------------------------------------------------------
    # 1. Define Transformations
    # --------------------------------------------------------------------------
    # Transformations are applied to the dataset images.
    # - transforms.ToTensor(): Converts a PIL Image or numpy.ndarray (H x W x C)
    #   in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    # - transforms.Normalize((mean,), (std,)): Normalizes a tensor image with mean and standard deviation.
    #   For MNIST, the images are grayscale, so they have one channel.
    #   The values 0.1307 and 0.3081 are commonly used mean and std for MNIST.
    #   Normalization helps the model converge faster.
    # --------------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # --------------------------------------------------------------------------
    # 2. Download and Load Datasets
    # --------------------------------------------------------------------------
    # `torchvision.datasets.MNIST` handles downloading and loading the dataset.
    # - `root`: The directory where the data will be stored (or is already stored).
    # - `train`: If True, creates dataset from training.pt, otherwise from test.pt.
    # - `download`: If True, downloads the dataset from the internet if it's not available at `root`.
    # - `transform`: Apply the defined transformations to each image.
    # --------------------------------------------------------------------------
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # --------------------------------------------------------------------------
    # 3. Create DataLoaders
    # --------------------------------------------------------------------------
    # `torch.utils.data.DataLoader` wraps an iterable over a dataset.
    # It provides features like:
    # - Batching the data.
    # - Shuffling the data (important for training to ensure batches are diverse).
    # - Loading data in parallel using multiple workers (`num_workers`).
    # --------------------------------------------------------------------------
    cpu_count = multiprocessing.cpu_count()
    num_workers = cpu_count // 2  # Use half of the available CPU cores for data loading
    # (to avoid overloading the system).
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    print(
        f"MNIST dataset loaded. Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}, Number of workers: {num_workers}")
    print(
        f"Train Dataloader batch size: {batch_size}, Test Dataloader batch size: {batch_size}, Number of workers: {num_workers}")
    return train_loader, test_loader


if __name__ == '__main__':
    # This is a simple test to see if the dataloaders work.
    # You can run `python dataset_utils.py` to test this file independently.
    print("Testing MNIST DataLoaders...")
    train_loader, test_loader = get_mnist_dataloaders(batch_size=4)

    # Fetch one batch from the train_loader
    images, labels = next(iter(train_loader))
    print("\nFetched one batch from train_loader:")
    # Expected: [batch_size, 1, 28, 28] for MNIST
    print(f"Images batch shape: {images.shape}")
    print(f"Labels batch shape: {labels.shape}")  # Expected: [batch_size]
    print(f"First image shape: {images[0].shape}")
    print(f"Label for first image: {labels[0]}")

    # Fetch one batch from the test_loader
    images_test, labels_test = next(iter(test_loader))
    print("\nFetched one batch from test_loader:")
    print(f"Images batch shape: {images_test.shape}")
    print(f"Labels batch shape: {labels_test.shape}")

    print("\n`dataset_utils.py` test complete.")
