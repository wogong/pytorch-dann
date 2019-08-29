"""Dataset setting and data loader for GTSRB."""

import os
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def get_gtsrb(dataset_root, batch_size, train):
    """Get GTSRB datasets loader."""
    shuffle_dataset = True
    random_seed = 42
    train_size = 31367

    # image pre-processing
    pre_process = transforms.Compose([
        transforms.Resize((40, 40)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # datasets and data_loader
    gtsrb_dataset = datasets.ImageFolder(
        os.path.join(dataset_root, 'Final_Training', 'Images'), transform=pre_process)

    dataset_size = len(gtsrb_dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    gtsrb_dataloader_train = torch.utils.data.DataLoader(gtsrb_dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    gtsrb_dataloader_test = torch.utils.data.DataLoader(gtsrb_dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    return gtsrb_dataloader_train, gtsrb_dataloader_test