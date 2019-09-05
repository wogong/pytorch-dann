"""Dataset setting and data loader for syn-digits."""

import os
import torch
from torchvision import datasets, transforms
import torch.utils.data as data


def get_syndigits(dataset_root, batch_size, train):
    """Get synth digits datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # datasets and data loader
    if train:
        syndigits_dataset = datasets.ImageFolder(os.path.join(dataset_root, 'TRAIN_separate_dirs'), transform=pre_process)
    else:
        syndigits_dataset = datasets.ImageFolder(os.path.join(dataset_root, 'TEST_separate_dirs'), transform=pre_process)

    syndigits_dataloader = torch.utils.data.DataLoader(
        dataset=syndigits_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return syndigits_dataloader