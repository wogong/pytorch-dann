"""Dataset setting and data loader for Office."""

import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import os
import params


def get_office(train, category):
    """Get Office datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(params.office_image_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                          mean=params.imagenet_dataset_mean,
                                          std=params.imagenet_dataset_mean)])

    # datasets and data_loader
    office_dataset = datasets.ImageFolder(
        os.path.join(params.dataset_root, 'office', category, 'images'),
        transform=pre_process)

    office_dataloader = torch.utils.data.DataLoader(
        dataset=office_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=8)

    return office_dataloader