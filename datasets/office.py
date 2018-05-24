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
                                         mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225)
                                     )])

    # datasets and data_loader
    office_dataset = datasets.ImageFolder(
        os.path.join(params.dataset_root, 'office', category, 'images'),
        transform=pre_process)

    office_dataloader = torch.utils.data.DataLoader(
        dataset=office_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=4)

    return office_dataloader