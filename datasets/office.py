"""Dataset setting and data loader for Office."""

import os
import torch
from torchvision import datasets, transforms
import torch.utils.data as data


def get_office(dataset_root, batch_size, category):
    """Get Office datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # datasets and data_loader
    office_dataset = datasets.ImageFolder(
        os.path.join(dataset_root, 'office', category, 'images'), transform=pre_process)

    office_dataloader = torch.utils.data.DataLoader(
        dataset=office_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return office_dataloader