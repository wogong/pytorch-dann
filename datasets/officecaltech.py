"""Dataset setting and data loader for Office_Caltech_10."""

import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import os
import params


def get_officecaltech(train, category):
    """Get Office_Caltech_10 datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(params.office_image_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                          mean=params.imagenet_dataset_mean,
                                          std=params.imagenet_dataset_mean)])

    # datasets and data_loader
    officecaltech_dataset = datasets.ImageFolder(
        os.path.join(params.dataset_root, 'office_caltech_10', category),
        transform=pre_process)

    officecaltech_dataloader = torch.utils.data.DataLoader(
        dataset=officecaltech_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=8)

    return officecaltech_dataloader