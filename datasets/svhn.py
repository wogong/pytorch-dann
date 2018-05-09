"""Dataset setting and data loader for SVHN."""


import torch
from torchvision import datasets, transforms
import os

import params


def get_svhn(train):
    """Get SVHN datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Grayscale(),
                                      transforms.Resize(params.image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    # datasets and data loader
    if train:
        svhn_dataset = datasets.SVHN(root=os.path.join(params.dataset_root,'svhn'),
                                   split='train',
                                   transform=pre_process,
                                   download=True)
    else:
        svhn_dataset = datasets.SVHN(root=os.path.join(params.dataset_root,'svhn'),
                                   split='test',
                                   transform=pre_process,
                                   download=True)

    svhn_data_loader = torch.utils.data.DataLoader(
        dataset=svhn_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        drop_last=True)

    return svhn_data_loader
