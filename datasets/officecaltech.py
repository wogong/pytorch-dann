"""Dataset setting and data loader for Office_Caltech_10."""

import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import os


def get_officecaltech(dataset_root, batch_size, category):
    """Get Office_Caltech_10 datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.Resize(227),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                          mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225)
                                     )])

    # datasets and data_loader
    officecaltech_dataset = datasets.ImageFolder(
        os.path.join(dataset_root, 'office_caltech_10', category),
        transform=pre_process)

    officecaltech_dataloader = torch.utils.data.DataLoader(
        dataset=officecaltech_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    return officecaltech_dataloader