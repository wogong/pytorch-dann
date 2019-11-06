"""Dataset setting and data loader for GTSRB. Pickle format and use roi info.
"""

import os
import torch
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pickle
from PIL import Image

class GTSRB(data.Dataset):
    def __init__(self, filepath, transform=None):
        with open(filepath,'rb') as f:
            self.data = pickle.load(f)
        self.keys = ['images', 'labels']
        self.images = self.data[self.keys[0]]
        self.labels = self.data[self.keys[1]]
        self.transform = transform
        self.n_data = len(self.labels)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(np.uint8(image))
        if self.transform is not None:
            image = self.transform(image)
            label = int(label)
        return image, label

    def __len__(self):
        return self.n_data

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
    gtsrb_dataset = GTSRB(os.path.join(dataset_root, 'gtsrb_train.p'), transform=pre_process)

    dataset_size = len(gtsrb_dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        #np.random.seed(random_seed)
        np.random.seed()
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    if train:
        gtsrb_dataloader_train = torch.utils.data.DataLoader(gtsrb_dataset, batch_size=batch_size, sampler=train_sampler)
        return gtsrb_dataloader_train
    else:
        gtsrb_dataloader_test = torch.utils.data.DataLoader(gtsrb_dataset, batch_size=batch_size, sampler=valid_sampler)
        return gtsrb_dataloader_test