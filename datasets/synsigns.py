"""Dataset setting and data loader for syn-signs."""

import os
import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from PIL import Image


class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            data = data.split(' ')
            self.img_paths.append(data[0])
            self.img_labels.append(data[1])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data

def get_synsigns(dataset_root, batch_size, train):
    """Get Synthetic Signs datasets loader."""
    # image pre-processing
    pre_process = transforms.Compose([
        transforms.Resize((40, 40)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # datasets and data_loader
    # using first 90K samples as training set
    train_list = os.path.join(dataset_root, 'train_labelling.txt')
    synsigns_dataset = GetLoader(
        data_root=os.path.join(dataset_root),
        data_list=train_list,
        transform=pre_process)

    synsigns_dataloader = torch.utils.data.DataLoader(
        dataset=synsigns_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return synsigns_dataloader