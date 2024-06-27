import os

import torch

from torchvision import datasets, transforms

# code was inspired from: https://github.com/hendrycks/robustness/blob/master/ImageNet-C/test.py
def corrupt(project_root, corruption_name='gaussian_noise', severity=1, batch_size=64,
          num_workers=1, shuffle=False, transform=None):
    """
    Returns a pytorch DataLoader object of the imagenet-c images using the pytorch ImageFolder convention
    :param project_root: Path to the root of the project (parent directory of the `data` folder)
    :param corruption_name: Corruption type (only speckle, gaussian, impulse or shot noise available)
    :param severity: Noise severity (1-5)
    :param batch_size: Suitable batch size to train a model on the data
    :param num_workers: Number of subprocesses to load the data
    :return: pytorch DataLoader object
    """
    # The mean and std of the imagenet-c images across corruptions; used to normalize the images
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    tlist = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    if transform is not None:
        tlist = transform

    # Dataset object using the ImageFolder convention with crop and normalization applied
    distorted_dataset = datasets.ImageFolder(
        root=os.path.join(project_root, 'data', 'tiny-c', 'val', corruption_name, str(severity)),
        transform=tlist
    )

    # Dataloader from the Dataset object above provided with the pass-through arguments
    return torch.utils.data.DataLoader(
        distorted_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    ), len(distorted_dataset.classes)

def clean(project_root, split='val', batch_size=64, num_workers=1, shuffle=False, transform=None):
    """
    Returns a pytorch DataLoader object of the imagenet images using the pytorch ImageFolder convention
    :param project_root: Path to the root of the project (parent directory of the `data` folder)
    :param batch_size: Suitable batch size to train a model on the data
    :param num_workers: Number of subprocesses to load the data
    :return: pytorch DataLoader object
    """
    # The mean and std of the imagenet images; used to normalize the images
    # The same mean and std were used by Hendrycks
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    tlist = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if transform is not None:
        tlist = transform

    # Dataset object using the ImageFolder convention with crop and normalization applied
    dataset = datasets.ImageFolder(
        root=os.path.join(project_root, 'data', 'tiny', split),
        transform=tlist
    )

    # Dataloader from the Dataset object above provided with the pass-through arguments
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )