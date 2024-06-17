import torch
from torchvision import datasets, transforms


# code was inspired from: https://github.com/hendrycks/robustness/blob/master/ImageNet-C/test.py
def imagenet_c_dataloader(project_root='../imagenet_data', corruption_name='gaussian_noise', severity=1, batch_size=64,
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
        root=f'{project_root}/' + corruption_name + '/' + str(severity),
        transform=tlist
    )

    # Dataloader from the Dataset object above provided with the pass-through arguments
    return torch.utils.data.DataLoader(
        distorted_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def imagenet_dataloader(project_root='../imagenet_data', batch_size=64, num_workers=1, shuffle=False, transform=None):
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
        root=f'{project_root}/imagenet/',
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


def caltech_c_dataloader(project_root='..', corruption_name='gaussian_noise', severity=1, batch_size=64,
                          num_workers=1, shuffle=False, transform=None):
    """
    Returns a pytorch DataLoader object of the caltech-c images using the pytorch ImageFolder convention
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

    list = [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    if transform is not None:
        list = transform.transforms

    # Dataset object using the ImageFolder convention with crop and normalization applied
    distorted_dataset = datasets.ImageFolder(
        root=f'{project_root}/data/caltech-101-c/' + corruption_name + '/' + str(severity),
        transform=transforms.Compose(
            list
        )
    )

    # Dataloader from the Dataset object above provided with the pass-through arguments
    return torch.utils.data.DataLoader(
        distorted_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def caltech_dataloader(project_root='..', batch_size=64, num_workers=1, shuffle=False, transform=None):
    """
    Returns a pytorch DataLoader object of the caltech images using the pytorch ImageFolder convention
    :param project_root: Path to the root of the project (parent directory of the `data` folder)
    :param batch_size: Suitable batch size to train a model on the data
    :param num_workers: Number of subprocesses to load the data
    :return: pytorch DataLoader object
    """
    # The mean and std of the imagenet images; used to normalize the images
    # The same mean and std were used by Hendrycks
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    if transform is not None:
        list = transform.transforms

    # Dataset object using the ImageFolder convention with crop and normalization applied
    dataset = datasets.ImageFolder(
        root=f'{project_root}/data/caltech-101/',
        transform=transforms.Compose(
            list
        )
    )

    # Dataloader from the Dataset object above provided with the pass-through arguments
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

# Inspired from the imagenet_c_dataloader() code
# def tiny_imagenet_c_dataloader(project_root='..', corruption_name='gaussian_noise', severity=1, batch_size=64,
#                                num_workers=1):
#     """
#     Returns a pytorch DataLoader object of the tiny-imagenet-c images using the pytorch ImageFolder convention
#     :param project_root: Path to the root of the project (parent directory of the `data` folder)
#     :param corruption_name: Corruption type (only gaussian, impulse or shot noise available)
#     :param severity: Noise severity (1-5)
#     :param batch_size: Suitable batch size to train a model on the data
#     :param num_workers: Number of subprocesses to load the data
#     :return: pytorch DataLoader object
#     """
#     # The mean and std of the "imagenet-c" images across corruptions; used to normalize the images
#     # Not sure of the those values work the same for tiny-image-net-c
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#
#     # Dataset object using the ImageFolder convention with crop and normalization applied
#     distorted_dataset = datasets.ImageFolder(
#         root=f'{project_root}/data/tiny-imagenet-c/' + corruption_name + '/' + str(severity),
#         transform=transforms.Compose(
#             [transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)]))
#
#     # Dataloader from the Dataset object above provided with the pass-through arguments
#     return torch.utils.data.DataLoader(
#         distorted_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
