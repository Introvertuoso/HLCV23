import torch
from torchvision import datasets, transforms


# code was inspired from: https://github.com/hendrycks/robustness/blob/master/ImageNet-C/test.py
def imagenet_c_dataloader(project_root='..', corruption_name='gaussian_noise', severity=3, batch_size=64,
                          num_workers=0):
    # TODO: Add more corruptions and modify docstring
    """
    Returns a pytorch DataLoader object of the imagenet-c images using the pytorch ImageFolder convention
    :param project_root: Path to the root of the project (parent directory of the `data` folder)
    :param corruption_name: Corruption type (only gaussian, impulse or shot noise available)
    :param severity: Noise severity (1-5)
    :param batch_size: Suitable batch size to train a model on the data
    :param num_workers: Number of subprocesses to load the data
    :return: pytorch DataLoader object
    """
    # The mean and std of the imagenet-c images across corruptions; used to normalize the images
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Dataset object using the ImageFolder convention with crop and normalization applied
    distorted_dataset = datasets.ImageFolder(
        root=f'{project_root}/data/imagenet-c/' + corruption_name + '/' + str(severity),
        transform=transforms.Compose(
            [transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)]))

    # Dataloader from the Dataset object above provided with the pass-through arguments
    return torch.utils.data.DataLoader(
        distorted_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


# Inspired from the imagenet_c_dataloader() code
def tiny_imagenet_c_dataloader(project_root='..', corruption_name='gaussian_noise', severity=3, batch_size=64,
                               num_workers=0):
    """
    Returns a pytorch DataLoader object of the tiny-imagenet-c images using the pytorch ImageFolder convention
    :param project_root: Path to the root of the project (parent directory of the `data` folder)
    :param corruption_name: Corruption type (only gaussian, impulse or shot noise available)
    :param severity: Noise severity (1-5)
    :param batch_size: Suitable batch size to train a model on the data
    :param num_workers: Number of subprocesses to load the data
    :return: pytorch DataLoader object
    """
    # The mean and std of the "imagenet-c" images across corruptions; used to normalize the images
    # Not sure of the those values work the same for tiny-image-net-c
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Dataset object using the ImageFolder convention with crop and normalization applied
    distorted_dataset = datasets.ImageFolder(
        root=f'{project_root}/data/tiny-imagenet-c/' + corruption_name + '/' + str(severity),
        transform=transforms.Compose(
            [transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)]))

    # Dataloader from the Dataset object above provided with the pass-through arguments
    return torch.utils.data.DataLoader(
        distorted_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
