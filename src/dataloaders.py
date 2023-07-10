from torchvision import datasets, transforms
import torch


def imagenet_c_dataloader(project_root='..', noise_name='gaussian_noise', severity=3, batch_size=64, num_workers=0):
    # TODO: Extend to other corruptions, only change noise_name and adjust path after adding the other corruptions
    """
    Loads the imagenet-c images using the pytorch ImageFolder convention
    :param project_root: The path to the root of the project (parent directory of the `data` folder)
    :param noise_name: The noise type (gaussian, impulse or shot noise)
    :param severity: Noise severity (1-5)
    :param batch_size: The suitable batch size to train a model on the data
    :param num_workers: Number of subprocesses to load the data
    :return: pytorch DataLoader object
    """
    # The mean and std of the imagenet-c images across corruptions; used to normalize the images
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Dataset object using the ImageFolder convention with crop and normalization applied
    distorted_dataset = datasets.ImageFolder(
        root=f'{project_root}/data/noise/' + noise_name + '/' + str(severity),
        transform=transforms.Compose(
            [transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)]))

    # Dataloader from the Dataset object above provided with the pass-through arguments
    return torch.utils.data.DataLoader(
        distorted_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
