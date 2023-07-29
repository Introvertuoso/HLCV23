import torch
import os

import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from src_imagebind import define_model, get_transform
from imagebind import data
from tqdm.notebook import tqdm
from imagenet_labels import lab_dict
from imagebind.models.imagebind_model import ModalityType

device = "cuda" if torch.cuda.is_available() else device
print(device)

model = define_model(device)
transform = get_transform()

cls_names = [lab_dict[i].replace('_', ' ')
             for i in os.listdir('/local/musaeed/HLCV23/data/imagenet')]
text = data.load_and_transform_text(cls_names, device)

inputs = {
    ModalityType.TEXT: text,
}
with torch.no_grad():
    embeddings = model(inputs)
text_features = embeddings[ModalityType.TEXT]

model = model.to(device)


# code was inspired from: https://github.com/hendrycks/robustness/blob/master/ImageNet-C/test.py
def imagenet_c_dataloader(project_root='/local/musaeed/HLCV23', corruption_name='gaussian_noise', severity=1, batch_size=16,
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

    list = [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    if transform is not None:
        list = transform.transforms

    # Dataset object using the ImageFolder convention with crop and normalization applied
    distorted_dataset = datasets.ImageFolder(
        root=f'{project_root}/data/imagenet-c/' +
        corruption_name + '/' + str(severity),
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


def get_acc(gt, preds=None):
    if preds is not None:
        return ((preds.argmax(1) == gt).sum()/len(preds)).cpu().numpy()
    return ((preds.argmax(1) == gt).sum()/len(preds)).cpu().numpy()


def get_test_acc(model, loader, device='cuda'):
    eval_acc = []
    for batch in tqdm(loader, leave=False):
        ims, labels = batch
        ims, labels = ims.to(device), labels.to(device)
        inputs = {
            ModalityType.VISION: ims,
        }
        with torch.no_grad():
            embeddings = model(inputs)
        image_features = embeddings[ModalityType.VISION].to(device)
        probs = (image_features @ text_features.T).softmax(dim=-1)

        val_acc = get_acc(labels.view(-1,), probs)
        eval_acc.append(val_acc)

    return np.mean(eval_acc)


gaussian_noise_acc = []
for sev in tqdm([1, 2, 3, 4, 5]):
    loader = imagenet_c_dataloader(
        corruption_name='gaussian_noise', severity=sev, batch_size=256, transform=transform)
    gaussian_noise_acc.append(get_test_acc(model, loader, device))

print(f"gaussian_noise_acc is {gaussian_noise_acc}")


impulse_noise_acc = []
for sev in tqdm([1, 2, 3, 4, 5]):
    loader = imagenet_c_dataloader(
        corruption_name='impulse_noise', severity=sev, batch_size=256, transform=transform)
    impulse_noise_acc.append(get_test_acc(model, loader, device))
print(f"impulse noise accuracy {impulse_noise_acc}")

shot_noise_acc = []
for sev in tqdm([1, 2, 3, 4, 5]):
    loader = imagenet_c_dataloader(
        corruption_name='shot_noise', severity=sev, batch_size=256, transform=transform)
    shot_noise_acc.append(get_test_acc(model, loader, device))

print(f"shot_noise_acc  accuracy {shot_noise_acc}")

speckle_noise_acc = []
for sev in tqdm([1, 2, 3, 4, 5]):
    loader = imagenet_c_dataloader(
        corruption_name='speckle_noise', severity=sev, batch_size=256, transform=transform)
    speckle_noise_acc.append(get_test_acc(model, loader, device))


print(f"speckle_noise_acc  accuracy {speckle_noise_acc}")
