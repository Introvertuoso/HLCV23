import os
import numpy as np
from tqdm.notebook import tqdm
from imagenet_labels import lab_dict
from lavis.processors.blip_processors import BlipCaptionProcessor
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image  # Import the PIL library explicitly
import matplotlib.pyplot as plt
# from src.dataloaders import imagenet_c_dataloader, imagenet_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Transforms for the input data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    # Resize the image to (224, 224) to match the ViT input size
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])


# Function to load images using PIL explicitly
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# Load the ImageNet dataset
data_dir = '/local/musaeed/HLCV23/data/imagenet'
dataset = datasets.ImageFolder(
    data_dir, transform=transform, loader=pil_loader)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load the ViT model and feature extractor
# ViT base model pre-trained on 21k classes
model_name = "google/vit-base-patch16-224-in21k"
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)
model = model.to(device)

cls_names = [lab_dict[i].replace('_', ' ') for i in os.listdir(
    '/local/musaeed/HLCV23/data/imagenet')]

text_processor = BlipCaptionProcessor(prompt="A picture of ")
cls_prompt = [text_processor(cls_nm) for cls_nm in cls_names]
print(cls_prompt)


def get_acc(gt, preds):
    # The function calculates the accuracy of the predictions
    # gt - ground truth labels
    # preds - model predictions
    return ((preds.argmax(1) == gt).sum() / len(preds)).cpu().numpy()


def get_test_acc(model, loader, device='cuda'):
    model.eval()
    eval_acc = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            ims, labels = batch
            ims, labels = ims.to(device), labels.to(device)

            outputs = model(ims)  # Directly use images

            val_acc = get_acc(labels.view(-1,), outputs.logits)
            eval_acc.append(val_acc)

    return np.mean(eval_acc)


# print(get_test_acc(model, loader, device))


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
