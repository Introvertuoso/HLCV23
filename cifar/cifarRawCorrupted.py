import os
import urllib.request
import tarfile
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt


# Specify the file path
file_name = "/content/CIFAR-10-C.tar"

# Check if the file already exists so we don't end-up downloading them many times
if not os.path.exists(file_name):
    # Download CIFAR-10-C dataset
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    urllib.request.urlretrieve(url, file_name)

# Extract the dataset if it hasn't been extracted before
cifar10c_dir = "/content/CIFAR-10-C"
if not os.path.exists(cifar10c_dir):
    with tarfile.open(file_name, "r") as tar:
        tar.extractall()

# Access the images
# corruptions = os.listdir(cifar10c_dir)
# corruption_images = {}
# for corruption in corruptions:
#     corruption_images[corruption] = []
#     corruption_path = os.path.join(cifar10c_dir, corruption)
#     if os.path.isdir(corruption_path):
#         image_files = [file for file in os.listdir(corruption_path) if file.endswith('.npy')]
#         for image_file in image_files:
#             image_path = os.path.join(corruption_path, image_file)
#             image = np.load(image_path)
#             corruption_images[corruption].append(image)




cifar10c_dir = "/content/CIFAR-10-C"

# Access the images
corruptions = os.listdir(cifar10c_dir)
corruption_images = {}
for corruption in corruptions:
    corruption_images[corruption] = []
    corruption_path = os.path.join(cifar10c_dir, corruption)
    if os.path.isdir(corruption_path):
        image_files = [file for file in os.listdir(corruption_path) if file.endswith('.npy')]
        for image_file in image_files:
            image_path = os.path.join(corruption_path, image_file)
            image = np.load(image_path)
            corruption_images[corruption].append(image)



cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())


