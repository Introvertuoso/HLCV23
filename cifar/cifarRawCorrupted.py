import os
import urllib.request
import tarfile
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
import clip

model, preprocess = clip.load("ViT-B/32", device=device)
del model

def split_dataset(dataset, train_ratio, val_ratio, test_ratio=0.1):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = val_indices

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = val_dataset

    return train_dataset, val_dataset, test_dataset



# # Specify the file path
# file_name = "./CIFAR-10-C.tar"

# # Check if the file already exists so we don't end-up downloading them many times
# if not os.path.exists(file_name):
#     # Download CIFAR-10-C dataset
#     url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
#     urllib.request.urlretrieve(url, file_name)

# # Extract the dataset if it hasn't been extracted before
# cifar10c_dir = "./CIFAR-10-C"
# if not os.path.exists(cifar10c_dir):
#     with tarfile.open(file_name, "r") as tar:
#         tar.extractall()

# directory = "./CIFAR-10-C"

# # Get the list of files in the directory
# files = os.listdir(directory)

# # Initialize the dictionary
# corruption_images = {}

# # Iterate through the files
# for file in files:
#     # Extract the corruption type (file name without the .npy extension)
#     corruption_type = os.path.splitext(file)[0]
    
#     # Load the numpy array
#     numpy_array = np.load(os.path.join(directory, file))
#     #load the labels data
#     labels = np.load(os.path.join(directory, "labels.npy"))
    
#     # Add the numpy array and labels to the dictionary
#     corruption_images[corruption_type] = (numpy_array, labels)

# # Print the dictionary keys and shapes
# for corruption_type in corruption_images:
#     numpy_array, labels = corruption_images[corruption_type]
#     print(f"Corruption Type: {corruption_type}")
#     print("Shape:", numpy_array.shape)
#     print("Labels:", labels)
#     print()


class TestCorruptDataset(Dataset):
    def __init__(self, corruption_type="gaussian_noise", severity=2, model_name='clip'):
        self.corruption_type = corruption_type

        # Load the original CIFAR-10 dataset
        # self.cifar10_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

        # Load the corruption images
        directory = "../cifar/CIFAR-10-C"
        files = os.listdir(directory)
        self.corruption_images = {}
        for file in files:
            corruption_type = os.path.splitext(file)[0]
            numpy_array = np.load(os.path.join(directory, file))
            self.corruption_images[corruption_type] = numpy_array
        
        self.corruption_images_list = self.corruption_images[corruption_type].copy()
        self.labels = self.corruption_images['labels'].copy()
        self.labels = self.labels[(severity-1)*10000: severity*10000]
        self.corruption_images_list = self.corruption_images_list[(severity-1)*10000: severity*10000]
        del  self.corruption_images[corruption_type]
         # Load the original CIFAR-10 dataset
        if model_name=='clip':
            data_transform = preprocess #transforms.Compose([transforms.ToTensor(),])
        elif model_name=='resnet':
            data_transform = transforms.Compose([ transforms.Resize(224), transforms.ToTensor(),
                                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            data_transform = transforms.Compose([transforms.ToTensor(),])
        
        self.transform = data_transform


    def __len__(self):
        return len(self.corruption_images_list)

    


    def __getitem__(self, idx):
      # Generate random indexes with the same size as the number of images
    #   random_indexes = np.random.randint(len(self.cifar10_dataset), size=self.num_images)
      
      # Initialize empty lists for images, corrupted images, and labels
  

      # Iterate through the random indexes
    #   for index in random_indexes:

            # Select the corresponding corruption image
        corruption_image = self.corruption_images_list[idx]
        label = self.labels[idx]

            # Use the same index for original and corrupted images
        #   corrupted_image = corruption_image[index]

        #   images.append(image)
        #   corrupted_images.append(corrupted_image)
        #   labels.append(label)
        return self.transform(corruption_image), label




class TestDataset(Dataset):
    def __init__(self, model_name):

         # Load the original CIFAR-10 dataset
        if model_name=='clip':
            data_transforms = preprocess #transforms.Compose([ transforms.ToTensor(),])
        elif model_name=='resnet':
            data_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            data_transforms = transforms.Compose([transforms.ToTensor(),])
        self.cifar10_dataset = CIFAR10(root='../cifar/data', train=False, download=True, transform=data_transforms)



    def __len__(self):
        return len(self.cifar10_dataset)



    def __getitem__(self, idx):
      # Generate random indexes with the same size as the number of images
    #   random_indexes = np.random.randint(len(self.cifar10_dataset), size=self.num_images)
      

      # Iterate through the random indexes
    #   for index in random_indexes:
        image, label = self.cifar10_dataset[idx]

            # Select the corresponding corruption image

            # Use the same index for original and corrupted images
        #   corrupted_image = corruption_image[index]

        #   images.append(image)
        #   corrupted_images.append(corrupted_image)
        #   labels.append(label)
        return image, label

class TrainDataset(Dataset):
    def __init__(self, model_name):

        # Load the original CIFAR-10 dataset
        if model_name=='clip':
            data_transform = preprocess #transforms.Compose([
        #  transforms.RandomHorizontalFlip(),transforms.ToTensor(),])
        elif model_name=='resnet':
            data_transform = transforms.Compose([ transforms.Resize(224),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            data_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),transforms.ToTensor(),])
        self.cifar10_dataset = CIFAR10(root='../cifar/data', train=True, download=True, transform=data_transform)



    def __len__(self):
        return len(self.cifar10_dataset)



    def __getitem__(self, idx):
      # Generate random indexes with the same size as the number of images
    #   random_indexes = np.random.randint(len(self.cifar10_dataset), size=self.num_images)
      

      # Iterate through the random indexes
    #   for index in random_indexes:
        image, label = self.cifar10_dataset[idx]

            # Select the corresponding corruption image

            # Use the same index for original and corrupted images
        #   corrupted_image = corruption_image[index]

        #   images.append(image)
        #   corrupted_images.append(corrupted_image)
        #   labels.append(label)
        return image, label


class CustomDataset(Dataset):
    def __init__(self, num_images=10, corruption_type="gaussian_noise"):
        self.num_images = num_images
        self.corruption_type = corruption_type

        # Load the original CIFAR-10 dataset
        self.cifar10_dataset = CIFAR10(root='../cifar/data', train=False, download=True, transform=transforms.ToTensor())

        # Load the corruption images
        directory = "CIFAR-10-C"
        files = os.listdir(directory)
        self.corruption_images = {}
        for file in files:
            corruption_type = os.path.splitext(file)[0]
            numpy_array = np.load(os.path.join(directory, file))
            self.corruption_images[corruption_type] = numpy_array
        
        self.corruption_images_list = self.corruption_images[corruption_type].copy()
        del  self.corruption_images[corruption_type]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.cifar10_dataset)

    


    def __getitem__(self, idx):
      # Generate random indexes with the same size as the number of images
    #   random_indexes = np.random.randint(len(self.cifar10_dataset), size=self.num_images)
      
      # Initialize empty lists for images, corrupted images, and labels
        images = []
        corrupted_images = []
        labels = []

      # Iterate through the random indexes
    #   for index in random_indexes:
        image, label = self.cifar10_dataset[idx]

            # Select the corresponding corruption image
        corruption_image = self.corruption_images_list[idx]

            # Use the same index for original and corrupted images
        #   corrupted_image = corruption_image[index]

        #   images.append(image)
        #   corrupted_images.append(corrupted_image)
        #   labels.append(label)
        return image, self.transform(corruption_image), label


# Specify the number of images to generate
num_images = 10

# Specify the corruption type (e.g., 'gaussian_noise', 'motion_blur', etc.)
corruption_type = 'gaussian_noise'

# # Create an instance of the CustomDataset
# dataset = CustomDataset(num_images=num_images, corruption_type=corruption_type)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)



def get_original_loaders(model_name='clip', batch_size=64):
    trainset = TrainDataset(model_name=model_name)
    test_dataset = TestDataset(model_name=model_name)

    # # now we split the dataset into train, validation, and test sets
    train_dataset, val_dataset, _ = split_dataset(trainset, train_ratio=0.8, val_ratio=0.2)

    # Create separate DataLoader instances for train, validation, and test sets
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader
    
def get_corrupt_loaders(corruption_type='gaussian_noise', severity=2, batch_size=64, model_name='clip'):

    # Create an instance of the CustomDataset

    test_corrupt_dataset = TestCorruptDataset(corruption_type, severity=severity, model_name=model_name)
    test_corrup_dataloader = torch.utils.data.DataLoader(test_corrupt_dataset, batch_size=batch_size, shuffle=False)
    return test_corrup_dataloader
    


