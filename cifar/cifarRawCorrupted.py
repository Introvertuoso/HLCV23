import os
import urllib.request
import tarfile
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt



def split_dataset(dataset, train_ratio, val_ratio, test_ratio):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset



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

directory = "/content/CIFAR-10-C"

# Get the list of files in the directory
files = os.listdir(directory)

# Initialize the dictionary
corruption_images = {}

# Iterate through the files
for file in files:
    # Extract the corruption type (file name without the .npy extension)
    corruption_type = os.path.splitext(file)[0]
    
    # Load the numpy array
    numpy_array = np.load(os.path.join(directory, file))
    #load the labels data
    labels = np.load(os.path.join(directory, "labels.npy"))
    
    # Add the numpy array and labels to the dictionary
    corruption_images[corruption_type] = (numpy_array, labels)

# Print the dictionary keys and shapes
for corruption_type in corruption_images:
    numpy_array, labels = corruption_images[corruption_type]
    print(f"Corruption Type: {corruption_type}")
    print("Shape:", numpy_array.shape)
    print("Labels:", labels)
    print()



class CustomDataset(Dataset):
    def __init__(self, num_images=10, corruption_type="gaussian_noise"):
        self.num_images = num_images
        self.corruption_type = corruption_type

        # Load the original CIFAR-10 dataset
        self.cifar10_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

        # Load the corruption images
        directory = "/content/CIFAR-10-C"
        files = os.listdir(directory)
        self.corruption_images = {}
        for file in files:
            corruption_type = os.path.splitext(file)[0]
            numpy_array = np.load(os.path.join(directory, file))
            self.corruption_images[corruption_type] = numpy_array

    def __len__(self):
        return self.num_images

    


    def __getitem__(self, idx):
      # Generate random indexes with the same size as the number of images
      random_indexes = np.random.randint(len(self.cifar10_dataset), size=self.num_images)
      
      # Initialize empty lists for images, corrupted images, and labels
      images = []
      corrupted_images = []
      labels = []

      # Iterate through the random indexes
      for index in random_indexes:
          image, label = self.cifar10_dataset[index]

          # Select the corresponding corruption image
          corruption_image = self.corruption_images[self.corruption_type][0]

          # Use the same index for original and corrupted images
          corrupted_image = corruption_image[index]

          images.append(image)
          corrupted_images.append(corrupted_image)
          labels.append(label)

      return images, corrupted_images, labels


# Specify the number of images to generate
num_images = 10

# Specify the corruption type (e.g., 'gaussian_noise', 'motion_blur', etc.)
corruption_type = 'gaussian_noise'

# # Create an instance of the CustomDataset
# dataset = CustomDataset(num_images=num_images, corruption_type=corruption_type)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)




# Create an instance of the CustomDataset
dataset = CustomDataset(num_images=1024, corruption_type=corruption_type)

# now we split the dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

# Create separate DataLoader instances for train, validation, and test sets
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


