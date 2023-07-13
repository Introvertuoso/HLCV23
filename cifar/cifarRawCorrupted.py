import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset



# Specify the directory path to download the cifar-10-c
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

# Create an instance of the CustomDataset
dataset = CustomDataset(num_images=num_images, corruption_type=corruption_type)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

