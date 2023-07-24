import os
import shutil

from tqdm import tqdm

# We rely on imagenet-c to organize imagenet
source = '../data/imagenet-c/gaussian_noise/1'
labels = os.listdir(source)
# We will reorganize the imagenet directory
target = '../data/imagenet'

# Creates directories for each of the classes
for label in labels:
    path = os.path.join(target, label)
    if not os.path.exists(path):
        os.mkdir(path)

# Moves same-label images to the same directories
for label in tqdm(labels):
    path = os.path.join(source, label)
    images = os.listdir(path)
    for image in tqdm(images, leave=False):
        image_from = os.path.join(target, image)
        image_to = os.path.join(target, label, image)
        if not os.path.exists(image_to):
            shutil.move(image_from, image_to)
