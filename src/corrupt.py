# This uses the imagenet_c package; installed from
# https://github.com/hendrycks/robustness/tree/master/ImageNet-C/imagenet_c

import os

import numpy as np

from imagenet_c import corrupt
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import transforms

source = '../data/caltech-101'
target = '../data/caltech-101-c'

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ]
)

if not os.path.exists(target):
    os.mkdir(target)

for noise in tqdm(['gaussian_noise', 'impulse_noise', 'shot_noise', 'speckle_noise'], desc='Noise types'):
    dir = os.path.join(target, noise)
    if not os.path.exists(dir):
        os.mkdir(dir)
    for sev in tqdm([1, 2, 3, 4, 5], leave=False, desc='Severity levels'):
        subdir = os.path.join(dir, str(sev))
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        for cls in tqdm(os.listdir(source), leave=False, desc='Classes'):
            label = os.path.join(subdir, cls)
            if not os.path.exists(label):
                os.mkdir(label)
            input = os.path.join(source, cls)
            for img in tqdm(os.listdir(input), leave=False, desc='Images'):
                save_path = os.path.join(label, img)
                # remove this condition if you wish to overwrite
                if not os.path.exists(save_path):
                    image = Image.open(os.path.join(input, img)).convert('RGB')
                    image = transform(image)
                    image = np.array(image)
                    image = corrupt(image, severity=sev, corruption_name=noise)
                    Image.fromarray(image).save(save_path, quality=85, optimize=True)
