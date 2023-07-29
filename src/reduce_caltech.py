import os

from tqdm import tqdm

path_to_caltech = '../data/caltech-101'
path_to_caltech_c = '../data/caltech-101-c'


def reduce_caltech(path):
    for cls in tqdm(os.listdir(path), leave=False, desc='Classes'):
        cls_path = os.path.join(path, cls)
        images = os.listdir(cls_path)
        if len(images) <= 59:  # This is the median as calculated in console.py
            continue
        for img in tqdm(images[59:], leave=False, desc='Images'):
            os.remove(os.path.join(cls_path, img))


reduce_caltech(path_to_caltech)

for noise in tqdm(['gaussian_noise', 'impulse_noise', 'shot_noise', 'speckle_noise'], desc='Noise types'):
    dir = os.path.join(path_to_caltech_c, noise)
    for sev in tqdm([1, 2, 3, 4, 5], leave=False, desc='Severity levels'):
        subdir = os.path.join(dir, str(sev))
        reduce_caltech(subdir)