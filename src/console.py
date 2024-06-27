import os
import shutil
from statistics import median, mean, mode

import tqdm

path_to_caltech = '../data/caltech-101'
path_to_caltech_c = '../data/caltech-101-c'


def stats(path):
    lengths = []
    for cls in os.listdir(path):
        lengths.append(len(os.listdir(os.path.join(path, cls))))

    print('min: ', min(lengths))
    print('max: ', max(lengths))
    print('mean: ', mean(lengths))
    print('mode: ', mode(lengths))
    print('median: ', median(lengths))


# stats(path_to_caltech)

# for noise in ['gaussian_noise', 'impulse_noise', 'shot_noise', 'speckle_noise']:
#     dir = os.path.join(path_to_caltech_c, noise)
#     for sev in [1, 2, 3, 4, 5]:
#         subdir = os.path.join(dir, str(sev))
#         stats(subdir)

# path = '../data/tiny/val'
# for cls in tqdm.tqdm(os.listdir(path), desc='Classes'):
#     cls_dir = os.path.join(path, cls)
#     if not os.path.isdir(cls_dir): continue
#     source_dir = os.path.join(cls_dir, 'images')
#     file_names = os.listdir(source_dir)
#     for file_name in tqdm.tqdm(file_names, leave=False, desc='Images'):
#         if not file_name.endswith('.JPEG'): continue
#         shutil.move(os.path.join(source_dir, file_name), cls_dir)
#     os.rmdir(source_dir)