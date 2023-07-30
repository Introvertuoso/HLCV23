import os
from statistics import median, mean, mode

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


stats(path_to_caltech)

# for noise in ['gaussian_noise', 'impulse_noise', 'shot_noise', 'speckle_noise']:
#     dir = os.path.join(path_to_caltech_c, noise)
#     for sev in [1, 2, 3, 4, 5]:
#         subdir = os.path.join(dir, str(sev))
#         stats(subdir)
