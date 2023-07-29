import os
from statistics import median, mean, mode

path = '../data/caltech-101'

lengths = []
for cls in os.listdir(path):
    lengths.append(len(os.listdir(os.path.join(path, cls))))

print('min: ', min(lengths))
print('max: ', max(lengths))
print('mean: ', mean(lengths))
print('mode: ', mode(lengths))
print('median: ', median(lengths))
