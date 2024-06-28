import os.path
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from torchvision.datasets import ImageFolder


def mosaic(clean_path: str, corruption_path: str, save_path: str, name: str, cls: int = None, sample_size: int = 6):
    clean_dataset = ImageFolder(root=clean_path, transform=transforms.ToTensor())
    corrupted_datasets = [ImageFolder(root=os.path.join(corruption_path, str(i)), transform=transforms.ToTensor())
                          for i in range(1, 6)]
    if cls is not None:
        indices = [i for i, x in enumerate(clean_dataset.targets) if x == cls]
        clean_dataset = Subset(clean_dataset, indices)
        corrupted_datasets = [Subset(ds, indices) for ds in corrupted_datasets]
    clean_loader = DataLoader(clean_dataset, batch_size=sample_size, shuffle=False, pin_memory=True)
    corrupted_loaders = [DataLoader(ds, batch_size=sample_size, shuffle=False, pin_memory=True,)
                         for ds in corrupted_datasets]
    clean_batch, _ = iter(clean_loader).next()
    corrupted_batches = [iter(dl).next()[0] for dl in corrupted_loaders]
    images = []
    for i in range(sample_size):
        for j in range(6):
            if j == 0:
                images.append(clean_batch[i])
            else:
                images.append(corrupted_batches[j - 1][i])
    save_image(make_grid(images, nrow=6), os.path.join(save_path, name + '.png'))


# mosaic('../data/tiny/val', '../data/tiny-c/val/brightness', '.', 'Mosaic', 199)
