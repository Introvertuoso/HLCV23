import torch 
from torch.utils.data import DataLoader, Dataset

class CachedTinyImageNet(Dataset):
    def __init__(self, cache_path):
        self.data = torch.load(cache_path)
        self.embeddings = self.data['embeddings']
        self.labels = self.data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

