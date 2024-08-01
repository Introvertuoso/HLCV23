import torch 
from torch.utils.data import DataLoader, Dataset

class CachedTinyImageNet(Dataset):
    def __init__(self, cache_path):
        self.data = torch.load(cache_path)
        self.embeddings = self.data['embeddings']
        self.labels = self.data['labels']
        self.classes = (torch.unique(self.labels)[0])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class RandomDataset(Dataset):
    
    def __init__(self, embed_size, length=100000):
        self.len = length
        # self.data = torch.randn(length, self.size)
        self.embed_size = embed_size
        self.classes = 100
        
    def __getitem__(self, index):
        return torch.randn(self.embed_size), torch.randint(0, 10, (1,)).item()
    
    def __len__(self):
        return self.len
    