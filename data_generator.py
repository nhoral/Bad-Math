import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DivisionDataset(Dataset):
    def __init__(self, num_samples, min_val=1, max_val=100):
        # Generate random numerators and denominators
        self.numerators = torch.randint(min_val, max_val, (num_samples,))
        self.denominators = torch.randint(min_val, max_val, (num_samples,))
        self.results = self.numerators.float() / self.denominators.float()
        
    def __len__(self):
        return len(self.numerators)
    
    def __getitem__(self, idx):
        X = torch.tensor([self.numerators[idx], self.denominators[idx]], dtype=torch.float32)
        y = torch.tensor([self.results[idx]], dtype=torch.float32)
        return X, y

def get_data_loaders(train_samples=10000, val_samples=2000, batch_size=32):
    train_dataset = DivisionDataset(train_samples)
    val_dataset = DivisionDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader