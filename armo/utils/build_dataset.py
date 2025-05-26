from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import torch

def data_collator(batch):
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    
    # Stack images and labels
    images = torch.stack(images)
    labels = torch.stack(labels)
    
    return {
        'images': images,
        'labels': labels,
        'prompts': prompts
    }
    

class CongestionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.prompts = df['prompt'].tolist()
        self.images = []
        self.labels = []
        for i, example in tqdm(df.iterrows()):
            image_id = example["id"]
            image = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/feature/{image_id}")
            label = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/label/{image_id}").squeeze()
            image = Image.fromarray(np.uint8(image * 255)).convert('RGB')
            label = Image.fromarray(np.uint8(label * 255)).convert('RGB')
            if transform:
                image = transform(image)
                label = transform(label)
            
            self.images.append(image)
            self.labels.append(label)
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'label': self.labels[idx],
            'prompt': self.prompts[idx]
        }