import os
import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import load_file
from argparse import ArgumentParser
from tqdm.auto import tqdm
from scipy.stats import spearmanr
import pandas as pd
from glob import glob
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import datasets
from PIL import Image
from scipy import stats
from timm.models.vision_transformer import Mlp
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from utils.metrics import build_metric
from pytorch_msssim import SSIM


# Enable TF32 for improved performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class CongestionDataset(Dataset):
    def __init__(self, last_hidden_tokens, labels):
        self.last_hidden_tokens = last_hidden_tokens
        self.labels = labels
        
    def __len__(self):
        return len(self.last_hidden_tokens)
    
    def __getitem__(self, idx):
        return self.last_hidden_tokens[idx], self.labels[idx]
    
    
class VisionDecoder(nn.Module):
    def __init__(self, hidden_dims, embed_dim=3584):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.decoder_input = nn.Linear(embed_dim, hidden_dims[0] * 4)
        
        modules = []
        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
            
        self.decoder = nn.ModuleList(modules)
        

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=1,
                                      kernel_size=3, padding=1),
                            nn.Sigmoid()
                            )
        
    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, self.hidden_dims[0], 2, 2)
        for i in range(len(self.hidden_dims)-1):
            x = self.decoder[i](x)
            
        x = self.final_layer(x)
        return x
    


        

def validation(model, test_loader, device, ssim_fn):
    model.eval()
    val_ssim_score = 0.0
    with torch.no_grad():
        for image_tokens, labels in test_loader:
            image_tokens, labels = image_tokens.to(device), labels.to(device)
            output = model(image_tokens)
            val_ssim_score += ssim_fn(output, labels.unsqueeze(1))
            
    return val_ssim_score / len(test_loader)


# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--dataset", type=str, default="/data1/felixchao/last_tokens.safetensors")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100000)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--latent_dim", type=int, default=25)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--embed_dim", type=int, default=3584)
args = parser.parse_args()

device = f"cuda:{args.device}" if args.device >= 0 else "cpu"


# Load embeddings
print("Loading embeddings...")
samples = load_file(args.dataset)

last_hidden_tokens = samples["last_hidden_tokens"]
labels = samples["label"]

# Load Dataset
print("Dataset preparing...")
train_tokens, test_tokens, train_labels, test_labels = train_test_split(last_hidden_tokens, labels, test_size=0.2, random_state=42)

train_dataset = CongestionDataset(train_tokens, train_labels)
test_dataset = CongestionDataset(test_tokens, test_labels)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Training data size:", len(train_dataset))
print("Testing data size:", len(test_dataset))

# Initialize model
model = VisionDecoder([512, 256, 128, 64, 32, 16, 8], embed_dim=args.embed_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)
loss_fn = nn.MSELoss()
num_epochs = args.epochs
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="cosine", optimizer=optimizer, num_warmup_steps=20, num_training_steps=num_training_steps
)

# Train model
print("Start Training...")

best_ssim = 0.0
ssim_fn = SSIM(data_range=1, size_average=True, channel=1)

for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    model.train()
    epoch_loss = 0
    
    for batch in train_loader:
        image_tokens, labels = batch
        image_tokens, labels = image_tokens.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(image_tokens)
        loss = loss_fn(output.squeeze(), labels.squeeze())
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        
    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {epoch_loss / len(train_loader)}")
        ssim_score = validation(model, test_loader, device, ssim_fn)
        print(f"SSIM Score: {ssim_score}")
        
        if ssim_score > best_ssim:
            best_ssim = ssim_score
            torch.save(model.state_dict(), "/home/felixchaotw/mllm-physical-design/armo/decoder/vision_model.pth")
            print("Model saved")
