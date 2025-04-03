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
from pytorch_msssim import SSIM
from diffusers import UNet2DConditionModel

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP



# Enable TF32 for improved performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Unetfeats(nn.Module):
    def __init__(self, unet_path, embed_dim=3584, latent_dim=25):
        super(Unetfeats, self).__init__()
        config = UNet2DConditionModel.load_config(unet_path, subfolder="unet")
        config["in_channels"] = 3
        config["out_channels"] = 1
        config["sample_size"] = 256
        config["block_out_channels"] = [
            32, 64, 128, 128
        ]
        config["cross_attention_dim"] = 128
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.cross_attention_dim = config["cross_attention_dim"]
        self.seq_len = 77
        self.model = UNet2DConditionModel.from_config(config)
        self.text_proj = Mlp(in_features=embed_dim + latent_dim * 2, hidden_features=self.cross_attention_dim * 77, out_features=self.cross_attention_dim * 77)
        
        
    def forward(self, images, multi_rewards, gating_weights, image_tokens):
        # Concatenate the inputs
        x = torch.cat([multi_rewards, gating_weights, image_tokens], dim=1)
        x = self.text_proj(x)
        x = x.view(-1, self.seq_len, self.cross_attention_dim)
        
        # Pass through the model
        out = self.model(
            sample=images,
            timestep=0,
            encoder_hidden_states=x,
        ).sample
        
        return F.sigmoid(out)

class CongestionDataset(Dataset):
    def __init__(self, images, image_tokens, gating_weights, multi_rewards, labels):
        self.images = images
        self.image_tokens = image_tokens
        self.gating_weights = gating_weights
        self.multi_rewards = multi_rewards
        self.labels = labels
        
    def __len__(self):
        return len(self.image_tokens)
    
    def __getitem__(self, idx):
        return (self.images[idx] * 2.0 - 1.0), self.image_tokens[idx], self.gating_weights[idx], self.multi_rewards[idx], self.labels[idx]



def validation(model, test_loader, device, ssim_fn):
    model.eval()
    val_ssim_score = 0.0
    with torch.no_grad():
        for image_tokens, gating_weights, multi_rewards, labels in test_loader:
            image_tokens, gating_weights, multi_rewards, labels = image_tokens.to(device), gating_weights.to(device), multi_rewards.to(device), labels.to(device)
            output = model(multi_rewards, gating_weights, image_tokens)
            val_ssim_score += ssim_fn(output, labels.unsqueeze(1))
            
    return val_ssim_score / len(test_loader)


def train(model, dataloader, optimizer, loss_fn, lr_scheduler, device, args):
    ssim_fn = SSIM(data_range=1, size_average=True, channel=1)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for batch in dataloader:
            images, image_tokens, gating_weights, multi_rewards, labels = batch
            images = images.to(device)
            image_tokens = image_tokens.to(device)
            gating_weights = gating_weights.to(device)
            multi_rewards = multi_rewards.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images, multi_rewards, gating_weights, image_tokens)
            loss = loss_fn(output.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")
        if epoch % 10 == 0:
            save_path = f"/lustre/fsw/portfolios/nvr/users/yundat/mllm-physical-design/armo/decoder/epoch-{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Epoch {epoch}: Model saved at {save_path}")


def main(args):

    device = torch.device("cuda:0")
    print("Loading embeddings...")

    samples = load_file(args.dataset)
    images = samples.pop("images")
    last_hidden_tokens = samples.pop("last_hidden_tokens")
    gating_weights = samples.pop("gating_weights")
    multi_rewards = samples.pop("multi_rewards")
    labels = samples.pop("label")

    # Dataset
    print("Dataset preparing...")
    train_dataset = CongestionDataset(images, last_hidden_tokens, gating_weights, multi_rewards, labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    print("Training data size:", len(train_dataset))

    # Model
    model = Unetfeats(
        unet_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)
    loss_fn = nn.MSELoss()
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=20,
        num_training_steps=args.epochs * len(train_loader)
    )

    train(model, train_loader, optimizer, loss_fn, lr_scheduler, device, args)



def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/lustre/fsw/portfolios/nvr/users/yundat/mllm-physical-design/armo/dataset/last_hidden_feats.safetensors")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent_dim", type=int, default=25)
    parser.add_argument("--embed_dim", type=int, default=3584)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    run()


