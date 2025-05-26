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
from transformers import T5EncoderModel, AutoTokenizer
from torchvision import transforms
import random
from torchvision.transforms.functional import hflip


# Enable TF32 for improved performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Unetfeats(nn.Module):
    def __init__(self, unet_path, text_encoder_path, embed_dim=3584, latent_dim=25):
        super(Unetfeats, self).__init__()
        config = UNet2DConditionModel.load_config(unet_path, subfolder="unet")
        config["in_channels"] = 3
        config["out_channels"] = 1
        config["sample_size"] = 256
        config["block_out_channels"] = [
            32, 64, 128, 128
        ]
        config["cross_attention_dim"] = 512
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.cross_attention_dim = config["cross_attention_dim"]
        self.model = UNet2DConditionModel.from_config(config)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, subfolder="tokenizer_3")
        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_path, subfolder="text_encoder_3")
        self.down_proj = Mlp(in_features=4096, hidden_features=self.cross_attention_dim, out_features=self.cross_attention_dim)
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
    def forward(self, images, tokens):
        # Pass through the model
        h = self.text_encoder(input_ids=tokens).last_hidden_state
        h = self.down_proj(h)
        out = self.model(
            sample=images,
            timestep=0,
            encoder_hidden_states=h,
        ).sample
        
        return F.sigmoid(out)



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
            if transform:
                is_hflip = random.random() < 0.5
                image = transform(image).float()
                label = transform(label).float()
                if is_hflip:
                    image = hflip(image)
                    label = hflip(label)
            
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


# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--unet_path", type=str, default="/data1/felixchao/diffusion")
parser.add_argument("--text_encoder_path", type=str, default="/data1/felixchao/sd3_5")
parser.add_argument("--dataset", type=str, default="/home/felixchaotw/mllm-physical-design/armo/dataset/train_feature_desc.csv")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--latent_dim", type=int, default=25)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--embed_dim", type=int, default=3584)
args = parser.parse_args()

device = f"cuda:{args.device}" if args.device >= 0 else "cpu"


# Load embeddings
print("Loading dataset...")
train_df = pd.read_csv(args.dataset)
test_df_a = pd.read_csv("/home/felixchaotw/mllm-physical-design/armo/dataset/test_feature_desc_a.csv")
test_df_b = pd.read_csv("/home/felixchaotw/mllm-physical-design/armo/dataset/test_feature_desc_b.csv")

all_df = pd.concat([train_df, test_df_a, test_df_b], ignore_index=True)

new_train_df = pd.read_csv('/home/felixchaotw/mllm-physical-design/armo/dataset/train.txt', sep=",", header=None)
new_test_df = pd.read_csv('/home/felixchaotw/mllm-physical-design/armo/dataset/test.txt', sep=",", header=None)
new_train_df.columns = ['id', 'prompt']
new_test_df.columns = ['id', 'prompt']
new_train_df.drop(columns=['prompt'], inplace=True)
new_test_df.drop(columns=['prompt'], inplace=True)
train_df = pd.merge(new_train_df, all_df, on='id', how='inner').reset_index(drop=True)
test_df = pd.merge(new_test_df, all_df, on='id', how='inner').reset_index(drop=True)

print("Train data size:", len(train_df))
print("Test data size:", len(test_df))

train_df.to_csv('/home/felixchaotw/mllm-physical-design/armo/dataset/new_train_feature_desc.csv', index=False)
test_df.to_csv('/home/felixchaotw/mllm-physical-design/armo/dataset/new_test_feature_desc.csv', index=False)