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
import timm
from timm.models.vision_transformer import Mlp
from timm.layers import trunc_normal_
from timm.models.layers import DropPath, LayerNorm2d
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from pytorch_msssim import SSIM, MS_SSIM
from diffusers import UNet2DConditionModel
from transformers import T5EncoderModel, AutoTokenizer
from torchvision import transforms
from torchvision.transforms.functional import hflip
from models.adapters import Adapter
from dataclasses import dataclass, field
import random


# Enable TF32 for improved performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class Config:
    # Unet configs
    unet_path: str = "/data1/felixchao/diffusion"
    text_encoder_path: str = "/data1/felixchao/sd3_5"
    latent_dim: int = 25
    embed_dim: int = 3584
    in_channels: int = 3
    out_channels: int = 1
    sample_size: int = 256
    block_out_channels: list = field(default_factory=lambda: [32, 64, 128, 128])
    cross_attention_dim: int = 512
    text_encoder_dim: int = 4096
    
    # Feature extraction configs
    cnn_backbone: str = "resnet50"
    pretrained: bool = True
    features_only: bool = True
    out_indices: list = field(default_factory=lambda: [1, 2, 3, 4])
    cnn_feature_num: int = 4
    feature_channels: list = field(default_factory=lambda: [256, 512, 1024, 2048])
    cnn_dim: int = 256
    grid_size: int = 16
    

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    

class Unetfeats(nn.Module):
    def __init__(self, config):
        super(Unetfeats, self).__init__()
        unet_config = UNet2DConditionModel.load_config(config.unet_path, subfolder="unet")
        unet_config["in_channels"] = config.in_channels
        unet_config["out_channels"] = config.out_channels
        unet_config["sample_size"] = config.sample_size
        unet_config["block_out_channels"] = config.block_out_channels
        unet_config["cross_attention_dim"] = config.cross_attention_dim
        self.embed_dim = config.embed_dim
        self.latent_dim = config.latent_dim
        self.cnn_dim = config.cnn_dim
        self.grid_size = config.grid_size
        self.cross_attention_dim = unet_config["cross_attention_dim"]
        
        # Load the UNet model
        self.model = UNet2DConditionModel.from_config(unet_config)
        
        # Load the text encoder and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder_path, subfolder="tokenizer_3")
        self.text_encoder = T5EncoderModel.from_pretrained(config.text_encoder_path, subfolder="text_encoder_3")
        
        # Down-projection layer
        self.down_proj = Mlp(in_features=config.text_encoder_dim, hidden_features=self.cross_attention_dim, out_features=self.cross_attention_dim)
        
        self.cnn_extractors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(config.feature_channels[i], config.cnn_dim, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.GELU(),
                    nn.Conv2d(config.cnn_dim, config.cnn_dim, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.AdaptiveAvgPool2d(config.grid_size),
                )
                for i in range(config.cnn_feature_num)
            ]
        )
        
        self.mha = Attention(
            dim=config.cnn_dim,
            num_heads=8,
            qkv_bias=True,
            qk_norm=None,
            attn_drop=0.,
            proj_drop=0.,
        )
        
        self.layer_norm = nn.LayerNorm(config.cnn_dim)
        self.gamma = nn.Parameter(torch.ones(config.cnn_dim))
        self.drop_path = DropPath(0.2)
        
        # Highway connections
        self.vision_highway = Adapter()
        
        # feature_extraction model
        self.cnn_feature_model = timm.create_model(
            config.cnn_backbone,
            pretrained=config.pretrained,
            features_only=config.features_only,
            out_indices=config.out_indices,
        )

            
    def freeze_parameters(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        for param in self.cnn_feature_model.parameters():
            param.requires_grad = False
            
    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True
            
    
    def forward_high_way(self, images):
        bsz = images.shape[0]
        feature_list = self.cnn_feature_model(images)
        
        cnn_tokens = []
        for i in range(len(feature_list)):
            cnn_feature = self.cnn_extractors[i](feature_list[i])
            cnn_feature = cnn_feature.permute(0, 2, 3, 1).reshape(bsz, -1, self.cnn_dim)
            cnn_tokens.append(cnn_feature)
            
        cnn_tokens = torch.cat(cnn_tokens, dim=1)
        
        cnn_tokens = cnn_tokens + self.drop_path(self.gamma * self.mha(self.layer_norm(cnn_tokens)))
        
        cnn_tokens = cnn_tokens.reshape(bsz, self.grid_size * 2, self.grid_size * 2, self.cnn_dim).permute(0, 3, 1, 2)
        
        highway_embeds = self.vision_highway(cnn_tokens)
        
        return highway_embeds
        
        
    def forward(self, images, tokens):
        # Pass through the model
        h = self.text_encoder(input_ids=tokens).last_hidden_state
        h = self.down_proj(h)
        # Highway connections
        down_block_additional_residuals = self.forward_high_way(images)
        # Pass through the UNet model
        out = self.model(
            sample=images,
            timestep=0,
            encoder_hidden_states=h,
            down_intrablock_additional_residuals=[
                sample for sample in down_block_additional_residuals
            ]
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
        
        
# Loss function
def ias(instance, target, lambda_=0.007):
    n_id = instance.shape[0]
    d_ae = abs(instance - target) / target.max()
    e_t = (d_ae > lambda_).nonzero()
    if len(e_t) == 0:
        e_t = torch.range(0, n_id-1)
    return e_t.tolist()
    

def amse_loss(output, target):
    bsz = output.shape[0]
    losses = []
    for i in range(bsz):
        output_t = output[i].flatten()
        target_t = target[i].flatten()
        e_t = ias(output_t, target_t)
        losses.append(F.mse_loss(output_t[e_t], target_t[e_t]))
        
    return torch.mean(torch.stack(losses))


# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--unet_path", type=str, default="/data1/felixchao/diffusion")
parser.add_argument("--text_encoder_path", type=str, default="/data1/felixchao/sd3_5")
parser.add_argument("--dataset", type=str, default="/home/felixchaotw/mllm-physical-design/armo/dataset/train_feature_desc.csv")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--latent_dim", type=int, default=25)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--embed_dim", type=int, default=3584)
args = parser.parse_args()

device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

configs = Config(
    unet_path=args.unet_path,
    text_encoder_path=args.text_encoder_path,
    latent_dim=args.latent_dim,
    embed_dim=args.embed_dim,
)

# Load embeddings
print("Loading dataset...")
train_df = pd.read_csv(args.dataset)

# Load Dataset
print("Dataset preparing...")

train_dataset = CongestionDataset(
    train_df,
    transforms.Compose([
            transforms.ToTensor(),
    ])
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=0,
)

print("Training data size:", len(train_dataset))

# Initialize model
model = Unetfeats(config=configs).to(device)
model.freeze_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
beta=0.84
loss_fn = nn.L1Loss()
mssim_fn = MS_SSIM(data_range=1, size_average=True, channel=1)

num_epochs = args.epochs
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="cosine", optimizer=optimizer, num_warmup_steps=20, num_training_steps=num_training_steps
)

# Train model
print("Start Training...")

for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    model.train()
    epoch_loss = 0
    
    for batch in train_loader:
        prompts = batch["prompts"]
        images = batch["images"].to(device)
        images = images * 2.0 - 1.0
        labels = batch["labels"].to(device)
        
        input_ids = model.tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=512, truncation=True).input_ids.to(device)
        
        optimizer.zero_grad()
        output = model(images, input_ids)
        loss = (1-beta) * loss_fn(output.squeeze(), labels.squeeze()) + beta * (1 - mssim_fn(output, labels))
        # loss = amse_loss(output, labels)
        print(f"Epoch {epoch + 1}/{num_epochs}, Batch Loss: {loss.item()}")
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")  
     
    model.eval()
    torch.save(model.state_dict(), f"/data1/felixchao/output/highway_checkpoint_mpgd-{epoch}.pth")
    print("Model saved")
