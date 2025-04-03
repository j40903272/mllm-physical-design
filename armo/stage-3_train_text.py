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
            32, 64, 128, 256
        ]
        config["cross_attention_dim"] = 1536
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.cross_attention_dim = config["cross_attention_dim"]
        self.model = UNet2DConditionModel.from_config(config)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, subfolder="tokenizer_3")
        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_path, subfolder="text_encoder_3")
        self.down_proj = Mlp(in_features=4096, hidden_features=self.cross_attention_dim, out_features=self.cross_attention_dim)
        
        
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



def validation(model, test_loader, device, ssim_fn):
    model.eval()
    val_ssim_score = 0.0
    with torch.no_grad():
        for image_tokens, gating_weights, multi_rewards, labels in test_loader:
            image_tokens, gating_weights, multi_rewards, labels = image_tokens.to(device), gating_weights.to(device), multi_rewards.to(device), labels.to(device)
            output = model(multi_rewards, gating_weights, image_tokens)
            val_ssim_score += ssim_fn(output, labels.unsqueeze(1))
            
    return val_ssim_score / len(test_loader)


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


# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--unet_path", type=str, default="/data1/felixchao/diffusion")
parser.add_argument("--text_encoder_path", type=str, default="/data1/felixchao/sd3_5")
parser.add_argument("--dataset", type=str, default="/home/felixchaotw/mllm-physical-design/armo/dataset/train_feature_desc.csv")
parser.add_argument("--batch_size", type=int, default=8)
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

model = Unetfeats(unet_path=args.unet_path, text_encoder_path=args.text_encoder_path, embed_dim=args.embed_dim, latent_dim=args.latent_dim).to(device)

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
        prompts = batch["prompts"]
        images = batch["images"].to(device)
        images = images * 2.0 - 1.0
        labels = batch["labels"].to(device)
        input_ids = model.tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=512, truncation=True).input_ids.to(device)
        
        optimizer.zero_grad()
        output = model(images, input_ids)
        loss = loss_fn(output.squeeze(), labels.squeeze())
        print(f"Epoch {epoch + 1}/{num_epochs}, Batch Loss: {loss.item()}")
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")  
     
    model.eval()
    torch.save(model.state_dict(), f"/data1/felixchao/output/checkpoint-{epoch}.pth")
    print("Model saved")
