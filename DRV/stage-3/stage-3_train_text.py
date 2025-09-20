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
from sklearn.metrics import accuracy_score, precision_score, f1_score


# Enable TF32 for improved performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Unetfeats(nn.Module):
    def __init__(self, unet_path, text_encoder_path, embed_dim=3584, latent_dim=25):
        super(Unetfeats, self).__init__()
        config = UNet2DConditionModel.load_config(unet_path, subfolder="unet")
        config["in_channels"] = 9
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
    def __init__(self, df, transform=None, split="train"):
        self.df = df
        self.transform = transform
        self.prompts = df['prompt'].tolist()
        self.images = []
        self.labels = []
        for i, example in tqdm(df.iterrows()):
            image_id = example["id"]
            image = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/DRC/feature/{image_id}")
            label = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/DRC/label/{image_id}").squeeze()
            if transform:
                image = transform(image).float()
                if split == "train":
                    label = torch.where(transform(label).float() >= 0.1, 1.0, 0.0) 
                else:
                    label = transform(label).float()
            
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
        

def validate(model, valid_loader, device):
    model.eval()
    ssim_fn = SSIM(data_range=1, size_average=True, channel=1)
    total_ssim = 0.0
    acc = 0.0
    precision = 0.0
    f1 = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            prompts = batch["prompts"]
            images = batch["images"].to(device)
            images = images * 2.0 - 1.0
            labels = batch["labels"].to(device)
            input_ids = model.tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=512, truncation=True).input_ids.to(device)
            
            output = model(images, input_ids)
            ssim_score = ssim_fn(output, labels).item()
            total_ssim += ssim_score
            print("SSIM score:", ssim_score)
            img1 = output.squeeze().cpu().numpy()
            img2 = labels.squeeze().cpu().numpy()
            img2_flatten = img2.flatten() >= 0.1
            img1_flatten = img1.flatten() >= 0.1
            acc_score = accuracy_score(img2_flatten, img1_flatten)
            acc += acc_score
            prec_score = precision_score(img2_flatten, img1_flatten)
            precision += prec_score
            f1_sc = f1_score(img2_flatten, img1_flatten)
            f1 += f1_sc
            print(f"Accuracy: {acc_score}")
            print(f"Precision: {prec_score}")
            print(f"F1 Score: {f1_sc}")
            
    return total_ssim / len(valid_loader)


# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--unet_path", type=str, default="/data1/felixchao/diffusion")
parser.add_argument("--text_encoder_path", type=str, default="/data1/felixchao/sd3_5")
parser.add_argument("--dataset", type=str, default="/home/felixchaotw/mllm-physical-design/DRV/dataset/train_feature_desc.csv")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--latent_dim", type=int, default=25)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--embed_dim", type=int, default=3584)
args = parser.parse_args()

device = f"cuda:{args.device}" if args.device >= 0 else "cpu"


# Load embeddings
print("Loading dataset...")
train_df = pd.read_csv(args.dataset)
train_df, valid_df = train_test_split(train_df, test_size=0.05, random_state=0, shuffle=True)

# Load Dataset
print("Dataset preparing...")

train_dataset = CongestionDataset(
    train_df,
    transforms.Compose([
        transforms.ToTensor(),
    ])
)

valid_dataset = CongestionDataset(
    valid_df,
    transforms.Compose([
        transforms.ToTensor(),
    ]),
    split="test",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=0,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=0,
)

print("Training data size:", len(train_dataset))

# Initialize model

model = Unetfeats(unet_path=args.unet_path, text_encoder_path=args.text_encoder_path, embed_dim=args.embed_dim, latent_dim=args.latent_dim).to(device)
model.load_state_dict(torch.load("/data1/felixchao/DRC/checkpoint-best.pth", weights_only=True, map_location=device))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)
# loss_fn = nn.MSELoss()
loss_fn = nn.BCELoss()
num_epochs = args.epochs
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="cosine", optimizer=optimizer, num_warmup_steps=20, num_training_steps=num_training_steps
)

# Train model
print("Start Training...")

best_ssim = 0.80
global_step = 0

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
        
        if global_step % 500 == 0:
            ssim_score = validate(model, valid_loader, device)
            print(f"Validation SSIM at step {global_step}: {ssim_score}")
            if ssim_score > best_ssim:
                best_ssim = ssim_score
                torch.save(model.state_dict(), f"/data1/felixchao/DRC/checkpoint-best.pth")
                print("Best model saved with SSIM:", best_ssim)
                
        global_step += 1
     
    model.eval()
    torch.save(model.state_dict(), f"/data1/felixchao/DRC/checkpoint-final-{epoch}.pth")
    print("Model saved")
