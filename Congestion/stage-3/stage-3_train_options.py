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
from pytorch_msssim import SSIM
from diffusers import UNet2DConditionModel
from transformers import T5EncoderModel, AutoTokenizer
from torchvision import transforms
from models.adapters import VLM_Adapter
from dataclasses import dataclass, field


import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch import nn


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
        self.down_proj_a = Mlp(in_features=config.text_encoder_dim, hidden_features=self.cross_attention_dim, out_features=self.cross_attention_dim)
        self.down_proj_b = Mlp(in_features=config.text_encoder_dim, hidden_features=self.cross_attention_dim, out_features=self.cross_attention_dim)
        # Highway connections
        self.vision_highway = VLM_Adapter()

            
    def freeze_parameters(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False

            
    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True
            
    
    def forward_high_way(self, vlm_tokens):

        highway_embeds = self.vision_highway(vlm_tokens)

        return highway_embeds
    
    
    def foward_configs(self, config_tokens):
        h = self.text_encoder(input_ids=config_tokens).last_hidden_state
        h = self.down_proj_a(h)
        return h
        
        
    def forward_description(self, desc_tokens):
        h = self.text_encoder(input_ids=desc_tokens).last_hidden_state
        h = self.down_proj_b(h)
        return h
        
        
    def forward(self, images, desc_tokens, vlm_tokens=None, config_tokens=None):
        # Pass through the text encoder
        if config_tokens is not None:
            h_config = self.foward_configs(config_tokens)
            h_desc = self.forward_description(desc_tokens)
            h = torch.cat([h_config, h_desc], dim=1)
        else:
            h = self.forward_description(desc_tokens)
        
        # Pass through the image encoder-decoder
        if vlm_tokens is not None:
            # Highway connections
            down_block_additional_residuals = self.forward_high_way(vlm_tokens)
            # Pass through the UNet model
            out = self.model(
                sample=images,
                timestep=0,
                encoder_hidden_states=h,
                down_intrablock_additional_residuals=[
                    sample for sample in down_block_additional_residuals
                ]
            ).sample
        else:
            out = self.model(
                sample=images,
                timestep=0,
                encoder_hidden_states=h
            ).sample
        
        return F.sigmoid(out)
    
    
def data_collator(batch):
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    prompts = [item['prompt'] for item in batch]
    configs = [item['config'] for item in batch]
    vlm_tokens = [item['vlm_tokens'] for item in batch]
    
    # Stack images and labels
    images = torch.stack(images)
    labels = torch.stack(labels)
    vlm_tokens = torch.stack(vlm_tokens) if vlm_tokens[0] is not None else None
    if configs[0] is None:
        configs = None
    
    
    return {
        'images': images,
        'labels': labels,
        'prompts': prompts,
        'configs': configs,
        'vlm_tokens': vlm_tokens,
    }
    

class CongestionDataset(Dataset):
    def __init__(self, df, vlm_tokens, config=False, transform=None):
        self.df = df
        self.transform = transform
        self.prompts = df['prompt'].tolist()
        self.configs = df['config'].tolist() if config else None
        self.images = []
        self.labels = []
        self.vlm_tokens = vlm_tokens
        self.config = config
        for i, example in tqdm(df.iterrows()):
            image_id = example["id"]
            image = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/feature/{image_id}")
            label = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/label/{image_id}").squeeze()
            image = Image.fromarray(np.uint8(image * 255)).convert('RGB')
            if transform:
                image = transform(image).float()
                label = transform(label).float()
            
            self.images.append(image)
            self.labels.append(label)
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'label': self.labels[idx],
            'prompt': self.prompts[idx],
            'config': self.configs[idx] if self.config is not None else None,
            'vlm_tokens': self.vlm_tokens[idx] if self.vlm_tokens is not None else None,
        }
    

def train(model, dataloader, sampler, optimizer, loss_fn, lr_scheduler, device, rank, args):
    ssim_fn = SSIM(data_range=1, size_average=True, channel=1)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0

        for idx, batch in enumerate(dataloader):
            images = batch["image"]
            labels = batch["label"]
            prompts = batch["prompt"]
            configs = batch["config"]
            vlm_tokens = batch["vlm_tokens"]
            
            images = images.to(device)
            labels = labels.to(device)
            prompt_input_ids = model.module.tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=512, truncation=True).input_ids.to(device)
            if vlm_tokens is not None:
                vlm_tokens = vlm_tokens.to(device)
            if configs is not None:
                config_input_ids = model.module.tokenizer(configs, return_tensors="pt", padding="max_length", max_length=77, truncation=True).input_ids.to(device)
            else:
                config_input_ids = None

            optimizer.zero_grad()
            output = model(images, prompt_input_ids, vlm_tokens=vlm_tokens, config_tokens=config_input_ids)
            loss = loss_fn(output.squeeze(), labels.squeeze())
            if rank == 0:
                print(f"Epoch {epoch + 1}/{args.epochs}, Batch {idx}/{len(dataloader)}, Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()

        if rank == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {epoch_loss / len(dataloader)}")
            save_path = f"/data1/felixchao/output/new_high_way-{epoch}.pth"
            torch.save({
                "model": model.module.state_dict(),  # use model.module when saving from DDP
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch
            }, save_path)
            print(f"[Rank {rank}] Epoch {epoch}: Model saved at {save_path}")
            
            
def loss_selection_fn(seg_loss):
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    beta = 0.85
    if seg_loss:
        
        def custom_loss(pred, target):
            loss = mse_loss(pred, target) + beta * bce_loss(pred, target)
            return loss
            
        return custom_loss
    else:
        return mse_loss  
    

def main(args):
    world_size = int(os.environ.get("WORLD_SIZE", 8))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    # model configs
    model_config = Config(
        unet_path=args.unet_path,
        text_encoder_path=args.text_encoder_path,
        latent_dim=args.latent_dim,
        embed_dim=args.embed_dim,
    )
    print(f"Start running on rank {rank}.")

    device_id = rank % torch.cuda.device_count()

    # Load embeddings (only print on rank 0)
    if rank == 0:
        print(f"Group initialized? {dist.is_initialized()}", flush=True)
        print("Loading embeddings...")

    if args.high_way:
        samples = load_file(args.vlm_output)
        vlm_tokens = samples.pop("vlm_tokens")
    else:
        vlm_tokens = None
    
    # Load training description dataset
    train_df = pd.read_csv(args.dataset)
    
    # Dataset
    dataset = CongestionDataset(df=train_df, vlm_tokens=vlm_tokens, config=args.config, transform=transforms.Compose([transforms.ToTensor()]))
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    # Model
    model = Unetfeats(
        config=model_config,
    ).to(device_id)
    
    model.freeze_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0)
    
    # Loss Selection
    loss_fn = loss_selection_fn(args.seg_loss)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=20,
        num_training_steps=args.epochs * len(dataloader)
    )

    if args.resume:
        # 🔁 Resume from latest checkpoint if exists
        ckpt_dir = "/lustre/fsw/portfolios/nvr/users/yundat/mllm-physical-design/armo/decoder"
        if rank == 0:
            print("Checking for existing checkpoints...")
        checkpoint_paths = sorted(glob.glob(os.path.join(ckpt_dir, "epoch-*.pth")))
        if checkpoint_paths:
            latest_ckpt = checkpoint_paths[-1]
            map_location = {"cuda:0": f"cuda:{device_id}"}
            checkpoint = torch.load(latest_ckpt, map_location=map_location)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if rank == 0:
                print(f"Resumed from checkpoint: {latest_ckpt}")
        else:
            if rank == 0:
                print("No checkpoint found. Starting from scratch.")

    model = DDP(model, device_ids=[device_id])

    train(model, dataloader, sampler, optimizer, loss_fn, lr_scheduler, device_id, rank, args)

    dist.destroy_process_group()
    
    
    
def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/home/felixchaotw/mllm-physical-design/armo/dataset/train_feature_desc.csv")
    parser.add_argument("--vlm_output", type=str, default="/data1/felixchao/vlm_tokens.safetensors")
    parser.add_argument("--unet_path", type=str, default="/data1/felixchao/diffusion")
    parser.add_argument("--text_encoder_path", type=str, default="/data1/felixchao/sd3_5")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent_dim", type=int, default=25)
    parser.add_argument("--embed_dim", type=int, default=3584)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", action="store_true")
    parser.add_argument("--high_way", action="store_true")
    parser.add_argument("--seg_loss", action="store_true")
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    run()
