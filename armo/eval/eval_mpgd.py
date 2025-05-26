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
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoProcessor
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
import models
from diffusers import UNet2DConditionModel
from transformers import T5EncoderModel, AutoTokenizer
from dataclasses import dataclass, field
from models.adapters import Adapter, VLM_Adapter
from skimage.metrics import normalized_root_mse


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
    

# Define the attributes for multi-objective reward modeling
attributes = [
    'rudy_gradient_variability',
    'clustered_macro_distance_std',
    'rudy_pin_clustering_coefficient',
    'macro_density_gradient',
    'macro_aspect_ratio_variance',
    'macro_compactness_index',
    'rudy_pin_compaction_ratio',
    'macro_variability_coefficient',
    'macro_symmetry_coefficient',
    'macro_cluster_density_contrast',
    'rudy_pin_distribution_kurtosis',
    'localized_rudy_variability_coefficient',
    'macro_distribution_clarity_index',
    'rudy_direction_consistency_index',
    'rudy_pin_area_masking_index',
    'rudy_pin_gradient_convergence',
    'rudy_intensity_symmetry_index',
    'rudy_deviation_effect_index',
    'demarcated_macro_proximity_index',
    'macro_surface_irregularity_index',
    'macro_rudy_boundary_interaction_index',
    'pin_density_peak_contrast',
    'rudy_pin_density_flux_index',
    'high_density_rudy_ratio',
    'high_density_rudy_pin_ratio'
]


default = {
        "freq" : {
                "c2": "500MHz",
                "c5": "200MHz",
                "c20": "50MHz",
        },
        "macros_a": {
                "1": 3, 
                "2": 4,
                "3": 5,
        },
        "macros_b": {
                "1": 13, 
                "2": 14,
                "3": 15,
        },
        "util":{
            "u0.7": "70%",
            "u0.75": "75%",
            "u0.8": "80%",
            "u0.85": "85%",
            "u0.9": "90%",
        },
        "macro_placement": 3,
        "p_mesh": 8,
        "filler": {
            "f0": "After routing",
            "f1": "After placement",
        }
}

def find_token_for_image(lst, imstart_id, imend_id):
    """Find the first occurrence of a token_pattern in a list."""
    imstarts = []
    imends = []
    for j in range(len(lst)):
        if lst[j] == imstart_id:
            imstarts.append(j)
        elif lst[j] == imend_id:
            imends.append(j)
            
    return imstarts, imends


def id_to_configs(id):
    global default
    configs = id.split("-")
    n = len(configs)
    if n == 10:
        design_name = configs[1] + "-" + configs[2] + "-" + configs[3]
        num_macros = default["macros_a"][configs[4]] if "a" in design_name else default["macros_b"][configs[4]]
        clock_freq = default["freq"][configs[5]]
        utilization = default["util"][configs[6]]
        macro_placement = default["macro_placement"]
        p_mesh = default["p_mesh"]
        filler = default["filler"][configs[9].split(".")[0]]
    else:
        design_name = configs[1] + "-" + configs[2]
        num_macros = default["macros_a"][configs[3]] if "a" in design_name else default["macros_b"][configs[3]]
        clock_freq = default["freq"][configs[4]]
        utilization = default["util"][configs[5]]
        macro_placement = default["macro_placement"]
        p_mesh = default["p_mesh"]
        filler = default["filler"][configs[8].split(".")[0]]
        
    config_prompt = f"""Following are the configuration details of the sample:
- Design name: {design_name}
- Number of macros: {num_macros}
- Clock frequency: {clock_freq}
- Utilization: {utilization}
- Macro placement: {macro_placement}
- Power mesh setting: {p_mesh}
- filler insertion: {filler}""".strip()
    return config_prompt
            
            
class GatingNetwork(nn.Module):
    """
    Gating Network: A simple MLP with softmax output and temperature scaling
    This network learns to combine multiple reward objectives based on the input context
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        temperature: float = 10,
        logit_scale: float = 1.0,
        hidden_dim: int = 1024,
        n_hidden: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones(1) * logit_scale)
        self.dropout_prob = dropout
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_features, hidden_dim))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features, bias=bias))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        # Apply the linear layers with ReLU and dropout
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout_prob > 0:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)
        # Apply softmax with temperature scaling
        x = F.softmax(x / self.temperature, dim=1)
        return x * self.logit_scale[0]
    
    
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

            
    def freeze_parameters(self):
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
            
    def unfreeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = True
            
        
        
    def forward(self, images, tokens):
        # Pass through the model
        h = self.text_encoder(input_ids=tokens).last_hidden_state
        h = self.down_proj(h)
        # Pass through the UNet model
        out = self.model(
            sample=images,
            timestep=0,
            encoder_hidden_states=h,
        ).sample
        
        return F.sigmoid(out)
    
def top_k_nrmse(prediction, label_image, percentile):
    k = int(len(label_image) * percentile)
    top_k_indices = np.argpartition(label_image, -k)[-k:]
    top_k_pred = prediction[top_k_indices]
    top_k_label = label_image[top_k_indices]
    k_normalized_root_mse = normalized_root_mse(top_k_label, top_k_pred, normalization='min-max')
    return k_normalized_root_mse
    
    
def peak_nrmse(prediction, label_image):
    """
    Calculate the Peak Normalized Root Mean Square Error (NRMSE) between the predicted and label images.
    """
    top_half = top_k_nrmse(prediction, label_image, 0.005)
    top_one = top_k_nrmse(prediction, label_image, 0.01)
    top_two = top_k_nrmse(prediction, label_image, 0.02)
    top_five = top_k_nrmse(prediction, label_image, 0.05)
    average = (top_half + top_one + top_two + top_five) / 4
    
    return top_half, top_one, top_two, top_five, average
    
            
            
# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--baseline", type=str, default="pretrained")
parser.add_argument("--model_path", type=str, default="/data1/felixchao/minicpm")
parser.add_argument("--regression_layer_path", type=str, default="/home/felixchaotw/mllm-physical-design/armo/regression_weights/MiniCPM-V-2_6_ArmoRM-Multi-Objective-Data-v0.1.pt")
parser.add_argument("--gating_network_path", type=str, default="/home/felixchaotw/mllm-physical-design/armo/gating_weights/config_gating_network_MiniCPM-V-2_6.pt")
parser.add_argument("--decoder_path", type=str, default="/data1/felixchao/output/mpgd_text_checkpoint-0.pth")
parser.add_argument("--logit_scale", type=float, default=1)
parser.add_argument("--temperature", type=float, default=10)
parser.add_argument("--n_hidden", type=int, default=3)
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--embed_dim", type=int, default=3584)
args = parser.parse_args()

device = f"cuda:{args.device}" if args.device >= 0 else "cpu"


# Load the gating network
print("Loading Decoder...")
config = Config()
decoder = Unetfeats(config)
decoder.load_state_dict(torch.load(args.decoder_path, weights_only=True, map_location=device))
decoder.to(device)
decoder.eval()


# Load the testing dataset
print("Loading testing dataset...")

testing_set = ["/home/felixchaotw/mllm-physical-design/armo/dataset/test_feature_desc_a.csv", "/home/felixchaotw/mllm-physical-design/armo/dataset/test_feature_desc_b.csv"]
testing_data = pd.concat([pd.read_csv(file) for file in testing_set])


print("Testing dataset loaded successfully!")
print("Number of test cases:", len(testing_data))

ssim_fn = SSIM(data_range=1, size_average=True, channel=1)
metric = 0.0
nrms = 0.0
p_nrms_half = 0.0
p_nrms_one = 0.0
p_nrms_two = 0.0
p_nrms_five = 0.0
p_nrms_avg = 0.0

for i, example in tqdm(testing_data.iterrows(), desc="Test cases"):
        image_id = example["id"]
        prompt = example["prompt"]
        numpy_images = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/feature/{image_id}")
        label_image = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/label/{image_id}").squeeze()
        label_image = torch.tensor(label_image).unsqueeze(0).unsqueeze(1).float().to(device)
        batch_image = numpy_images.transpose(2,0,1)
        image_tensors = torch.tensor(batch_image).unsqueeze(0).float().to(device)
        image_tensors = image_tensors * 2.0 - 1.0
        
        input_ids = decoder.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512, truncation=True).input_ids.to(device)
        
        with torch.no_grad():
            prediction = decoder(images=image_tensors, tokens=input_ids)
            ssim = ssim_fn(prediction, label_image)
            print(f"SSIM: {ssim.item()}")
            metric += ssim.item()
            img1 = prediction.squeeze().cpu().numpy()
            img2 = label_image.squeeze().cpu().numpy()
            nrmse_value = normalized_root_mse(img2.flatten(), img1.flatten(), normalization='min-max')
            nrms += nrmse_value
            print(f"NRMS: {nrmse_value}")
            p_half, p_one, p_two, p_five, p_avg = peak_nrmse(img1.flatten(), img2.flatten())
            print(f"Peak NRMSE 0.5%: {p_half}", f"Peak NRMSE 1%: {p_one}", f"Peak NRMSE 2%: {p_two}", f"Peak NRMSE 5%: {p_five}", f"Peak NRMSE average: {p_avg}")
            p_nrms_half += p_half
            p_nrms_one += p_one
            p_nrms_two += p_two
            p_nrms_five += p_five
            p_nrms_avg += p_avg


print("===> Avg. {}: {:.4f}".format("SSIM", metric / len(testing_data))) 
print("===> Avg. {}: {:.4f}".format("NRMS", nrms / len(testing_data)))
print("===> Avg. {}: {:.4f}".format("peak NRMSE 0.5%", p_nrms_half / len(testing_data)))
print("===> Avg. {}: {:.4f}".format("peak NRMSE 1%", p_nrms_one / len(testing_data)))
print("===> Avg. {}: {:.4f}".format("peak NRMSE 2%", p_nrms_two / len(testing_data)))
print("===> Avg. {}: {:.4f}".format("peak NRMSE 5%", p_nrms_five / len(testing_data)))
print("===> Avg. {}: {:.4f}".format("peak NRMSE average", p_nrms_avg / len(testing_data)))
print("===> Finished testing!")
            