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
from timm.models.vision_transformer import Mlp
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
from pytorch_msssim import SSIM
import models
from diffusers import UNet2DConditionModel


# Enable TF32 for improved performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

def find_token_for_gating(lst, token_id):
    """Find the last occurrence of a token_pattern in a list."""
    search_end = len(lst)
    for j in range(search_end - 1, -1, -1):
        if lst[j] == token_id:
            return j
    raise ValueError("Token pattern not found in the list.")

def find_token_for_image(lst, token_id):
    """Find the first occurrence of a token_pattern in a list."""
    for j in range(len(lst)):
        if lst[j] == token_id:
            return j
    raise ValueError("Token pattern not found in the list.")

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, image_token):
        B, N, C = image_token.shape
        kv = (
            self.kv(image_token)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)

        B, N, C = query.shape
        q = (
            self.q(query)
            .reshape(B, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q = q[0]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class VAEDecoder(nn.Module):
    def __init__(self, hidden_dims, embed_dim=3584, latent_dim=25):
        super().__init__()
        self.latent_dims = latent_dim
        self.hidden_dims = hidden_dims
        self.dp_1 = nn.Dropout(0.2)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 4)
        
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
        
        self.cross_attn = nn.ModuleList(
            [
                CrossAttention(dim=hidden_dims[i], num_heads=8) for i in range(3)
            ]
        )
        
        self.down_proj = nn.ModuleList(
            [
                Mlp(in_features=embed_dim,hidden_features=hidden_dims[i], out_features=hidden_dims[i]) for i in range(3)
            ]
        )
        
        
        
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
        
        self.shift_factor = nn.Parameter(
            torch.randn(latent_dim) * 0.02
        )
        
        self.scale_factor_list = [nn.Parameter(torch.randn(hidden_dims[i]) * 0.02) for i in range(3)]
        
        self._init_parameters()
        
    def forward(self, features, weights, image_tokens):
        z = self.reparameterize(features, weights)
        x = self.dp_1(self.decoder_input(z))
        x = x.view(-1, self.hidden_dims[0], 2, 2)
        for i in range(len(self.hidden_dims)-1):
            if i < 3:
                x = x.permute(0, 2, 3, 1).view(-1, 2**(2*(i+1)), self.hidden_dims[i])
                down_tokens = self.down_proj[i](image_tokens)
                x = x + self.cross_attn[i](x, down_tokens) * self.scale_factor_list[i].to(x.device)
                x = x.permute(0, 2, 1).view(-1, self.hidden_dims[i], 2**(i+1), 2**(i+1))
            x = self.decoder[i](x)
            
        x = self.final_layer(x)
        return x
    
    def reparameterize(self, scores, weights):
        return scores * weights + self.shift_factor
    

    def _init_parameters(self):
        trunc_normal_(self.shift_factor, std=0.02)
        for i in range(3):
            trunc_normal_(self.scale_factor_list[i], std=0.02)
            
            
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
            
            
# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--baseline", type=str, default="pretrained")
parser.add_argument("--model_path", type=str, default="/data1/felixchao/minicpm")
parser.add_argument("--regression_layer_path", type=str, default="/home/felixchaotw/mllm-physical-design/armo/regression_weights/MiniCPM-V-2_6_ArmoRM-Multi-Objective-Data-v0.1.pt")
parser.add_argument("--gating_network_path", type=str, default="/home/felixchaotw/mllm-physical-design/armo/gating_weights/config_gating_network_MiniCPM-V-2_6.pt")
parser.add_argument("--decoder_path", type=str, default="/home/felixchaotw/mllm-physical-design/armo/decoder/best_model.pth")
parser.add_argument("--logit_scale", type=float, default=1)
parser.add_argument("--temperature", type=float, default=10)
parser.add_argument("--n_hidden", type=int, default=3)
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--embed_dim", type=int, default=3584)
args = parser.parse_args()

device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

# Load regression layer
print("Loading regression layer...")
regression_layer = torch.load(args.regression_layer_path, map_location=device)["weight"]

n_attributes, hidden_size = regression_layer.shape


# Load model and processor
print("Loading model and processor...")
model = AutoModel.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,  # Use bfloat16 precision for model weights to save memory
    device_map=device,
    attn_implementation="flash_attention_2",  # Specify the attention implementation for efficiency
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(
    args.model_path, 
    trust_remote_code=True,
)

# Load the gating network
print("Loading gating network...")

gating_network = GatingNetwork(
    args.embed_dim,
    regression_layer.shape[0],
    n_hidden=args.n_hidden,
    hidden_dim=args.hidden_size,
    logit_scale=args.logit_scale,
    temperature=args.temperature,
    dropout=args.dropout,
)

gating_network.load_state_dict(torch.load(args.gating_network_path, weights_only=True, map_location=device))
gating_network.to(device)
gating_network.eval()


# Load the gating network
print("Loading Decoder...")
if args.baseline == "pure":
    decoder = VisionDecoder([512, 256, 128, 64, 32, 16, 8], embed_dim=args.embed_dim)
else:
    decoder = Unetfeats(unet_path="/data1/felixchao/diffusion", embed_dim=args.embed_dim, latent_dim=25)
decoder.load_state_dict(torch.load(args.decoder_path, weights_only=True, map_location=device))
decoder.to(device)
decoder.eval()

print("Model and gating network loaded successfully!")

# Load the testing dataset
print("Loading testing dataset...")

testing_set = ["/home/felixchaotw/mllm-physical-design/armo/dataset/test_df_a.csv", "/home/felixchaotw/mllm-physical-design/armo/dataset/test_df_b.csv"]
testing_data = pd.concat([pd.read_csv(file) for file in testing_set])


print("Testing dataset loaded successfully!")
print("Number of test cases:", len(testing_data))

ssim_fn = SSIM(data_range=1, size_average=True, channel=1)
metric = 0.0

if args.baseline == "pretrained":
    opt = {'task': 'congestion_gpdl', 'save_path': 'work_dir/congestion_gpdl/', 'pretrained': '/home/felixchaotw/CircuitNet/model/congestion.pth', 'max_iters': 200000, 'plot_roc': False, 'arg_file': None, 'cpu': False, 'dataroot': '../../training_set/congestion', 'ann_file_train': './files/train_N28.csv', 'ann_file_test': './files/test_N28.csv', 'dataset_type': 'CongestionDataset', 'batch_size': 16, 'aug_pipeline': ['Flip'], 'model_type': 'GPDL', 'in_channels': 3, 'out_channels': 1, 'lr': 0.0002, 'weight_decay': 0, 'loss_type': 'MSELoss', 'eval_metric': ['NRMS', 'SSIM', 'EMD'], 'ann_file': './files/test_N28.csv', 'test_mode': True}
    model = models.__dict__["GPDL"](**opt)
    model.init_weights(**opt)
    model.to(device)
    
    for i, example in tqdm(testing_data.iterrows(), desc="Pretrained"):
        numpy_images = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/feature/{example['id']}")
        label_image = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/label/{example['id']}").squeeze()
        label_image = torch.tensor(label_image).unsqueeze(0).unsqueeze(1).float().to(device)
        batch_image = numpy_images.transpose(2,0,1)
        with torch.no_grad():
            input_image = torch.tensor(batch_image).unsqueeze(0).float().to(device)
            output_image = model(input_image)
            ssim = ssim_fn(output_image, label_image)
            metric += ssim.item()
            
elif args.baseline == "vision":
    
    for i, example in tqdm(testing_data.iterrows(), desc="Vision cases"):
        cur_msgs = []
        system_message = id_to_configs(example["id"])
        user_message = "Can you predict the congestion level of this sample from the given images?"
        image_id = example["id"]
        numpy_images = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/feature/{image_id}")
        label_image = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/label/{image_id}").squeeze()
        label_image = torch.tensor(label_image).unsqueeze(0).unsqueeze(1).float().to(device)
        batch_image = numpy_images.transpose(2,0,1)
        image_features = []
        
        cur_msgs.append(system_message)
        
        for image in batch_image:
            image_features.append(Image.fromarray(np.uint8(image * 255)))
            cur_msgs.append("(<image>./</image>)")

        cur_msgs.append(user_message)
        
        msg = {
            "role": "user",
            "content": "\n".join(cur_msgs),
        }
        
        conv_formatted = processor.tokenizer.apply_chat_template(
            [msg], tokenize=False, add_generation_prompt=False
        )
        # Tokenize the formatted conversation and move tensors to the specified device
        conv_tokenized = processor([conv_formatted], image_features, return_tensors="pt").to(device)

        input_ids = conv_tokenized.data["input_ids"]
        position_ids = torch.arange(conv_tokenized.data["input_ids"].size(1)).long()
        conv_tokenized.data["position_ids"] = position_ids.unsqueeze(0).to(device)
            
        with torch.no_grad():
            output = model(data=conv_tokenized, output_hidden_states=True)
            last_hidden_state = output.hidden_states[-1][0].to(device)

            # Find the position of the gating token and extract embeddings
            gating_token_position = find_token_for_gating(
                input_ids[0].tolist(), processor.tokenizer.im_end_id
            )
            prompt_embedding = last_hidden_state[gating_token_position].float()
            last_token_embedding = last_hidden_state[-1].float()
            
            prediction = decoder(last_token_embedding.unsqueeze(0))
            ssim = ssim_fn(prediction, label_image)
            metric += ssim.item()
else:
    for i, example in tqdm(testing_data.iterrows(), desc="Test cases"):
        cur_msgs = []
        system_message = id_to_configs(example["id"])
        user_message = "Can you predict the congestion level of this sample from the given images?"
        image_id = example["id"]
        numpy_images = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/feature/{image_id}")
        label_image = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/label/{image_id}").squeeze()
        label_image = torch.tensor(label_image).unsqueeze(0).unsqueeze(1).float().to(device)
        batch_image = numpy_images.transpose(2,0,1)
        image_tensors = torch.tensor(batch_image).unsqueeze(0).float().to(device)
        image_tensors = image_tensors * 2.0 - 1.0
        image_features = []
        
        cur_msgs.append(system_message)
        
        for image in batch_image:
            image_features.append(Image.fromarray(np.uint8(image * 255)))
            cur_msgs.append("(<image>./</image>)")

        cur_msgs.append(user_message)
        
        msg = {
            "role": "user",
            "content": "\n".join(cur_msgs),
        }
        
        conv_formatted = processor.tokenizer.apply_chat_template(
            [msg], tokenize=False, add_generation_prompt=False
        )
        # Tokenize the formatted conversation and move tensors to the specified device
        conv_tokenized = processor([conv_formatted], image_features, return_tensors="pt").to(device)

        input_ids = conv_tokenized.data["input_ids"]
        position_ids = torch.arange(conv_tokenized.data["input_ids"].size(1)).long()
        conv_tokenized.data["position_ids"] = position_ids.unsqueeze(0).to(device)
            
        with torch.no_grad():
            output = model(data=conv_tokenized, output_hidden_states=True)
            last_hidden_state = output.hidden_states[-1][0].to(device)

            # Find the position of the gating token and extract embeddings
            gating_token_position = find_token_for_gating(
                input_ids[0].tolist(), processor.tokenizer.im_end_id
            )
            prompt_embedding = last_hidden_state[gating_token_position].float()
            last_token_embedding = last_hidden_state[-1].float()
            
            gating_weights = gating_network(prompt_embedding.unsqueeze(0))
            multi_rewards = last_token_embedding.unsqueeze(0) @ regression_layer.T
            
            prediction = decoder(image_tensors, multi_rewards, gating_weights, last_token_embedding.unsqueeze(0))
            ssim = ssim_fn(prediction, label_image)
            print(f"SSIM: {ssim.item()}")
            metric += ssim.item()

print("===> Avg. {}: {:.4f}".format("SSIM", metric / len(testing_data))) 
            