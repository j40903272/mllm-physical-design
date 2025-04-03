import os
import models
import datasets
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from argparse import ArgumentParser
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from glob import glob
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoProcessor
from PIL import Image
from scipy import stats
from timm.models.vision_transformer import Mlp
from timm.layers import trunc_normal_
from transformers import get_scheduler
from pytorch_msssim import SSIM
from diffusers import UNet2DConditionModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler



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
            
            
class CongestionEvalDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "id": row["id"],
            "config": id_to_configs(row["id"])
        }


def load_components(args, device):
    print("Loading regression layer...")
    regression_layer = torch.load(args.regression_layer_path, map_location=device)["weight"]

    print("Loading model and processor...")
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

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
    gating_network.to(device).eval()

    print("Loading decoder...")
    if args.baseline == "pure":
        decoder = VisionDecoder([512, 256, 128, 64, 32, 16, 8], embed_dim=args.embed_dim)
    else:
        decoder = Unetfeats(unet_path="stable-diffusion-v1-5/stable-diffusion-v1-5", embed_dim=args.embed_dim, latent_dim=25)
    decoder.load_state_dict(torch.load(args.decoder_path, weights_only=True, map_location=device))
    decoder.to(device).eval()

    return model, processor, gating_network, decoder, regression_layer


def load_testing_data():
    print("Loading testing dataset...")
    testing_set = [
        "/lustre/fsw/portfolios/nvr/users/yundat/mllm-physical-design/armo/dataset/test_df_a.csv",
        "/lustre/fsw/portfolios/nvr/users/yundat/mllm-physical-design/armo/dataset/test_df_b.csv",
    ]
    testing_data = pd.concat([pd.read_csv(file) for file in testing_set])
    print("Testing dataset loaded successfully! Total samples:", len(testing_data))
    return testing_data


def evaluate_distributed(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model, processor, gating_network, decoder, regression_layer = load_components(args, device)

    # Wrap model and decoder with DDP
    model = DDP(model, device_ids=[local_rank])
    decoder = DDP(decoder, device_ids=[local_rank])
    gating_network = DDP(gating_network, device_ids=[local_rank])

    # Load data and create sampler
    df = load_testing_data()
    dataset = CongestionEvalDataset(df)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    ssim_fn = SSIM(data_range=1, size_average=True, channel=1)
    local_metric = 0.0

    for e, batch in enumerate(dataloader):
        image_id = batch["id"][0]
        cur_msgs = [batch["config"][0]]
        user_message = "Can you predict the congestion level of this sample from the given images?"

        numpy_images = np.load(f"/lustre/fsw/portfolios/nvr/users/yundat/mllm-physical-design/dataset/CircuitNet-N28/Dataset/congestion/feature/{image_id}")
        label_image = np.load(f"/lustre/fsw/portfolios/nvr/users/yundat/mllm-physical-design/dataset/CircuitNet-N28/Dataset/congestion/label/{image_id}").squeeze()
        label_image = torch.tensor(label_image).unsqueeze(0).unsqueeze(1).float().to(device)
        batch_image = numpy_images.transpose(2, 0, 1)
        image_tensors = torch.tensor(batch_image).unsqueeze(0).float().to(device)
        image_tensors = image_tensors * 2.0 - 1.0

        image_features = [Image.fromarray(np.uint8(image * 255)) for image in batch_image]
        cur_msgs += ["(<image>./</image>)" for _ in batch_image] + [user_message]
        msg = {"role": "user", "content": "\n".join(cur_msgs)}

        conv_formatted = processor.tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=False)
        conv_tokenized = processor([conv_formatted], image_features, return_tensors="pt").to(device)
        input_ids = conv_tokenized.data["input_ids"]
        position_ids = torch.arange(input_ids.size(1)).long().unsqueeze(0).to(device)
        conv_tokenized.data["position_ids"] = position_ids

        with torch.no_grad():
            output = model(data=conv_tokenized, output_hidden_states=True)
            last_hidden_state = output.hidden_states[-1][0].to(device)

            gating_token_position = find_token_for_gating(input_ids[0].tolist(), processor.tokenizer.im_end_id)
            prompt_embedding = last_hidden_state[gating_token_position].float()
            last_token_embedding = last_hidden_state[-1].float()

            gating_weights = gating_network(prompt_embedding.unsqueeze(0))
            multi_rewards = last_token_embedding.unsqueeze(0) @ regression_layer.T

            prediction = decoder(image_tensors, multi_rewards, gating_weights, last_token_embedding.unsqueeze(0))
            ssim = ssim_fn(prediction, label_image)
            local_metric += ssim.item()

            interval = len(dataloader) // 20
            if rank == 0 and e % interval == 0:
                print(f"{e}/{len(dataloader)}")

    # Reduce metrics across all ranks
    metric_tensor = torch.tensor(local_metric, dtype=torch.float32, device=device)
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)

    # Print only from rank 0
    if rank == 0:
        avg_ssim = metric_tensor.item() / len(dataloader)
        print("===> Avg. SSIM: {:.4f}".format(avg_ssim))

    dist.destroy_process_group()


def main():
    parser = ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--baseline", type=str, default="no")
    parser.add_argument("--model_path", type=str, default="openbmb/MiniCPM-V-2_6")
    parser.add_argument("--regression_layer_path", type=str, default="/lustre/fsw/portfolios/nvr/users/yundat/mllm-physical-design/armo/regression_weights/MiniCPM-V-2_6_ArmoRM-Multi-Objective-Data-v0.1.pt")
    parser.add_argument("--gating_network_path", type=str, default="/lustre/fsw/portfolios/nvr/users/yundat/mllm-physical-design/armo/gating_weights/config_gating_network_MiniCPM-V-2_6.pt")
    parser.add_argument("--decoder_path", type=str)
    parser.add_argument("--logit_scale", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=10)
    parser.add_argument("--n_hidden", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--embed_dim", type=int, default=3584)
    args = parser.parse_args()

    evaluate_distributed(args)

if __name__ == "__main__":
    main()
            