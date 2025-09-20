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
from transformers import T5EncoderModel, AutoTokenizer
from skimage.metrics import normalized_root_mse
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, roc_auc_score


# Enable TF32 for improved performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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


def correlation_metrics(prediction, label_image):
    """
    Calculate the Spearman and Pearson and Kendall correlation coefficients between the predicted and label images.
    """
    # Spearman correlation
    spearman_corr, _ = stats.spearmanr(label_image, prediction)
    
    # Pearson correlation
    pearson_corr, _ = stats.pearsonr(label_image, prediction)
    
    # Kendall correlation
    kendall_corr, _ = stats.kendalltau(label_image, prediction)
    
    return spearman_corr, pearson_corr, kendall_corr
            
            
# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--baseline", type=str, default="no")
parser.add_argument("--model_path", type=str, default="/data1/felixchao/minicpm")
parser.add_argument("--regression_layer_path", type=str, default="/home/felixchaotw/mllm-physical-design/DRV/regression_weights/MiniCPM-V-2_6_ArmoRM-Multi-Objective-Data-v0.1.pt")
parser.add_argument("--gating_network_path", type=str, default="/home/felixchaotw/mllm-physical-design/DRV/gating_weights/config_gating_network_MiniCPM-V-2_6.pt")
parser.add_argument("--decoder_path", type=str, default="/data1/felixchao/DRC/checkpoint-best.pth")
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
decoder = Unetfeats(unet_path="/data1/felixchao/diffusion", text_encoder_path="/data1/felixchao/sd3_5", embed_dim=args.embed_dim, latent_dim=25)
decoder.load_state_dict(torch.load(args.decoder_path, weights_only=True, map_location=device))
decoder.to(device)
decoder.eval()

print("Model and gating network loaded successfully!")

# Load the testing dataset
print("Loading testing dataset...")

testing_set = ["/home/felixchaotw/mllm-physical-design/DRV/dataset/test_feature_desc_a.csv", "/home/felixchaotw/mllm-physical-design/DRV/dataset/test_feature_desc_b.csv"]
# testing_set = ["/home/felixchaotw/mllm-physical-design/DRV/dataset/train_feature_desc.csv"]
testing_data = pd.concat([pd.read_csv(file) for file in testing_set])


print("Testing dataset loaded successfully!")
print("Number of test cases:", len(testing_data))

ssim_fn = SSIM(data_range=1, size_average=True, channel=1)
loss_fn = nn.MSELoss()
acc = 0.0
precision = 0.0
f1 = 0.0
metric = 0.0
nrms = 0.0
p_nrms_half = 0.0
p_nrms_one = 0.0
p_nrms_two = 0.0
p_nrms_five = 0.0
p_nrms_avg = 0.0
srcc = 0.0
plcc = 0.0
krcc = 0.0
tpr = 0.0
fpr = 0.0
roc_auc = 0.0
valid_n = 0


for i, example in tqdm(testing_data.iterrows(), desc="Test cases"):
        image_id = example["id"]
        prompt = example["prompt"]
        numpy_images = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/DRC/feature/{image_id}")
        label_image = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/DRC/label/{image_id}").squeeze()
        label_image = torch.tensor(label_image).unsqueeze(0).unsqueeze(1).float().to(device)
        batch_image = numpy_images.transpose(2,0,1)
        image_tensors = torch.tensor(batch_image).unsqueeze(0).float().to(device)
        image_tensors = image_tensors * 2.0 - 1.0
        
        input_ids = decoder.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512, truncation=True).input_ids.to(device)
        
            
        with torch.no_grad():
            prediction = decoder(images=image_tensors, tokens=input_ids)
            ssim = ssim_fn(prediction, label_image)
            metric += ssim.item()
            print(f"SSIM: {ssim.item()}")
            img1 = prediction.squeeze().cpu().numpy()
            img2 = label_image.squeeze().cpu().numpy()
            img2_flatten = np.where(img2.flatten() >= 0.1, 1.0, 0.0)
            img1_flatten = np.where(img1.flatten() >= 0.1, 1.0, 0.0)
            acc_score = accuracy_score(img2_flatten, img1_flatten)
            acc += acc_score
            print(f"Accuracy: {acc_score}")
            nrmse_value = normalized_root_mse(img2.flatten(), img1.flatten(), normalization='min-max')
            nrms += nrmse_value
            print(f"NRMS: {nrmse_value}")
            cf = confusion_matrix(img2_flatten, img1_flatten, labels=[0, 1])
            tn = cf[0, 0]
            fp = cf[0, 1]
            fn = cf[1, 0]
            tp = cf[1, 1]
            if int(fp)==0 and int(tn)==0:
                continue
            elif int(tp)==0 and int(fn)==0:
                continue
            elif int(tp)==0 and int(fp)==0:
                continue
            else:
                tpr_score = tp / (tp + fn)
                fpr_score = fp / (fp + tn)
                tpr += tpr_score
                fpr += fpr_score
                print(f"TPR: {tpr_score}")
                print(f"FPR: {fpr_score}")
                prec_score = precision_score(img2_flatten, img1_flatten, zero_division=1)
                precision += prec_score
                f1_sc = f1_score(img2_flatten, img1_flatten, zero_division=1)
                f1 += f1_sc
                print(f"Precision: {prec_score}")
                print(f"F1 Score: {f1_sc}")
                try:
                    roc_auc_sc = roc_auc_score(img2_flatten, img1.flatten())
                except ValueError:
                    roc_auc_sc = 1.0
                roc_auc += roc_auc_sc
                print(f"ROC_AUC: {roc_auc_sc}")
                valid_n += 1


print("===> Avg. {}: {:.4f}".format("Accuracy", acc / len(testing_data)))
print("===> Avg. {}: {:.4f}".format("Precision", precision / valid_n))
print("===> Avg. {}: {:.4f}".format("F1 Score", f1 / valid_n))
print("===> Avg. {}: {:.4f}".format("SSIM", metric / len(testing_data))) 
print("===> Avg. {}: {:.4f}".format("NRMS", nrms / len(testing_data)))
print("===> Avg. {}: {:.4f}".format("TPR", tpr / valid_n))
print("===> Avg. {}: {:.4f}".format("FPR", fpr / valid_n))
print("===> Avg. {}: {:.4f}".format("ROC_AUC", roc_auc / valid_n))
print("===> Finished testing!")
            