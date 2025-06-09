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
    
    
# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--baseline", type=str, default="pretrained")
parser.add_argument("--model_path", type=str, default="/data1/felixchao/minicpm")
parser.add_argument("--regression_layer_path", type=str, default="/home/felixchaotw/mllm-physical-design/armo/regression_weights/MiniCPM-V-2_6_ArmoRM-Multi-Objective-Data-v0.1.pt")
parser.add_argument("--gating_network_path", type=str, default="/home/felixchaotw/mllm-physical-design/armo/gating_weights/config_gating_network_MiniCPM-V-2_6.pt")
parser.add_argument("--decoder_path", type=str, default="/data1/felixchao/output/checkpoint-0.pth")
parser.add_argument("--logit_scale", type=float, default=1)
parser.add_argument("--temperature", type=float, default=10)
parser.add_argument("--n_hidden", type=int, default=3)
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--embed_dim", type=int, default=3584)
args = parser.parse_args()
    
    
device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
ssim_fn = SSIM(data_range=1, size_average=True, channel=1)

testing_set = ["/home/felixchaotw/mllm-physical-design/armo/dataset/train_feature_desc.csv", "/home/felixchaotw/mllm-physical-design/armo/dataset/test_feature_desc_a.csv", "/home/felixchaotw/mllm-physical-design/armo/dataset/test_feature_desc_b.csv"]
testing_data = pd.concat([pd.read_csv(file) for file in testing_set])
    
decoder = Unetfeats(unet_path="/data1/felixchao/diffusion", text_encoder_path="/data1/felixchao/sd3_5", embed_dim=512, latent_dim=25)
decoder.load_state_dict(torch.load(args.decoder_path, weights_only=True, map_location=device))
decoder.to(device)
decoder.eval()


for i, example in tqdm(testing_data.iterrows(), desc="Test cases"):
    design_id = example["id"]
    label_image = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/label/{design_id}").squeeze()
    label_tensor = torch.tensor(label_image).unsqueeze(0).unsqueeze(1).float().to(device)

    opt = {'task': 'congestion_gpdl', 'save_path': 'work_dir/congestion_gpdl/', 'pretrained': '/home/felixchaotw/CircuitNet/model/congestion.pth', 'max_iters': 200000, 'plot_roc': False, 'arg_file': None, 'cpu': False, 'dataroot': '../../training_set/congestion', 'ann_file_train': './files/train_N28.csv', 'ann_file_test': './files/test_N28.csv', 'dataset_type': 'CongestionDataset', 'batch_size': 16, 'aug_pipeline': ['Flip'], 'model_type': 'GPDL', 'in_channels': 3, 'out_channels': 1, 'lr': 0.0002, 'weight_decay': 0, 'loss_type': 'MSELoss', 'eval_metric': ['NRMS', 'SSIM', 'EMD'], 'ann_file': './files/test_N28.csv', 'test_mode': True}
    model = models.__dict__["GPDL"](**opt)
    model.init_weights(**opt)
    model.to(device)
        
    numpy_images = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/feature/{design_id}")
    batch_image = numpy_images.transpose(2,0,1)
    input_image = torch.tensor(batch_image).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output_image = model(input_image)
    gpdl_ssim = ssim_fn(output_image, label_tensor)
    gpdl_output_image = output_image.squeeze().cpu().numpy()
    gpdl_output_image = (gpdl_output_image - (gpdl_output_image.min() + 0.07)) / (gpdl_output_image.max() - (gpdl_output_image.min()+0.07))
    gpdl_output_image = np.clip(gpdl_output_image, 0, 1)
    gpdl_image_pil = Image.fromarray(np.uint8(gpdl_output_image * 255))


    prompt = testing_data[testing_data["id"] == design_id]["prompt"].values[0]
    image_tensors = torch.tensor(batch_image).unsqueeze(0).float().to(device)
    image_tensors = image_tensors * 2.0 - 1.0

    input_ids = decoder.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512, truncation=True).input_ids.to(device)
    with torch.no_grad():
        prediction = decoder(images=image_tensors, tokens=input_ids)
    our_ssim = ssim_fn(prediction, label_tensor)
    output_image = prediction.squeeze().cpu().numpy()
    output_image = (output_image - (output_image.min() + 0.07)) / (output_image.max() - (output_image.min()+0.07))
    output_image = np.clip(output_image, 0, 1)
    image_pil = Image.fromarray(np.uint8(output_image * 255))
    
    
    if gpdl_ssim < our_ssim:
        print(f"Design ID: {design_id}")
        print(f"GPDL SSIM: {gpdl_ssim}, Ours SSIM: {our_ssim}")
        gpdl_image_pil.save(f"/home/felixchaotw/mllm-physical-design/armo/qualitive_anaylsis/{design_id}_gpdl.png")
        image_pil.save(f"/home/felixchaotw/mllm-physical-design/armo/qualitive_anaylsis/{design_id}_ours.png")