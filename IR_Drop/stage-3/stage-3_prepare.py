import os
import torch
import numpy as np
from safetensors.torch import load_file
from safetensors.torch import save_file
from argparse import ArgumentParser
from tqdm.auto import tqdm
from scipy.stats import spearmanr
import pandas as pd
from glob import glob
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoProcessor
from PIL import Image
from scipy import stats
from sklearn.metrics import ndcg_score
import heapq
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use(['science','grid','retro'])


# Enable TF32 for improved performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Define attributes (reward objectives)
attributes = [
 "horizontal_power_distribution_symmetry",
 "mean_power_sca",
 "heat_intensity_correlation",
 "central_power_saturation",
 "vertical_power_distribution_symmetry",
 "proximity_power_pattern_asymmetry",
 "macro_power_proximity",
 "mean_power_density_deviation",
 "edge_power_intensity",
 "power_sink_effect",
 "mean_power_all",
 "mean_power_i",
 "power_balance_ratio",
 "power_gradient_variation",
 "localized_coupling_variability",
 "power_intensity_anomaly_detection",
 "localized_gradient_intensity",
 "spatial_correlation_power_i",
 "uniformity_index_power_i",
 "spatial_density_power_i"
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


def find_token_for_gating(lst, token_id):
    """Find the last occurrence of a token_pattern in a list."""
    search_end = len(lst)
    for j in range(search_end - 1, -1, -1):
        if lst[j] == token_id:
            return j
    raise ValueError("Token pattern not found in the list.")


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
    

def features_description(gating_weights, multi_rewards, label):
    """
    Generate feature descriptions based on the gating weights, multi-rewards, and label
    """
    # Sort the gating weights and multi-rewards based on the gating weights
    gating_weights = gating_weights.squeeze()
    multi_rewards = multi_rewards.squeeze()
    # Sort the gating weights and multi-rewards in descending order
    sorted_indices = torch.argsort(gating_weights, descending=True)
    desc = "The design has the following features:\n"
    # Generate the feature descriptions
    feature_description = []
    for i in range(len(sorted_indices)):
        feature_description.append(
            f"{attributes[sorted_indices[i]]} is {multi_rewards[sorted_indices[i]]:.4f}, the importance is {gating_weights[sorted_indices[i]]:.4f}."
        )
    # Add the predicted congestion level to the feature descriptions
    feature_description.append(f"Congestion level: {label:.2f}.")
    return desc + "\n".join(feature_description)


# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="/data1/felixchao/minicpm")
parser.add_argument("--regression_layer_path", type=str, default="/home/felixchaotw/mllm-physical-design/IR_Drop/regression_weights/MiniCPM-V-2_6_ArmoRM-Multi-Objective-Data-v0.1.pt")
parser.add_argument("--gating_network_path", type=str, default="/home/felixchaotw/mllm-physical-design/IR_Drop/gating_weights/config_gating_network_MiniCPM-V-2_6.pt")
parser.add_argument("--dataset", type=str, default="/home/felixchaotw/mllm-physical-design/IR_Drop/dataset/train_df.csv")
parser.add_argument("--device", type=int, default=0)
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

print("Model and gating network loaded successfully!")

# Load the testing dataset
print("Loading testing dataset...")

train_df = pd.read_csv(args.dataset)

print("Testing dataset loaded successfully!")
print("Number of test cases:", len(train_df))


images_list = []
image_tokens_embedding_list = []
gating_weights_list = []
multi_rewards_list = []
label_list = []
last_hidden_tokens_list = []


for i, example in tqdm(train_df.iterrows(), desc="Test cases"):
    cur_msgs = []
    system_message = id_to_configs(example["id"])
    user_message = "Can you predict the congestion level of this sample from the given images?"
    image_id = example["id"]
    numpy_images = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/IR_Drop/feature/{image_id}")
    label_image = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/IR_Drop/label/{image_id}").squeeze()
    label = torch.tensor(label_image).float()
    batch_image = numpy_images.transpose(2,0,1)
    image_tensor = torch.tensor(batch_image).float()
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
        last_hidden_state = output.hidden_states[-1][0]
        
        # Find the position of the gating token and extract embeddings
        gating_token_position = find_token_for_gating(
            input_ids[0].tolist(), processor.tokenizer.im_end_id
        )
        
        prompt_embedding = last_hidden_state[gating_token_position].float()
        last_token_embedding = last_hidden_state[-1].float()
            
        gating_weights = gating_network(prompt_embedding.unsqueeze(0))
        multi_rewards = last_token_embedding.unsqueeze(0) @ regression_layer.T
        
        # Find the position of the image token and extract embeddings
        image_start_positions, image_end_positions = find_token_for_image(
            input_ids[0].tolist(), processor.tokenizer.im_start_id, processor.tokenizer.im_end_id
        )
        
        slicing_indicies = []
        
        for j in range(len(image_start_positions)):
            slicing_indicies += list(range(image_start_positions[j] + 1, image_end_positions[j]))
        
        # Checking the vision tokens length is fixed
        # print("L:", len(slicing_indicies))
        image_tokens_embedding = last_hidden_state[slicing_indicies].permute(1,0).reshape(-1, 8, 8).float()
        # print(image_tokens_embedding.shape)
        
        pred_val = torch.sum(multi_rewards * gating_weights, dim=-1).item()
        
        # Generate feature descriptions
        feat_desc = features_description(gating_weights, multi_rewards, pred_val * 100)
        train_df.loc[i, "prompt"] = feat_desc
        train_df.loc[i, "config"] = system_message
        print(f"Sample {i+1} - {image_id}: {feat_desc}")
        
        image_tokens_embedding_list.append(image_tokens_embedding.cpu())
        gating_weights_list.append(gating_weights.squeeze().cpu())
        multi_rewards_list.append(multi_rewards.squeeze().cpu())
        label_list.append(label.cpu())
        last_hidden_tokens_list.append(last_token_embedding.cpu())
        images_list.append(image_tensor)
        
# Convert lists of embeddings to tensors
image_tokens_embedding_tensors = torch.stack(image_tokens_embedding_list)
gating_weights_embeddings = torch.stack(gating_weights_list)
multi_rewards_tensors = torch.stack(multi_rewards_list)
label_tensors = torch.stack(label_list)
last_hidden_token_tensors = torch.stack(last_hidden_tokens_list)
image_tensors = torch.stack(images_list)

# samples = {
#     "images": image_tensors,
#     "gating_weights": gating_weights_embeddings,
#     "last_hidden_tokens": last_hidden_token_tensors,
#     "multi_rewards": multi_rewards_tensors,
#     "label": label_tensors
# }

samples = {
    "vlm_tokens": image_tokens_embedding_tensors,
}

save_file(samples, f"/data1/felixchao/vlm_tokens.safetensors")
print("Embeddings extracted successfully!")

train_df = train_df[["id", "prompt", "config"]]
train_df.to_csv("/home/felixchaotw/mllm-physical-design/IR_Drop/dataset/train_feature_desc.csv", index=False)
print("Feature descriptions generated successfully!")