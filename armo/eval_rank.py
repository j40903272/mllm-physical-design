import os
import torch
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
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoProcessor
from PIL import Image
from scipy import stats
from sklearn.metrics import ndcg_score
import models
import heapq
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use(['science','grid','retro'])


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


tile_size = 16
image_size = 256


def get_tiles_congestion(image_array):
    tiles = []
    for x in range(0, image_size, tile_size):
        for y in range(0, image_size, tile_size):
            tile = image_array[x:x+tile_size, y:y+tile_size]
            tiles.append(np.mean(tile))
            
    tiles = heapq.nlargest(20, tiles)
    return tiles

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


def replaceWithRank(arr):
    n = len(arr)
    res = [0] * n
    pq = []
    for i in range(n):
        heapq.heappush(pq, (arr[i], i))

    rank = 0
    lastNum = float('inf')

    while pq:
        curr, index = heapq.heappop(pq)

        if lastNum == float('inf') or curr != lastNum:
            rank += 1
            
        res[index] = rank - 1
        lastNum = curr

    return res


def evalute_corr(congestion_set, predicted, corr_metrics):
    x = np.array(list(congestion_set.values()))[::20]
    x_label = list(congestion_set.keys())
    y = np.array([predicted[id] for id in x_label])[::20]
    results = {}
    if "PLCC" in corr_metrics:
        results["PLCC"] = stats.pearsonr(x, y)
    if "SRCC" in corr_metrics:
        results["SRCC"] = stats.spearmanr(x, y)
    if "KRCC" in corr_metrics:
        results["KRCC"] = stats.kendalltau(x, y)
    
    return results


def evaluate_design(df):
    global design_type, baseline
    congestion_set = dict(zip(df["id"], df["label"]))
    congestion_set = dict(sorted(congestion_set.items(), key=lambda x: x[1]))
    predicted = dict(zip(df["id"], df["prediction"]))
    corr_metrics = ["PLCC", "SRCC", "KRCC"]
    results = evalute_corr(congestion_set, predicted, corr_metrics)
    x = list(congestion_set.keys())[::20]
    x = [name.split("-")[0] for name in x]
    x_label = list(range(0,len(x)))
    
    y = [predicted[file_path] for file_path in congestion_set.keys()][::20]
    y_label = replaceWithRank(y)
    plt.figure(figsize=(10,5))
    plt.plot(x, y_label, linewidth="2", marker="o")
    plt.plot(x, x_label, linewidth="2", marker="o")
    plt.xticks(ticks=x_label, labels=x, rotation=90)
    plt.xlabel("Images")
    plt.ylabel("Rank Order")
    if baseline:
        plt.title("GPDL Baseline")
        plt.savefig(f"/home/felixchaotw/mllm-physical-design/armo/plots/GPDL_Baseline_{design_type}.png")
    else:
        plt.title("Interpretible MLLM")
        plt.savefig(f"/home/felixchaotw/mllm-physical-design/armo/plots/Interpretible_MLLM_{design_type}.png")
    return results


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
    
def load_embeddings(embedding_path_pattern, device):
    """
    Load embeddings from safetensors files
    """
    # Examine if the embedding path pattern is correct
    file_paths = glob(embedding_path_pattern)
    if len(file_paths) == 0:
        raise ValueError(f"Embeddings not found at {embedding_path_pattern}")
    embeddings, prompt_embeddings = [], []
    for embedding_path in file_paths:
        embeddings_data = load_file(embedding_path)
        embeddings.append(embeddings_data["embeddings"].to(device))
        prompt_embeddings.append(embeddings_data["prompt_embeddings"].to(device))
    embeddings = torch.cat(embeddings, dim=0).float()
    prompt_embeddings = torch.cat(prompt_embeddings, dim=0).float()
    return embeddings, prompt_embeddings


# Set up argument parser
parser = ArgumentParser()
parser.add_argument("--baseline", action="store_true", default=False)
parser.add_argument("--model_path", type=str, default="openbmb/MiniCPM-V-2_6")
parser.add_argument("--regression_layer_path", type=str, default="/home/felixchaotw/mllm-physical-design/armo/regression_weights/MiniCPM-V-2_6_ArmoRM-Multi-Objective-Data-v0.1.pt")
parser.add_argument("--gating_network_path", type=str, default="/home/felixchaotw/mllm-physical-design/armo/gating_weights/config_gating_network_MiniCPM-V-2_6.pt")
parser.add_argument(
    "--eval_dataset",
    type=str,
    default="zero-riscy-b",
)
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

design_type = args.eval_dataset
baseline = args.baseline

if design_type == "zero-riscy-a":
    testing_path = "/home/felixchaotw/mllm-physical-design/armo/dataset/test_df_a.csv"
else:
    testing_path = f"/home/felixchaotw/mllm-physical-design/armo/dataset/test_df_b.csv"
    
testing_df = pd.read_csv(testing_path)
testing_df = testing_df[testing_df["label"].notna()]


print("Testing dataset loaded successfully!")
print("Design type:", design_type)
print("Number of test cases:", len(testing_df))


if args.baseline:
    opt = {'task': 'congestion_gpdl', 'save_path': 'work_dir/congestion_gpdl/', 'pretrained': '/home/felixchaotw/CircuitNet/model/congestion.pth', 'max_iters': 200000, 'plot_roc': False, 'arg_file': None, 'cpu': False, 'dataroot': '../../training_set/congestion', 'ann_file_train': './files/train_N28.csv', 'ann_file_test': './files/test_N28.csv', 'dataset_type': 'CongestionDataset', 'batch_size': 16, 'aug_pipeline': ['Flip'], 'model_type': 'GPDL', 'in_channels': 3, 'out_channels': 1, 'lr': 0.0002, 'weight_decay': 0, 'loss_type': 'MSELoss', 'eval_metric': ['NRMS', 'SSIM', 'EMD'], 'ann_file': './files/test_N28.csv', 'test_mode': True}
    model = models.__dict__["GPDL"](**opt)
    model.init_weights(**opt)
    model.to(device)
    
    for i, example in tqdm(testing_df.iterrows(), desc="Baseline"):
        numpy_images = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/feature/{example['id']}")
        batch_image = numpy_images.transpose(2,0,1)
        with torch.no_grad():
            input_image = torch.tensor(batch_image).unsqueeze(0).float().to(device)
            output_image = model(input_image)
            prediction = np.mean(get_tiles_congestion(output_image.cpu().numpy().squeeze()))
          
        testing_df.loc[i, "prediction"] = prediction
    testing_df = testing_df.sort_values(by="label")[["id", "label", "prediction"]]
    testing_df = testing_df.drop_duplicates(subset=["label"])
else:
    for i, example in tqdm(testing_df.iterrows(), desc="Test cases"):
        cur_msgs = []
        system_message = id_to_configs(example["id"])
        user_message = "Can you predict the congestion level of this sample from the given images?"
        image_id = example["id"]
        numpy_images = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/feature/{image_id}")
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
            last_hidden_state = output.hidden_states[-1][0]

            # Find the position of the gating token and extract embeddings
            gating_token_position = find_token_for_gating(
                input_ids[0].tolist(), processor.tokenizer.im_end_id
            )
            prompt_embedding = last_hidden_state[gating_token_position].float()
            last_token_embedding = last_hidden_state[-1].float()
            
            gating_weights = gating_network(prompt_embedding.unsqueeze(0))
            top_k = torch.topk(gating_weights.squeeze(), 5).indices
            multi_rewards = last_token_embedding.unsqueeze(0) @ regression_layer.T
            pred_val = torch.sum(multi_rewards * gating_weights, dim=-1).item()
        
        testing_df.loc[i, "prediction"] = pred_val
        for j in range(len(top_k)):
            testing_df.loc[i, f"top_{j}"] = f"{attributes[top_k[j]]}: {gating_weights.squeeze()[top_k[j]]}"
    testing_df = testing_df.sort_values(by="label")[["id", "label", "prediction","top_0","top_1","top_2","top_3","top_4"]]
    testing_df = testing_df.drop_duplicates(subset=["label"])


results = evaluate_design(testing_df)
print("Evaluation results:")
print(results)

testing_df.to_csv(f"/home/felixchaotw/mllm-physical-design/armo/results/{design_type}_results.csv", index=False)