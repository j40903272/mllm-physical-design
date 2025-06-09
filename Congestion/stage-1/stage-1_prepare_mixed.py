import os
import torch
import datasets
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoProcessor
from tqdm.auto import tqdm
from safetensors.torch import save_file
from argparse import ArgumentParser
import torch
import torch.nn as nn
from PIL import Image


# Set up CUDA optimizations for faster computation
torch.backends.cuda.matmul.allow_tf32 = (
    True  # Enable TensorFloat-32 matrix multiplication on CUDA
)
torch.backends.cudnn.allow_tf32 = (
    True  # Allow TensorFloat-32 in cuDNN for faster convolution operations
)

# Define attributes (reward objectives)
attributes = [
 "total_wirelength",
 "number_vias",
 "number_of_multi_cut_vias",
 "number_of_single_cut_vias",
 "max_overcon",
 "total_overcon",
 "worst_layer_gcell_overcon_rate",
 "hard_to_access_pins_ratio",
 "instance_blockages_count",
 "early_gr_overflow_percentage",
 "horizontal_overflow_percentage",
 "congestion_prediction_accuracy",
 "initial_placement_efficiency",
 "area_based_congestion_density",
 "multi_layer_pin_access_variability",
 "average_layer_congestion",
 "pin_density_variance_map",
 "non_default_routing_rule_usage",
 "crosstalk_sensitive_zones",
 "inter_macro_channel_congestion",
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
        

# Initialize the argument parser to handle command-line inputs
parser = ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    default="openbmb/MiniCPM-V-2_6",
    help="Path to the pre-trained model (HuggingFace path or local folder)",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="./dataset/train_df_mixed.csv",
    help="Path to the dataset (HuggingFace path or local folder)",
)
parser.add_argument(
    "--n_shards",
    type=int,
    default=1,
    help="Total number of shards to divide the dataset into",
)
parser.add_argument(
    "--shard_idx", type=int, default=1, help="Index of the current shard"
)
parser.add_argument(
    "--device", type=int, default=0, help="CUDA device index to use for computation"
)
args = parser.parse_args()  # Parse the provided command-line arguments

ds = pd.read_csv(args.dataset_path)
ds = ds.sample(frac=1).reset_index(drop=True)

if args.n_shards > 1:
    ds = ds.shard(
        num_shards=args.n_shards, index=args.shard_idx - 1
    )  # Divide dataset into shards if needed

# Load the pre-trained model and tokenizer from the specified path
rm = AutoModel.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # Use bfloat16 precision for model weights to save memory
    attn_implementation="flash_attention_2",  # Specify the attention implementation for efficiency
)

device = f"cuda:{args.device}"  # Define the CUDA device string
rm = rm.to(device)  # Move the model to the specified CUDA device


processor = AutoProcessor.from_pretrained(
    args.model_path, 
    trust_remote_code=True
)

# Initialize lists to store embeddings and corresponding labels
embeddings = []
labels = []

# Iterate over each example in the dataset with a progress bar
for _, example in tqdm(ds.iterrows(), desc="Processing dataset"):
    cur_msgs = []
    system_message = id_to_configs(example["id"])
    print(system_message)
    user_message = "Can you predict the congestion level of this design from the given images?"
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
    position_ids = torch.arange(conv_tokenized.data["input_ids"].size(1)).long()
    conv_tokenized.data["position_ids"] = position_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        output = rm(data=conv_tokenized, output_hidden_states=True)  # Forward pass through the model
        # Extract the last hidden state of the last token and move it to CPU
        embeddings.append(output.hidden_states[-1][0][-1].cpu())

    # Extract labels for the current example based on predefined attributes
    label = [example[attr] for attr in attributes]
    # Replace None values with NaN for consistent numerical processing
    label = [np.nan if l is None else l for l in label]
    labels.append(label)  # Append the processed labels to the list

# Convert the list of labels to a NumPy array with float32 precision
labels = np.array(labels, dtype=np.float32)
labels = torch.from_numpy(labels)  # Convert the NumPy array to a PyTorch tensor
embeddings = torch.stack(embeddings, dim=0)  # Stack all embeddings into a single tensor

# Define the path to save the embeddings and labels
HOME = os.path.expanduser("~")  # Get the home directory of the current user
model_name = args.model_path.split("/")[
    -1
]  # Extract the model name from the model path
dataset_name = args.dataset_path.split("/")[
    -1
]  # Extract the dataset name from the dataset path
save_path = "/home/felixchaotw/mllm-physical-design/armo/dataset/"
os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

# Save the embeddings and labels in a safetensors file with shard indexing
save_file(
    {"embeddings": embeddings, "labels": labels},
    f"{save_path}train_log.safetensors",
)

# Print a confirmation message with the path to the saved embeddings
print(
    f"Saved embeddings to {save_path}train_log.safetensors"
)
