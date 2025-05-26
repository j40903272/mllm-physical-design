import os
import torch
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoProcessor
from tqdm.auto import tqdm
from safetensors.torch import save_file
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from PIL import Image


# Set up CUDA optimizations for faster computation
torch.backends.cuda.matmul.allow_tf32 = (
    True  # Enable TensorFloat-32 matrix multiplication on CUDA
)
torch.backends.cudnn.allow_tf32 = (
    True  # Allow TensorFloat-32 in cuDNN for faster convolution operations
)

# Define token patterns for gating different model families
token_patterns = {
    # Llama3 token IDs of "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    "llama3": [128009, 128006, 78191, 128007, 271],
    # Gemma2 token IDs of "<end_of_turn>\n<start_of_turn>model\n"
    "gemma2": [107, 108, 106, 2516, 108],
}

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


def find_token_for_gating(lst, token_id):
    """Find the last occurrence of a token_pattern in a list."""
    search_end = len(lst)
    for j in range(search_end - 1, -1, -1):
        if lst[j] == token_id:
            return j
    raise ValueError("Token pattern not found in the list.")


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
    "--model_family", type=str, default="llama3", help="Model family (llama3 or gemma2)"
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default="/home/felixchaotw/mllm-physical-design/armo/dataset/preference_df.csv",
    help="Path to the dataset (HuggingFace path or local folder)",
)
parser.add_argument(
    "--source", default=None, type=str, help="Source filter for the dataset"
)
parser.add_argument(
    "--dataset_split", type=str, default="train", help="Dataset split to use"
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
parser.add_argument(
    "--seq_len", type=int, default=8192, help="Maximum sequence length for input"
)
args = parser.parse_args()  # Parse the provided command-line arguments

# Verify that the model family is passed correctly
config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

save_path = f"/home/felixchaotw/mllm-physical-design/armo/dataset/config_paired_embeddings"

# Load and prepare the dataset
ds = pd.read_csv(args.dataset_path)
ds = ds.sample(frac=0.5).reset_index(drop=True)
if args.source is not None:
    ds = ds.filter(lambda x: x["source"] == args.source)
if args.n_shards > 1:
    ds = ds.shuffle(seed=0)
    ds = ds.shard(num_shards=args.n_shards, index=args.shard_idx - 1)

# Load the pre-trained model and tokenizer
device = f"cuda:{args.device}"
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

# Initialize lists to store embeddings and prompt embeddings
embeddings = []
prompt_embeddings = []

# Process each example in the dataset
for _, example in tqdm(ds.iterrows(), desc="Processing dataset"):
    chosen_id = example["chosen"]
    rejected_id = example["rejected"]
    chosen_msgs = []
    rejected_msgs = []
    
    c_system_message = id_to_configs(example["chosen"])
    r_system_message = id_to_configs(example["rejected"])
    chosen_msgs.append(c_system_message)
    rejected_msgs.append(r_system_message)
    
    user_message = "Can you predict the congestion level of this design from the given images?"
    c_numpy_images = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/feature/{chosen_id}")
    r_numpy_images = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/congestion/feature/{rejected_id}")
    
    c_batch_image = c_numpy_images.transpose(2,0,1)
    r_batch_image = r_numpy_images.transpose(2,0,1)
    c_image_features = []
    r_image_features = []
    
    for image in c_batch_image:
        c_image_features.append(Image.fromarray(np.uint8(image * 255)))
        chosen_msgs.append("(<image>./</image>)")
        
    for image in r_batch_image:
        r_image_features.append(Image.fromarray(np.uint8(image * 255)))
        rejected_msgs.append("(<image>./</image>)")
    
    chosen_msgs.append(user_message)
    rejected_msgs.append(user_message)

    chosen = [
        {"role": "user", "content": "\n".join(chosen_msgs)},
    ]
    rejected = [
        {"role": "user", "content": "\n".join(rejected_msgs)},
    ]

    pair_embeddings = []
    pair_prompt_embeddings = []

    for iter_example in [(chosen, c_image_features), (rejected, r_image_features)]:
        msgs, image_features = iter_example
        # Format the conversation messages using the tokenizer's chat template without tokenization
        conv_formatted = processor.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )

        # Tokenize the formatted conversation and move tensors to the specified device
        conv_tokenized = processor([conv_formatted], image_features, return_tensors="pt").to(device)

        input_ids = conv_tokenized.data["input_ids"]
        position_ids = torch.arange(conv_tokenized.data["input_ids"].size(1)).long()
        conv_tokenized.data["position_ids"] = position_ids.unsqueeze(0).to(device)
        
        # We only have one sequence so batch size is 1
        if input_ids.shape[1] > args.seq_len:
            continue

        with torch.no_grad():
            output = model(data=conv_tokenized, output_hidden_states=True)
            last_hidden_state = output.hidden_states[-1][0]

            # Find the position of the gating token and extract embeddings
            gating_token_position = find_token_for_gating(
                input_ids[0].tolist(), processor.tokenizer.im_end_id
            )
            prompt_embedding = last_hidden_state[gating_token_position].cpu()
            last_token_embedding = last_hidden_state[-1].cpu()
            pair_embeddings.append(last_token_embedding)
            pair_prompt_embeddings.append(prompt_embedding)

    # Only add the pair if both chosen and rejected embeddings were successfully computed
    if len(pair_embeddings) == 2:
        embeddings.append(torch.stack(pair_embeddings))
        prompt_embeddings.append(torch.stack(pair_prompt_embeddings))

# Convert lists of embeddings to tensors
embeddings = torch.stack(embeddings)
prompt_embeddings = torch.stack(prompt_embeddings)

# Prepare the directory for saving embeddings
os.makedirs(os.path.dirname(save_path), exist_ok=True)
file_name = (
    f"{save_path}-{args.shard_idx:05d}-of-{args.n_shards:05d}"
    if args.n_shards > 1
    else save_path
)

# Save the embeddings and prompt embeddings using safetensors
save_file(
    {"embeddings": embeddings, "prompt_embeddings": prompt_embeddings},
    f"{save_path}.safetensors",
)

# Print a confirmation message with the path to the saved embeddings
print(f"Saved embeddings to {save_path}.safetensors")
