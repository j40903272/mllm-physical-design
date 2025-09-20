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
from PIL import Image
from scipy import stats
from sklearn.metrics import ndcg_score
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoProcessor

# Enable TF32 for improved performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Define the attributes for multi-objective reward modeling
attributes = [
 'max_congestion_ripple',
 'macro_interference_zone',
 'macro_compactness_index',
 'cell_density_variance_gradient',
 'mean_macro_proximity',
 'congestion_gradient',
 'cell_density_anisotropy',
 'mean_eGR_local_variability',
 'diagonal_cell_density_gradient',
 'mean_cell_density_fluctuation',
 'macro_transition_band',
 'cell_density_skewness',
 'cell_density_skewness_gradient',
 'macro_interaction_perimeter',
 'cell_density_fluctuation_balance',
 'congestion_pressure_fluctuation',
 'congestion_variability_throughout_hierarchy',
 'congestion_transition_amplitude',
 'cell_density_dipole',
 'mean_eGR_local_adjacent_cohesion',
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


def find_proper_verbosity_penalties(cluster_V, verbosity_dim=4, corr_threshold=0.028):
    """
    Find appropriate penalties for verbosity to reduce its correlation with other dimensions
    """
    verbosity_penalties = [
        0,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.125,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ]
    verbosity_penalties = sorted(verbosity_penalties)
    K = cluster_V.shape[1]
    candidate_dims = set(range(K))
    candidate_dims.remove(verbosity_dim)
    dimwise_verbosity_penalties = np.ones(K)
    dimwise_corr = np.ones(K)
    for verbosity_penalty in verbosity_penalties:
        if len(candidate_dims) == 0:
            break
        V_adjusted = cluster_V - verbosity_penalty * cluster_V[:, [verbosity_dim]]
        corrs = {
            i: spearmanr(V_adjusted[:, i], cluster_V[:, verbosity_dim])[0]
            for i in candidate_dims
        }
        for dim, corr in corrs.items():
            if corr <= corr_threshold:
                candidate_dims.remove(dim)
                dimwise_verbosity_penalties[dim] = verbosity_penalty
                dimwise_corr[dim] = corr
            else:
                dimwise_corr[dim] = np.min([dimwise_corr[dim], corr])
        if len(candidate_dims) == 0:
            break
    return {"penalty": dimwise_verbosity_penalties, "corr": dimwise_corr}

def find_token_for_gating(lst, token_id):
    """Find the last occurrence of a token_pattern in a list."""
    search_end = len(lst)
    for j in range(search_end - 1, -1, -1):
        if lst[j] == token_id:
            return j
    raise ValueError("Token pattern not found in the list.")


def evalute_corr(congestion_set, predicted, corr_metrics):
    x = np.array(list(congestion_set.values()))
    x_label = list(congestion_set.keys())
    y = np.array([predicted[id] for id in x_label])
    results = {}
    if "PLCC" in corr_metrics:
        results["PLCC"] = stats.pearsonr(x, y)
    if "SRCC" in corr_metrics:
        results["SRCC"] = stats.spearmanr(x, y)
    if "KRCC" in corr_metrics:
        results["KRCC"] = stats.kendalltau(x, y)
    
    return results


def evaluate_design(df):
    congestion_set = dict(zip(df["id"], df["label"]))
    congestion_set = dict(sorted(congestion_set.items(), key=lambda x: x[1]))
    predicted = dict(zip(df["id"], df["prediction"]))
    corr_metrics = ["PLCC", "SRCC", "KRCC"]
    results = evalute_corr(congestion_set, predicted, corr_metrics)
    return results


def validation(testing_df, model, processor, device, gating_network, regression_layer):
    for i, example in tqdm(testing_df.iterrows(), desc="Test cases"):
        cur_msgs = []
        system_message = id_to_configs(example["id"])
        user_message = "Can you predict the congestion level of this sample from the given images?"
        image_id = example["id"]
        numpy_images = np.load(f"/data2/NVIDIA/CircuitNet-N28/Dataset/DRC/feature/{image_id}")
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
    
    results = evaluate_design(testing_df)        
    return results


def eval_reward_bench(df_examples, acc_column="correct"):
    """
    Evaluate the model on the RewardBench dataset
    """
    categories = {
        "chat": [
            "alpacaeval-easy",
            "alpacaeval-length",
            "alpacaeval-hard",
            "mt-bench-easy",
            "mt-bench-med",
        ],
        "chat-hard": [
            "mt-bench-hard",
            "llmbar-natural",
            "llmbar-adver-neighbor",
            "llmbar-adver-GPTInst",
            "llmbar-adver-GPTOut",
            "llmbar-adver-manual",
        ],
        "safety": [
            "refusals-dangerous",
            "refusals-offensive",
            "xstest-should-refuse",
            "xstest-should-respond",
            "donotanswer",
        ],
        "reasoning": [
            "math-prm",
            "hep-cpp",
            "hep-go",
            "hep-java",
            "hep-js",
            "hep-python",
            "hep-rust",
        ],
    }

    df_acc = pd.DataFrame(columns=["category", "subset", "accuracy"])
    for category, subsets in categories.items():
        for subset in subsets:
            df_subset = df_examples[df_examples["subset"] == subset]
            acc = df_subset[acc_column].values.mean()
            row = {
                "category": category,
                "subset": subset,
                "n": len(df_subset),
                "accuracy": [acc],
            }
            df_acc = pd.concat([df_acc, pd.DataFrame(row)], ignore_index=True)

    EXAMPLE_COUNTS = {
        "alpacaeval-easy": 100,
        "alpacaeval-length": 95,
        "alpacaeval-hard": 95,
        "mt-bench-easy": 28,
        "mt-bench-med": 40,
        "mt-bench-hard": 37,
        "math-prm": 984,
        "refusals-dangerous": 100,
        "refusals-offensive": 100,
        "llmbar-natural": 100,
        "llmbar-adver-neighbor": 134,
        "llmbar-adver-GPTInst": 92,
        "llmbar-adver-GPTOut": 47,
        "llmbar-adver-manual": 46,
        "xstest-should-refuse": 250,
        "xstest-should-respond": 154,
        "donotanswer": 136,
        "hep-cpp": 164,
        "hep-go": 164,
        "hep-java": 164,
        "hep-js": 164,
        "hep-python": 164,
        "hep-rust": 164,
    }

    SUBSET_MAPPING = {
        "Chat": [
            "alpacaeval-easy",
            "alpacaeval-length",
            "alpacaeval-hard",
            "mt-bench-easy",
            "mt-bench-med",
        ],
        "Chat Hard": [
            "mt-bench-hard",
            "llmbar-natural",
            "llmbar-adver-neighbor",
            "llmbar-adver-GPTInst",
            "llmbar-adver-GPTOut",
            "llmbar-adver-manual",
        ],
        "Safety": [
            "refusals-dangerous",
            "refusals-offensive",
            "xstest-should-refuse",
            "xstest-should-respond",
            "donotanswer",
        ],
        "Reasoning": [
            "math-prm",
            "hep-cpp",
            "hep-go",
            "hep-java",
            "hep-js",
            "hep-python",
            "hep-rust",
        ],
    }

    all_subsets = df_examples["subset"].unique()

    metrics = {}
    for subset in all_subsets:
        df_subset = df_acc.loc[df_acc["subset"] == subset]
        acc = df_subset["accuracy"].values[0]
        metrics[subset] = acc

    scores_per_section = calculate_scores_per_section(
        EXAMPLE_COUNTS, SUBSET_MAPPING, metrics
    )
    score_weights = {"Chat": 1, "Chat Hard": 1, "Safety": 1, "Reasoning": 1}
    scores_per_section["Score"] = round(
        sum([v * score_weights[k] for k, v in scores_per_section.items()])
        / sum(score_weights.values()),
        2,
    )
    return scores_per_section, metrics


def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    """
    Calculate scores for each section of the RewardBench
    """
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                total_weighted_score += metrics[test] * example_counts[test]
                total_examples += example_counts[test]
        if total_examples > 0:
            section_scores[section] = 100 * total_weighted_score / total_examples
        else:
            section_scores[section] = 0
    return section_scores


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
parser.add_argument("--model_path", type=str, default="/data1/felixchao/minicpm")
parser.add_argument(
    "--multi_objective_dataset",
    type=str,
    default="RLHFlow/ArmoRM-Multi-Objective-Data-v0.1",
)
parser.add_argument(
    "--preference_dataset", type=str, default="RLHFlow/pair_data_v2_80K_wsafety"
)
parser.add_argument(
    "--reference_dataset",
    type=str,
    default=None,
)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--n_steps", type=int, default=500000)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument(
    "--verbosity_dim", type=int, default=4, help="Dimension of the verbosity attribute"
)
parser.add_argument(
    "--corr_threshold",
    type=float,
    default=0.03,
    help="Correlation threshold for verbosity penalty",
)
parser.add_argument("--model_family", type=str, default="llama3", help="Model family")
parser.add_argument(
    "--eval_reward_bench", action="store_true", help="Evaluate on RewardBench"
)
parser.add_argument("--logit_scale", type=float, default=1)
parser.add_argument("--temperature", type=float, default=10)
parser.add_argument("--n_hidden", type=int, default=3)
parser.add_argument("--hidden_size", type=int, default=1024)
parser.add_argument("--dropout", type=float, default=0.2)
args = parser.parse_args()

# Define default paths
HOME = os.path.expanduser("~")

if args.reference_dataset is None:
    args.reference_dataset = args.preference_dataset
    print(
        f"Using {args.preference_dataset} as the reference dataset for verbosity debiasing."
    )

args.model_name = args.model_path.split("/")[-1]
args.multi_objective_dataset_name = args.multi_objective_dataset.split("/")[-1]
args.preference_dataset_name = args.preference_dataset.split("/")[-1]
args.reference_dataset_name = args.reference_dataset.split("/")[-1]

args.embedding_path = f"/home/felixchaotw/mllm-physical-design/DRV/dataset/config_paired_embeddings.safetensors"
args.regression_layer_path = f"/home/felixchaotw/mllm-physical-design/DRV/regression_weights/MiniCPM-V-2_6_ArmoRM-Multi-Objective-Data-v0.1.pt"
args.reward_bench_embedding_path = (
    f"{HOME}/data/ArmoRM/embeddings/{args.model_name}/reward-bench-filtered.safetensors"
)

device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

# Print the paths for verification
print(f"Embedding path: {args.embedding_path}")
print(f"Regression layer path: {args.regression_layer_path}")
print(f"Reward bench embedding path: {args.reward_bench_embedding_path}")

# Load embeddings
print("Loading embeddings...")
embeddings, prompt_embeddings = load_embeddings(args.embedding_path, device=device)

print(embeddings.shape, prompt_embeddings.shape)

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

# Load the testing dataset
print("Loading testing dataset...")

design_type = "zero-riscy-a"

if design_type == "zero-riscy-a":
    testing_path = "/home/felixchaotw/mllm-physical-design/DRV/dataset/test_df_a.csv"
else:
    testing_path = f"/home/felixchaotw/mllm-physical-design/DRV/dataset/test_df_b.csv"
    
testing_df = pd.read_csv(testing_path)
testing_df = testing_df[testing_df["label"].notna()]

# # Load reference dataset embeddings
# embedding_path = f"/home/felixchaotw/mllm-physical-design/armo/dataset/paired_embeddings.safetensors"
# ref_embeddings, ref_prompt_embeddings = load_embeddings(embedding_path, device=device)

# # Calculate pairwise rewards and rewards difference
# pairwise_rewards = ref_embeddings @ regression_layer.T
# rewards = pairwise_rewards.reshape(-1, pairwise_rewards.shape[-1])
# rewards_diff = pairwise_rewards[:, 0] - pairwise_rewards[:, 1]

# # Find proper verbosity penalties
# penalties = find_proper_verbosity_penalties(
#     rewards.cpu().numpy().reshape(-1, n_attributes),
#     verbosity_dim=args.verbosity_dim,
#     corr_threshold=args.corr_threshold,
# )
# print("Penalties:", penalties)

# # Create reward transform matrix
# reward_transform_matrix = torch.eye(n_attributes)
# reward_transform_matrix[args.verbosity_dim, :] -= torch.from_numpy(penalties["penalty"])
# reward_transform_matrix = reward_transform_matrix.to(device)

# Prepare data for training
X = prompt_embeddings  # condition for gating network
Z = embeddings  # multi-objective rewards
# R = embeddings @ regression_layer.T @ reward_transform_matrix  # multi-objective rewards
# # Split train/val
# X_train, X_val, Z_train, Z_val, R_train, R_val = train_test_split(
#     X, Z, R, test_size=0.2, random_state=0
# )
X_train, Z_train = X, Z

# Initialize gating network
print("Initializing gating network...")
gating_network = GatingNetwork(
    X_train.shape[-1],
    regression_layer.shape[0],
    n_hidden=args.n_hidden,
    hidden_dim=args.hidden_size,
    logit_scale=args.logit_scale,
    temperature=args.temperature,
    dropout=args.dropout,
).to(device)

# Define loss function and optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    gating_network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_steps)

max_acc = 0.0
# Training loop
print("Starting training...")
for step in tqdm(range(args.n_steps)):
    optimizer.zero_grad()

    # Sample batch
    idx = torch.randint(0, X_train.shape[0], (args.batch_size,))
    X_batch = X_train[idx]
    Z_batch = Z_train[idx]
    
    # Forward pass
    gating_weights = gating_network(X_batch)
    pred = torch.sum(Z_batch @ regression_layer.T * gating_weights, dim=-1)

    # Compute loss
    loss = loss_fn(pred[:, 0] - pred[:, 1], torch.ones_like(pred[:, 0]))

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
        # Evaluation
        print("Evaluating model...")
        gating_network.eval()
        results = validation(testing_df, model, processor, device, gating_network, regression_layer)
        print(f"Validation results: {results}")
        
        if results["SRCC"].statistic > max_acc:
            max_acc = results["SRCC"].statistic
            # Save the trained gating network
            save_path = f"/home/felixchaotw/mllm-physical-design/DRV/gating_weights/config_gating_network_{args.model_name}.pt"
            torch.save(gating_network.state_dict(), save_path)
            print(f"Saved gating network to {save_path}")
            
        gating_network.train()