from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm.auto import tqdm


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

train_df = pd.read_csv("/home/felixchaotw/mllm-physical-design/armo/dataset/train_feature_desc.csv")
gt_df = pd.read_csv("/home/felixchaotw/mllm-physical-design/armo/dataset/train_df.csv")


for i, example in tqdm(train_df.iterrows()):
    prompt = example["prompt"]
    gt = gt_df[gt_df["id"] == example["id"]]["label"].values[0]
    
    for attr in attributes:
        res = re.search(rf"{attr} is ([-\d.]+), the importance is ([-\d.]+)\.", prompt)
        train_df.loc[i, attr] = float(res.group(1)) if res else 0.0
        train_df.loc[i, f"{attr}_weight"] = float(res.group(2)) if res else 0.0
        
        
    cong = re.search(r"Congestion level: ([-\d.]+)\.", prompt)
    train_df.loc[i, "label"] = float(cong.group(1)) if cong else 0.0
    train_df.loc[i, "gt"] = gt


train_df = train_df.drop(columns=["prompt","config"])
train_df = train_df[train_df["gt"].notna()].reset_index(drop=True)


neigh = NearestNeighbors(radius=1.6)
X = train_df[attributes].values
y = train_df["gt"].values
neigh.fit(X, y)


test_id = "216-RISCY-a-1-c2-u0.85-m3-p1-f1.npy"
test_df = train_df[train_df["id"] == test_id]

x = test_df[attributes].values
rng = neigh.radius_neighbors(x, radius=0.7, sort_results=True)
index = np.asarray(rng[1][0])
nn = train_df.loc[index, ["id", "gt", "label"] + attributes]
nn = nn.reset_index(drop=True)
nn.to_csv(f"/home/felixchaotw/mllm-physical-design/armo/vis/nearest_neighbors.csv", index=False)

# important_attributes = ["rudy_pin_gradient_convergence", "rudy_pin_clustering_coefficient"]
important_attributes = ["rudy_pin_gradient_convergence"]
# important_attributes = ["macro_rudy_boundary_interaction_index"]


total_gt = 0.0
total_pred = 0.0
count = 0

for i, example in tqdm(nn.iterrows()):
    if example["id"] != test_id:
        is_lower = True
        for attr in important_attributes:
            if example[attr] >= test_df[attr].values[0]:
                is_lower = False
                break
            
        if is_lower:
            total_gt += example["gt"]
            total_pred += example["label"]
            print(f"GT: {example['gt']}, Pred: {example['label']}, ID: {example['id']}")
            count += 1
            
total_gt /= count
total_pred /= count
print(f"Total GT: {total_gt}")
print(f"Total Pred: {total_pred}")
        

    
    



