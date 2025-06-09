from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
import re


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
    str_id = example["id"].split("-")[0]
    
    for attr in attributes:
        res = re.search(rf"{attr} is ([-\d.]+), the importance is ([-\d.]+)\.", prompt)
        train_df.loc[i, attr] = float(res.group(1)) if res else 0.0
        train_df.loc[i, f"{attr}_weight"] = float(res.group(2)) if res else 0.0
        
        
    cong = re.search(r"Congestion level: ([-\d.]+)\.", prompt)
    train_df.loc[i, "label"] = float(cong.group(1)) if cong else 0.0
    train_df.loc[i, "gt"] = gt
    train_df.loc[i, "id"] = str_id     


train_df = train_df.drop(columns=["prompt"])
train_df = train_df[train_df["gt"].notna()]


fig, axs = plt.subplots(5, 5, figsize=(50, 50))

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)


for i in range(len(attributes)):
    r = i // 5
    c = i % 5
    axs[r, c].set_title(attributes[i])
    axs[r, c].set_xlabel("value")
    axs[r, c].set_ylabel("gating weights")
    sns.scatterplot(
        x=train_df[attributes[i]],
        y=train_df[f"{attributes[i]}_weight"],
        hue=train_df["gt"],
        size=train_df["gt"],
        palette=cmap,
        ax=axs[r, c],
    )


plt.savefig("train_feature_gt.png", bbox_inches='tight', dpi=300)