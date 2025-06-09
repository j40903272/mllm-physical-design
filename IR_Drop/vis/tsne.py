from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
import re


test_a = pd.read_csv("/home/felixchaotw/mllm-physical-design/armo/dataset/test_feature_desc_a.csv")
test_b = pd.read_csv("/home/felixchaotw/mllm-physical-design/armo/dataset/test_feature_desc_b.csv")


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


for i, example in tqdm(test_a.iterrows()):
    prompt = example["prompt"]
    str_id = example["id"].split("-")[0]
    
    for attr in attributes:
        res = re.search(rf"{attr} is ([-\d.]+)", prompt)
        test_a.loc[i, attr] = float(res.group(1)) if res else 0.0
        
    cong = re.search(r"Congestion level: ([-\d.]+)\.", prompt)
    test_a.loc[i, "label"] = float(cong.group(1)) if cong else 0.0
    test_a.loc[i, "id"] = str_id     
        
for i, example in tqdm(test_b.iterrows()):
    prompt = example["prompt"]
    str_id = example["id"].split("-")[0]
    
    for attr in attributes:
        res = re.search(rf"{attr} is ([-\d.]+)", prompt)
        test_b.loc[i, attr] = float(res.group(1)) if res else 0.0
    
    cong = re.search(r"Congestion level: ([-\d.]+)\.", prompt)
    test_b.loc[i, "label"] = float(cong.group(1)) if cong else 0.0
    test_b.loc[i, "id"] = str_id


tsne = TSNE(n_components=2, random_state=501)
X_b = test_b[attributes].values
tsne_b = tsne.fit_transform(X_b)
test_b["tsne_1"] = tsne_b[:, 0]
test_b["tsne_2"] = tsne_b[:, 1]


plt.figure(figsize=(64, 30))
plt.scatter(
    test_b["tsne_1"],
    test_b["tsne_2"],
    c=test_b["label"],
    cmap="plasma",
    s = 60,
)

for line in range(0,test_b.shape[0]):
     plt.text(
          test_b["tsne_1"][line]+0.2,
          test_b["tsne_2"][line],
          test_b["id"][line],
          ha='left',
          fontsize=6,
     )
     
plt.title("t-SNE visualization of test_b")
plt.savefig("test_b_tsne.png")

test_b.to_csv("test_b_tsne.csv", index=False)

